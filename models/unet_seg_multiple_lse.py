import torch
import torch.nn as nn 
#from model_resnet_rs import ResnetRS
from functools import partial
from typing import List
from torchvision import models

from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.layers.squeeze_excite import EffectiveSEModule
from timm.models import create_model,register_model, build_model_with_cfg, named_apply, generate_default_cfgs
from timm.models.layers import DropPath
from timm.models.layers import LayerNorm2d

from timm.layers.squeeze_excite import EffectiveSEModule
#from test_se.layers import DepthToSpace2DWithConv
import torch.nn.functional as F
#from  torchvision.models.resnet import BasicBlock
from dynamic_network_architectures.architectures.unet import ResidualEncoderUNet


def log_sum_exp(out, r):
    if len(out.size()) > 3:
        out = out.flatten(start_dim=2)
    out = torch.permute(out,(0,2,1))
    b, c, w = out.size()
    multi = r* torch.ones_like(out)
    out_max = torch.max(out, dim=1)[0].unsqueeze(1)
    out =     out_max.squeeze(1) + torch.div( torch.log(torch.mean(torch.exp(multi*(out-out_max)),dim=1)), torch.mean(1e-8+ multi,dim=1))
            
    return out


class LOG_SUM_EXP(nn.Module):
    def __init__(
        self,            
        r: float = 3.,
        num_classes: int = 6,
    ):
        super().__init__()
        self.bias = nn.Parameter(torch.zeros((num_classes)), requires_grad=True)
        self.check = nn.Identity()
        self.r = r
    def forward(self, x):
        _, c , _ ,_ = x.size()
        x = x - self.bias[None,:, None,None]
        x = self.check(x)
        final = log_sum_exp(x, self.r)

        return final



class SegRDNet(nn.Module):
    def __init__(
        self,
        num_classes = 6,
        
        ):
        super().__init__()
        
        self.unet = ResidualEncoderUNet(
        input_channels=3,
        n_stages=8,
        features_per_stage=[32, 64, 128, 256, 512, 512, 512, 512],
        conv_op=torch.nn.Conv2d,
        kernel_sizes=[[3, 3] for _ in range(8)],
        strides=[[1, 1], [2, 2], [2, 2], [2, 2], [2, 2], [2, 2], [2, 2], [2, 2]],
        n_blocks_per_stage=[2, 2, 2, 2, 2, 2, 2, 2],
        num_classes=num_classes,
        n_conv_per_stage_decoder=[2, 2, 2, 2, 2, 2, 2],
        conv_bias=True,
        norm_op=LayerNorm2d,
        norm_op_kwargs={"eps": 1e-5, "affine": True},
        nonlin=torch.nn.LeakyReLU,
        nonlin_kwargs={"inplace": True},
        deep_supervision=True,
    )
        print('use unet multiple lse')
        self.lse = nn.ModuleList([LOG_SUM_EXP(num_classes = num_classes) for _ in range(7)] )
               
    def forward(self, x, train= False):

        outputs = self.unet(x)

        L = len(outputs)
        assert L == 7
        mse_loss = []
        
        for i in range(1,L):
            #print(i, outputs[i-1].size()[2:])
            curr = F.mse_loss(F.adaptive_avg_pool2d(outputs[i-1], outputs[i].size()[2:]) , outputs[i] )
            mse_loss.append(curr)

        lse_out = []

        for i in range(L):
            lse_out+= [self.lse[i](outputs[i])]
        #print(len(lse_out), len(mse_loss), outputs[0].size(), outputs[1].size(), outputs[-1].size())
        if train:
            return lse_out, mse_loss
        else:
            return lse_out[0]
            
'''        
model = SegRDNet()

print(sum(p.numel() for p in model.parameters() if p.requires_grad))
'''
