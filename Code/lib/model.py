import torch
import torch.nn as nn
import torchvision.models as models
from torch.nn import functional as F
#from Code.lib.res2net_v1b_depth import Res2Net_model #as Res2Net_depth
from Code.lib.res2net_gba import GBARES
from Code.lib.cbam import CBAM

def maxpool():
    pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
    return pool


def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x


class GCM0(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(GCM0, self).__init__()
        self.relu = nn.ReLU(True)
        self.branch0 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
        )
        self.branch1 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 3), padding=(0, 1)),
            BasicConv2d(out_channel, out_channel, kernel_size=(3, 1), padding=(1, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=3, dilation=3)
        )
        self.branch2 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 5), padding=(0, 2)),
            BasicConv2d(out_channel, out_channel, kernel_size=(5, 1), padding=(2, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=5, dilation=5)
        )
        self.branch3 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 7), padding=(0, 3)),
            BasicConv2d(out_channel, out_channel, kernel_size=(7, 1), padding=(3, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=7, dilation=7)
        )
        self.conv_cat = BasicConv2d(4*out_channel, out_channel, 3, padding=1)
        self.conv_res = BasicConv2d(in_channel, out_channel, 1)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)

        x_cat = self.conv_cat(torch.cat((x0, x1, x2, x3), 1))

        x = self.relu(x_cat + self.conv_res(x))
        return x


#Global Contextual module
class GCM(nn.Module):
    def __init__(self, out_channel):
        super(GCM, self).__init__()
        self.relu = nn.ReLU(True)
        self.branch0 = nn.Sequential(
            BasicConv2d(out_channel, out_channel, 1),
        )
        self.branch1 = nn.Sequential(
            #BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 3), padding=(0, 1)),
            BasicConv2d(out_channel, out_channel, kernel_size=(3, 1), padding=(1, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=3, dilation=3)
        )
        self.branch2 = nn.Sequential(
            #BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 5), padding=(0, 2)),
            BasicConv2d(out_channel, out_channel, kernel_size=(5, 1), padding=(2, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=5, dilation=5)
        )
        self.branch3 = nn.Sequential(
            #BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 7), padding=(0, 3)),
            BasicConv2d(out_channel, out_channel, kernel_size=(7, 1), padding=(3, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=7, dilation=7)
        )
        self.conv_cat = BasicConv2d(4*out_channel, out_channel, 3, padding=1)
        self.conv_res = BasicConv2d(out_channel, out_channel, 1)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)

        x_cat = self.conv_cat(torch.cat((x0, x1, x2, x3), 1))

        x = self.relu(x_cat + self.conv_res(x))
        return x



###############################################################################

class CDA0(nn.Module):    
    def __init__(self,in_dim, out_dim):
        super(CDA0, self).__init__()
        
        act_fn = nn.ReLU(inplace=True)
        

        self.layer_10 = nn.Conv2d(out_dim, out_dim, kernel_size=3, stride=1, padding=1)
        self.layer_20 = nn.Conv2d(out_dim, out_dim, kernel_size=3, stride=1, padding=1)   
        
        self.rgb_d = CBAM(out_dim, 1)
        self.d_rgb = CBAM(out_dim, 1)

        self.layer_ful1 = nn.Sequential(nn.Conv2d(out_dim*2, out_dim, kernel_size=3, stride=1, padding=1),nn.BatchNorm2d(out_dim),act_fn,)
        

    def forward(self, rgb, depth):
        
        ################################
        
        x_rgb1 = self.layer_10(rgb)
        x_dep1 = self.layer_20(depth)
        
        ## fusion 
        x_dep_r = self.rgb_d(x_rgb1, x_dep1)
        x_rgb_r = self.d_rgb(x_dep1, x_rgb1)

        x_cat = torch.cat((x_rgb_r, x_dep_r),dim=1)
        out1 = self.layer_ful1(x_cat)
        #out1 = x_rgb1 + x_dep1
        return out1


class CDA(nn.Module):    
    def __init__(self,in_dim, out_dim):
        super(CDA, self).__init__()
        act_fn = nn.ReLU(inplace=True)
        
        self.reduc_1 = nn.Sequential(nn.Conv2d(in_dim, out_dim, kernel_size=1), act_fn)
        self.reduc_2 = nn.Sequential(nn.Conv2d(in_dim, out_dim, kernel_size=1), act_fn)
        
        self.layer_10 = nn.Conv2d(out_dim, out_dim, kernel_size=3, stride=1, padding=1)
        self.layer_20 = nn.Conv2d(out_dim, out_dim, kernel_size=3, stride=1, padding=1)

        self.rgb_d = CBAM(out_dim, 1)
        self.d_rgb = CBAM(out_dim, 1)

        self.layer_ful1 = nn.Sequential(nn.Conv2d(out_dim*2, out_dim, kernel_size=3, stride=1, padding=1),nn.BatchNorm2d(out_dim),act_fn,)
        self.layer_ful2 = nn.Sequential(nn.Conv2d(out_dim+out_dim//2, out_dim, kernel_size=3, stride=1, padding=1),nn.BatchNorm2d(out_dim),act_fn,)


    def forward(self, rgb, depth, xx):
        
        ################################
        x_rgb = self.reduc_1(rgb)
        x_dep = self.reduc_2(depth)
        
        x_rgb1 = self.layer_10(x_rgb)
        x_dep1 = self.layer_20(x_dep)
        
        x_dep_r = self.rgb_d(x_rgb1, x_dep1)
        x_rgb_r = self.d_rgb(x_dep1, x_rgb1)

        x_cat   = torch.cat((x_rgb_r, x_dep_r),dim=1)
        out1 = self.layer_ful1(x_cat)
        out2 = self.layer_ful2(torch.cat([out1,xx],dim=1))
        #out2 = out1 + xx
        return out2



class AttFusion(nn.Module):    
    def __init__(self,in_dim, out_dim):
        super(AttFusion, self).__init__()
        act_fn = nn.ReLU(inplace=True)

        self.reduc_1 = nn.Sequential(nn.Conv2d(in_dim, out_dim, kernel_size=1), act_fn)
        #self.reduc_2 = nn.Sequential(nn.Conv2d(in_dim, out_dim, kernel_size=1), act_fn)

        self.layer_10 = nn.Conv2d(out_dim, out_dim, kernel_size=3, stride=1, padding=1)
        self.layer_20 = nn.Conv2d(out_dim, out_dim, kernel_size=3, stride=1, padding=1)

        self.rgb_d = CBAM(out_dim, 1)
        self.d_rgb = CBAM(out_dim, 1)

        self.layer_ful1 = nn.Sequential(nn.Conv2d(out_dim*2, out_dim, kernel_size=3, stride=1, padding=1),nn.BatchNorm2d(out_dim),act_fn,)


    def forward(self, rgb, depth):

        ################################
        x_rgb = self.reduc_1(rgb)
        x_dep = depth
        
        x_rgb1 = self.layer_10(x_rgb)
        x_dep1 = self.layer_20(x_dep)
        
        x_dep_r = self.rgb_d(x_rgb1, x_dep1)
        x_rgb_r = self.d_rgb(x_dep1, x_rgb1)

        x_cat   = torch.cat((x_rgb_r, x_dep_r),dim=1)
        out1 = self.layer_ful1(x_cat)
        #out1 = x_rgb1 + x_dep1
        return out1


class CDA_last(nn.Module):    
    def __init__(self,in_dim, out_dim):
        super(CDA_last, self).__init__()
        act_fn = nn.ReLU(inplace=True)

        #self.reduc_1 = nn.Sequential(nn.Conv2d(in_dim, out_dim, kernel_size=1), act_fn)
        #self.reduc_2 = nn.Sequential(nn.Conv2d(in_dim, out_dim, kernel_size=1), act_fn)

        self.layer_10 = nn.Conv2d(out_dim, out_dim, kernel_size=3, stride=1, padding=1)
        self.layer_20 = nn.Conv2d(out_dim, out_dim, kernel_size=3, stride=1, padding=1)

        self.rgb_d = CBAM(out_dim, 1)
        self.d_rgb = CBAM(out_dim, 1)

        #self.reduc_11 = nn.Sequential(nn.Conv2d(out_dim, 256, kernel_size=1), act_fn)
        #self.reduc_22 = nn.Sequential(nn.Conv2d(out_dim, 256, kernel_size=1), act_fn)

        #self.fusion = FeatureFusionNetwork(d_model=256,nhead=8, num_featurefusion_layers=4, dim_feedforward=2048, dropout=0.1, activation="relu" )
        #self.up = nn.Sequential(nn.Conv2d(256, out_dim, kernel_size=1), act_fn)

        self.layer_ful1 = nn.Sequential(nn.Conv2d(out_dim*3, out_dim, kernel_size=3, stride=1, padding=1),nn.BatchNorm2d(out_dim),act_fn,)
        self.layer_ful2 = nn.Sequential(nn.Conv2d(out_dim+out_dim//2, out_dim, kernel_size=3, stride=1, padding=1),nn.BatchNorm2d(out_dim),act_fn,)


    def forward(self, rgb, depth, fus, xx):

        ################################
        #x_rgb = self.reduc_1(rgb)
        #x_dep = self.reduc_2(depth)
        x_rgb = rgb
        x_dep = depth

        x_rgb1 = self.layer_10(x_rgb)
        x_dep1 = self.layer_20(x_dep)

        out2 = x_dep1 + x_dep1 + xx
        return out2


class EMI0(nn.Module):    
    def __init__(self,in_dim):
        super(EMI0, self).__init__()
         
        self.relu = nn.ReLU(inplace=True)
        
        self.layer_10 = nn.Conv2d(in_dim, in_dim, kernel_size=3, stride=1, padding=1)
        self.layer_20 = nn.Conv2d(in_dim, in_dim, kernel_size=3, stride=1, padding=1)   
        self.layer_cat1 = nn.Sequential(nn.Conv2d(in_dim*2, in_dim, kernel_size=3, stride=1, padding=1),nn.BatchNorm2d(in_dim),)        
        
    def forward(self, x_ful, x1, x2):
        
        ################################
    
        x_ful_1 = x_ful.mul(x1)
        x_ful_2 = x_ful.mul(x2)
        
     
        x_ful_w = self.layer_cat1(torch.cat([x_ful_1, x_ful_2],dim=1))
        out     = self.relu(x_ful + x_ful_w)
        
        return out
    
    
class EMI(nn.Module):    
    def __init__(self,in_dim):
        super(EMI, self).__init__()

        self.relu = nn.ReLU(inplace=True)

        #self.layer_10 = nn.Conv2d(in_dim, in_dim, kernel_size=3, stride=1, padding=1)
        #self.layer_20 = nn.Conv2d(in_dim, in_dim, kernel_size=3, stride=1, padding=1)   
        self.layer_cat1 = nn.Sequential(nn.Conv2d(in_dim*3, in_dim, kernel_size=3, stride=1, padding=1),nn.BatchNorm2d(in_dim),)        
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        k_size = 3
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False) 
        self.sigmoid = nn.Sigmoid()


    def forward(self, x_ful, x1, x2):

        ################################
        x = self.layer_cat1(torch.cat([x1, x2, x_ful],dim=1))

        #x = self.layer_cat1(torch.cat([x1, x2, x_ful],dim=1))
        y = self.avg_pool(x)

        # Two different branches of ECA module
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)

        # Multi-scale information fusion
        y = self.sigmoid(y)

        #ful = x1.mul(x2) #.mul(x_ful)
        #out = ful.mul(x)
        #out = ful.mul(x)


        #x_ful_w = self.layer_cat1(torch.cat([x_ful_1, x_ful_2],dim=1))
        out     = self.relu(x_ful + x * y.expand_as(x))
        #out = x_ful + x1 + x2
        return out

  
   
###############################################################################

class HiDANet(nn.Module):
    def __init__(self, channel=32):
        super(HiDANet, self).__init__()
        
       
        act_fn = nn.ReLU(inplace=True)

        self.relu = nn.ReLU(inplace=True)
        
        self.upsample_2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.upsample_4 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
        self.upsample_8 = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True)
        
        #Backbone model
        #Backbone model
        self.encoder = GBARES()

        self.layer_dep0 = nn.Conv2d(1, 3, kernel_size=1)
        
        ###############################################
        # funsion encoders #
        ###############################################
        self.fu_0 = CDA0(64, 64)
        
        self.fu_1 = CDA(256, 128) #MixedFusion_Block_IMfusion
        self.pool_fu_1 = maxpool()
        
        self.fu_2 = CDA(512, 256)
        self.pool_fu_2 = maxpool()
        
        self.fu_3 = CDA(1024, 512)
        self.pool_fu_3 = maxpool()

        self.fu_4 = CDA(2048, 1024)

        self.pool_fu_4 = maxpool()

        self.fu_rgb0 = AttFusion(64, channel) 
        self.fu_rgb1 = AttFusion(256, channel) 
        self.fu_rgb2 = AttFusion(512, channel) 
        self.fu_rgb3 = AttFusion(1024, channel) 


        self.fu_dep0 = AttFusion(64, channel) 
        self.fu_dep1 = AttFusion(256, channel) 
        self.fu_dep2 = AttFusion(512, channel) 
        self.fu_dep3 = AttFusion(1024, channel) 


        self.fu_ful0 = AttFusion(64, channel) 
        self.fu_ful1 = AttFusion(128, channel) 
        self.fu_ful2 = AttFusion(256, channel) 
        self.fu_ful3 = AttFusion(512, channel) 
        
        
        ###############################################
        # decoders #
        ###############################################
        
        ## rgb
        self.rgb_gcm_4    = GCM0(2048, channel)
        
        self.rgb_gcm_3    = GCM(channel)

        self.rgb_gcm_2    = GCM(channel)

        self.rgb_gcm_1    = GCM(channel)

        self.rgb_gcm_0    = GCM(channel)        
        self.rgb_conv_out = nn.Conv2d(channel, 1, 1)
        
        ## depth
        self.dep_gcm_4    = GCM0(2048, channel)
        
        self.dep_gcm_3    = GCM(channel)

        self.dep_gcm_2    = GCM(channel)

        self.dep_gcm_1    = GCM(channel)

        self.dep_gcm_0    = GCM(channel)        
        self.dep_conv_out = nn.Conv2d(channel, 1, 1)

        ## fusion
        self.ful_gcm_4    = GCM0(1024, channel)
        
        self.ful_gcm_3    = GCM(channel)

        self.ful_gcm_2    = GCM(channel)

        self.ful_gcm_1    = GCM(channel)

        self.ful_gcm_0    = GCM(channel)        
        self.ful_conv_out = nn.Conv2d(channel, 1, 1)
        
        self.ful_layer4   = EMI(channel)
        self.ful_layer3   = EMI(channel)
        self.ful_layer2   = EMI(channel)
        self.ful_layer1   = EMI(channel)
        self.ful_layer0   = EMI(channel)
        
                
        self.linearr1 = nn.Conv2d(channel, 1, kernel_size=3, stride=1, padding=1)
        self.linearr2 = nn.Conv2d(channel, 1, kernel_size=3, stride=1, padding=1)
        self.linearr3 = nn.Conv2d(channel, 1, kernel_size=3, stride=1, padding=1)
        self.linearr4 = nn.Conv2d(channel, 1, kernel_size=3, stride=1, padding=1)

        self.lineard1 = nn.Conv2d(channel, 1, kernel_size=3, stride=1, padding=1)
        self.lineard2 = nn.Conv2d(channel, 1, kernel_size=3, stride=1, padding=1)
        self.lineard3 = nn.Conv2d(channel, 1, kernel_size=3, stride=1, padding=1)
        self.lineard4 = nn.Conv2d(channel, 1, kernel_size=3, stride=1, padding=1)

        self.linearm1 = nn.Conv2d(channel, 1, kernel_size=3, stride=1, padding=1)
        self.linearm2 = nn.Conv2d(channel, 1, kernel_size=3, stride=1, padding=1)
        self.linearm3 = nn.Conv2d(channel, 1, kernel_size=3, stride=1, padding=1)
        self.linearm4 = nn.Conv2d(channel, 1, kernel_size=3, stride=1, padding=1)

    def forward(self, imgs, depths, bin):
        shape = imgs.size()[2:] 
        rfeats, dfeats = self.encoder(imgs, depths, bin)

        img_0, img_1, img_2, img_3, img_4 = rfeats
        dep_0, dep_1, dep_2, dep_3, dep_4 = dfeats

        ####################################################
        ## fusion
        ####################################################
        ful_0    = self.fu_0(img_0, dep_0)
        ful_1    = self.fu_1(img_1, dep_1, self.pool_fu_1(ful_0))
        ful_2    = self.fu_2(img_2, dep_2, self.pool_fu_1(ful_1))
        ful_3    = self.fu_3(img_3, dep_3, self.pool_fu_2(ful_2))
        ful_4    = self.fu_4(img_4, dep_4, self.pool_fu_3(ful_3))
        
        ####################################################
        ## decoder rgb
        ####################################################        
        #
        #import pdb
        #pdb.set_trace()
        x_rgb_42    = self.rgb_gcm_4(img_4)
        
        x_rgb_3_cat = self.fu_rgb3(img_3, self.upsample_2(x_rgb_42))
        x_rgb_32    = self.rgb_gcm_3(x_rgb_3_cat)
        
        x_rgb_2_cat = self.fu_rgb2(img_2, self.upsample_2(x_rgb_32))
        x_rgb_22    = self.rgb_gcm_2(x_rgb_2_cat)        

        x_rgb_1_cat = self.fu_rgb1(img_1, self.upsample_2(x_rgb_22))
        x_rgb_12    = self.rgb_gcm_1(x_rgb_1_cat)     

        #import pdb
        #pdb.set_trace()

        x_rgb_0_cat = self.fu_rgb0(img_0, self.upsample_2(x_rgb_12))
        x_rgb_02    = self.rgb_gcm_0(x_rgb_0_cat)     
        rgb_out     = self.upsample_2(self.rgb_conv_out(x_rgb_02))
        
        
        ####################################################
        ## decoder depth
        ####################################################        
        #
        x_dep_42    = self.dep_gcm_4(dep_4)

        x_dep_3_cat = self.fu_dep3(dep_3, self.upsample_2(x_dep_42))
        x_dep_32    = self.dep_gcm_3(x_dep_3_cat)
        
        x_dep_2_cat = self.fu_dep2(dep_2, self.upsample_2(x_dep_32))
        x_dep_22    = self.dep_gcm_2(x_dep_2_cat)        

        x_dep_1_cat = self.fu_dep1(dep_1, self.upsample_2(x_dep_22))
        x_dep_12    = self.dep_gcm_1(x_dep_1_cat)     

        x_dep_0_cat = self.fu_dep0(dep_0, self.upsample_2(x_dep_12))
        x_dep_02    = self.dep_gcm_0(x_dep_0_cat)     
        dep_out     = self.upsample_2(self.dep_conv_out(x_dep_02))
        

        ####################################################
        ## decoder fusion
        ####################################################        
        #
        x_ful_42    = self.ful_gcm_4(ful_4)
        
        x_ful_3_cat = self.fu_ful3(ful_3, self.ful_layer3(self.upsample_2(x_ful_42),self.upsample_2(x_rgb_42),self.upsample_2(x_dep_42)))
        x_ful_32    = self.ful_gcm_3(x_ful_3_cat)
        
        x_ful_2_cat = self.fu_ful2(ful_2, self.ful_layer2(self.upsample_2(x_ful_32),self.upsample_2(x_rgb_32),self.upsample_2(x_dep_32)))
        x_ful_22    = self.ful_gcm_2(x_ful_2_cat)        

        x_ful_1_cat = self.fu_ful1(ful_1, self.ful_layer1(self.upsample_2(x_ful_22),self.upsample_2(x_rgb_22),self.upsample_2(x_dep_22)))
        x_ful_12    = self.ful_gcm_1(x_ful_1_cat)     

        x_ful_0_cat = self.fu_ful0(ful_0, self.ful_layer0(self.upsample_2(x_ful_12), self.upsample_2(x_rgb_12), self.upsample_2(x_dep_12)))
        x_ful_02    = self.ful_gcm_0(x_ful_0_cat)     
        ful_out     = self.upsample_2(self.ful_conv_out(x_ful_02))

        out4r = F.interpolate(self.linearr4(x_rgb_42), size=shape, mode='bilinear')
        out3r = F.interpolate(self.linearr3(x_rgb_32), size=shape, mode='bilinear')
        out2r = F.interpolate(self.linearr2(x_rgb_22), size=shape, mode='bilinear')
        out1r = F.interpolate(self.linearr1(x_rgb_12), size=shape, mode='bilinear')
        
        out4d = F.interpolate(self.linearr4(x_dep_42), size=shape, mode='bilinear')
        out3d = F.interpolate(self.linearr3(x_dep_32), size=shape, mode='bilinear')
        out2d = F.interpolate(self.linearr2(x_dep_22), size=shape, mode='bilinear')
        out1d = F.interpolate(self.linearr1(x_dep_12), size=shape, mode='bilinear')

        out4m = F.interpolate(self.linearr4(x_ful_42), size=shape, mode='bilinear')
        out3m = F.interpolate(self.linearr3(x_ful_32), size=shape, mode='bilinear')
        out2m = F.interpolate(self.linearr2(x_ful_22), size=shape, mode='bilinear')
        out1m = F.interpolate(self.linearr1(x_ful_12), size=shape, mode='bilinear')


        return rgb_out, dep_out, ful_out, out4r, out3r, out2r, out1r, out4d, out3d, out2d, out1d, out4m, out3m, out2m, out1m

    
    

    def _make_agant_layer(self, inplanes, planes):
        layers = nn.Sequential(
            nn.Conv2d(inplanes, planes, kernel_size=1,
                      stride=1, padding=0, bias=False),
            nn.BatchNorm2d(planes),
            nn.ReLU(inplace=True)
        )
        return layers

    def _make_transpose(self, block, planes, blocks, stride=1):
        upsample = None
        if stride != 1:
            upsample = nn.Sequential(
                nn.ConvTranspose2d(self.inplanes, planes,
                                   kernel_size=2, stride=stride,
                                   padding=0, bias=False),
                nn.BatchNorm2d(planes),
            )
        elif self.inplanes != planes:
            upsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes),
            )

        layers = []

        for i in range(1, blocks):
            layers.append(block(self.inplanes, self.inplanes))

        layers.append(block(self.inplanes, planes, stride, upsample))
        self.inplanes = planes

        return nn.Sequential(*layers)
    
   

 
