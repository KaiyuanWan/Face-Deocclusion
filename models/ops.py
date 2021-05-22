from __future__ import division
import torch.nn as nn
import torch
from math import sqrt
import cv2
import numpy as np

def get_one_hot(label, N):
    size = list(label.size())
    label = label.view(-1)
    ones = torch.sparse.torch.eye(N)
    ones = ones.index_select(0, label.long())
    size.append(N)
    ones = ones.view(*size)[:,:,:,:].permute(0,3,1,2)
    return ones

def hard_prob(seg_mask):

    seg_mask = seg_mask[:,0,:,:].data.cpu().numpy().squeeze().astype(np.float32)
    seg_mask[np.where(seg_mask > 0.2)] = 1
    seg_mask[np.where(seg_mask <= 0.2)] = 0
    seg_mask_save = seg_mask * 255
    kernel = np.ones((5,5))
    seg_mask = cv2.dilate(seg_mask, kernel)
    seg_mask = torch.from_numpy(np.expand_dims(np.expand_dims(seg_mask, 0), 0))

    return seg_mask, seg_mask_save




def make_conv_layers(in_channels, out_channels, conv_layers=2, batch_norm=False, bn_aff=True):
    layers = []
    for i in range(conv_layers):
        layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
        if batch_norm:
            layers.append(nn.InstanceNorm2d(out_channels, affine=bn_aff))
        layers.append(nn.ReLU(inplace=True))
        in_channels = out_channels

    return nn.Sequential(*layers)



def make_gated_conv_layers(in_channels, out_channels, conv_layers=2, batch_norm=False):
    layers = []
    for i in range(conv_layers):
        rate = in_channels/out_channels
        if rate>2.5:
            out_channels_mul = out_channels*2
        else:
            out_channels_mul = out_channels
        layers.append(GatedConv(in_channels, out_channels_mul, kernel_size=3, padding=1, batch_norm=batch_norm, activation=nn.ReLU(inplace=True)))
        in_channels = out_channels_mul

    return nn.Sequential(*layers)


""" Non Local Layer"""
class Non_Local(nn.Module):

    def __init__(self,in_dim):
        super(Non_Local,self).__init__()
        self.chanel_in = in_dim
        self.query_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim//2 , kernel_size= 1)
        self.key_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim//2 , kernel_size= 1)
        self.value_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim , kernel_size= 1)
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax  = nn.Softmax(dim=-1) #

    def forward(self,x, mask):

        m_batchsize,C,width ,height = x.size()
        proj_query  = self.query_conv(x).view(m_batchsize,-1,width*height).permute(0,2,1) # B X CX(N)
        proj_key =  self.key_conv(x) # B X C x (*W*H)
        proj_key_with_mask = (proj_key * (1 - mask)).view(m_batchsize, -1, width * height)
        energy =  torch.bmm(proj_query,proj_key_with_mask) # transpose check
        attention = self.softmax(energy) # BX (N) X (N)
        proj_value = self.value_conv(x).view(m_batchsize,-1,width*height) # B X C X N

        out = torch.bmm(proj_value,attention.permute(0,2,1))
        out = out.view(m_batchsize,C,width,height)
        out = out * mask

        out = self.gamma*out + x
        # out = out + x

        return out


""" Non Local Layer"""
class Skin_Non_Local(nn.Module):

    def __init__(self,in_dim):
        super(Skin_Non_Local,self).__init__()
        self.chanel_in = in_dim
        self.query_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim//2 , kernel_size= 1)
        self.key_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim//2 , kernel_size= 1)
        self.value_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim , kernel_size= 1)
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax  = nn.Softmax(dim=-1) #

    def forward(self,x, mask, skin_mask):

        m_batchsize,C,width ,height = x.size()
        proj_query  = self.query_conv(x).view(m_batchsize,-1,width*height).permute(0,2,1) # N*(WH)*C
        proj_key =  self.key_conv(x)
        proj_key_with_mask = (proj_key * (1 - mask)*skin_mask).view(m_batchsize, -1, width * height) # N*C*(WH)
        energy =  torch.bmm(proj_query,proj_key_with_mask) # transpose check
        attention = self.softmax(energy) # BX (N) X (N)
        proj_value = self.value_conv(x).view(m_batchsize,-1,width*height) # B X C X N

        out = torch.bmm(proj_value,attention.permute(0,2,1))
        out = out.view(m_batchsize,C,width,height)
        out = out * mask* skin_mask

        out = self.gamma*out + x
        # out = out + x

        return out



class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, conv_layers=2, batch_norm=False, bn_aff=True):
        super(ConvBlock, self).__init__()
        self.conv_block = make_conv_layers(in_channels, out_channels, conv_layers, batch_norm, bn_aff)

    def forward(self, x):
        out = self.conv_block(x)

        return out



class GatedConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1, batch_norm=False, activation=nn.ReLU(inplace=True)):
        super(GatedConv, self).__init__()
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation=dilation)
        self.mask_conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation=dilation)
        self.sigmoid = nn.Sigmoid()
        self.activation = activation
        self.bn_aff = True
        self.batch_norm_bool = batch_norm
        self.batch_norm = nn.InstanceNorm2d(out_channels, affine=self.bn_aff)

    def gated(self, mask):
        return self.sigmoid(mask)

    def forward(self, input):
        x = self.conv2d(input)
        mask = self.mask_conv2d(input)
        if self.batch_norm_bool is True:
            x = self.batch_norm(x)
        if self.activation is not None:
            output = self.activation(x) * self.gated(mask)
        else:
            output = x * self.gated(mask)

        return output


class UpConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, skip_conct=False, conv_layers=2, batch_norm= False, ASPP=None):
        super(UpConvBlock, self).__init__()

        bn_aff = True
        self.bn = nn.InstanceNorm2d(out_channels, affine=bn_aff)
        self.actv = nn.ReLU(inplace=True)
        self.skip_conct = skip_conct
        if self.skip_conct:
            self.skip_conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
            self.skip_bn = nn.InstanceNorm2d(in_channels, affine=bn_aff)
            in_channels = in_channels*2

        self.conv = nn.Conv2d(in_channels, in_channels//2, kernel_size=3, padding=1)
        self.ASPP = ASPP
        self.conv_block = make_conv_layers(in_channels//2, out_channels, conv_layers, batch_norm, bn_aff)

    def forward(self, x, skip_layer=None):
        out = nn.functional.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        if self.skip_conct:
            assert skip_layer is not None,\
                'Layer with skip connection, must have an skip connected layer input'
            skip_layer = self.actv(self.skip_bn(self.skip_conv(skip_layer)))
            out = torch.cat((out, skip_layer), 1)
        out = self.conv(out)                        #channel>>>>channel/2
        if self.ASPP is not None:
            out = self.ASPP(out)
        out = self.conv_block(out)

        return out

class Gated_UpConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, skip_conct=False, conv_layers=2, batch_norm= False, ASPP=None, half_flag=False):
        super(Gated_UpConvBlock, self).__init__()

        bn_aff = True
        self.bn = nn.InstanceNorm2d(out_channels, affine=bn_aff)
        self.actv = nn.ReLU(inplace=True)
        self.skip_conct = skip_conct
        if self.skip_conct:
            if half_flag:
                self.skip_conv = nn.Conv2d(in_channels//2, in_channels//2, kernel_size=3, padding=1)
                self.skip_bn = nn.InstanceNorm2d(in_channels//2,affine=bn_aff)
                in_channels_ = in_channels + in_channels//2 + 1
            else:
                self.skip_conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
                self.skip_bn = nn.InstanceNorm2d(in_channels, affine=bn_aff)
                in_channels_ = in_channels*2 + 1

        self.conv = GatedConv(in_channels_, in_channels, kernel_size=3, padding=1)
        self.ASPP = ASPP
        self.conv_block = make_gated_conv_layers(in_channels, out_channels, conv_layers, batch_norm)

    def forward(self, x, skip_layer, shelter_mask):
        out = nn.functional.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        if self.skip_conct:
            assert skip_layer is not None,\
                'Layer with skip connection, must have an skip connected layer input'
            skip_layer = self.actv(self.skip_bn(self.skip_conv(skip_layer)))
            out = torch.cat((out, skip_layer, shelter_mask), 1)
        out = self.conv(out)                        #channel>>>>channel/2
        if self.ASPP:
            out = self.ASPP(out)
        out = self.conv_block(out)

        return out

class Gated_UpConvBlock_Gen(nn.Module):
    def __init__(self, in_channels, out_channels, skip_conct=False, conv_layers=2, up_sample=True, batch_norm=False):
        super(Gated_UpConvBlock_Gen, self).__init__()

        bn_aff = True
        self.actv = nn.ReLU(inplace=True)
        self.skip_conct = skip_conct
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.bn = nn.InstanceNorm2d(in_channels, affine=bn_aff)
        if not up_sample and not skip_conct:
            in_channels = in_channels + 8    ##encoder + seg_mask + parsing_mask
        elif up_sample and skip_conct :
            in_channels = in_channels * 2 + 8    ##encoder+decoder+ seg_mask + parsing_mask
        else:
            in_channels = in_channels + 1  ##encoder+decoder+ seg_mask
        self.conv_block = make_gated_conv_layers(in_channels, out_channels, conv_layers, batch_norm)

    def forward(self, x, skip_layer=None, seg_mask=None, parsing_mask=None):
        if x is None:
            skip_layer = self.conv(skip_layer)
            out = self.actv(self.bn(skip_layer))
            out = torch.cat((out, seg_mask, parsing_mask), 1)
        else:
            x_up = nn.functional.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
            if self.skip_conct:
                assert skip_layer is not None,\
                    'Layer with skip connection, must have an skip connected layer input'
                skip_layer = self.conv(skip_layer)
                out = self.actv(self.bn(skip_layer))
                out = torch.cat((x_up, out, seg_mask, parsing_mask), 1)
            else:
                out = torch.cat((x_up, seg_mask),1)
        out = self.conv_block(out)
        return out

class ASPPmodule(nn.Module):
    def __init__(self, in_channel, out_channel, input_size, output_size, dilated_rate, channel_down=False):
        super(ASPPmodule, self).__init__()
        self.channel_down = channel_down
        self.in_channel = in_channel
        self.out_channel = out_channel

        # ASPP
        self.imagepooling = nn.AdaptiveAvgPool2d((1,1))
        self.up = nn.Sequential(
            nn.Upsample(size=output_size, mode='nearest'))

        self.ASPP0 = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=3, dilation=dilated_rate[0], padding=dilated_rate[0]),
            nn.InstanceNorm2d(out_channel),
            nn.ReLU(inplace=True))
        self.ASPP1 = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=3, dilation=dilated_rate[1], padding=dilated_rate[1]),
            nn.InstanceNorm2d(out_channel),
            nn.ReLU(inplace=True))
        self.ASPP2 = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=3, dilation=dilated_rate[2], padding=dilated_rate[2]),
            nn.InstanceNorm2d(out_channel),
            nn.ReLU(inplace=True))
        self.conv = nn.Sequential(
            nn.Conv2d(out_channel * 4, in_channel, kernel_size=1),
            nn.InstanceNorm2d(in_channel),
            nn.ReLU(inplace=True))

        if channel_down:
            self.fc = nn.Sequential(
                nn.Conv2d(in_channel, out_channel, kernel_size=1),
                nn.InstanceNorm2d(out_channel),
                nn.ReLU(inplace=True))



    def forward(self, input_feature):

        #ASPP
        aspp_0 = self.ASPP0(input_feature)
        aspp_1 = self.ASPP1(input_feature)
        aspp_2 = self.ASPP2(input_feature)
        pool = self.imagepooling(input_feature)
        if self.channel_down:
            pool = self.fc(pool)

        pool = self.up(pool)
        output = torch.cat((pool, aspp_0, aspp_1, aspp_2), dim=1)
        output = self.conv(output)

        return output





class SNConvWithActivation(torch.nn.Module):
    """
    SN convolution for spetral normalization conv
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, activation=torch.nn.LeakyReLU(0.2, inplace=True)):
        super(SNConvWithActivation, self).__init__()
        self.conv2d = torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        self.conv2d = torch.nn.utils.spectral_norm(self.conv2d)
        self.activation = activation
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    m.bias.data.zero_()
    def forward(self, input):
        x = self.conv2d(input)
        if self.activation is not None:
            return self.activation(x)
        else:
            return x




class Self_Attn(nn.Module):
    """ Self attention Layer"""
    def __init__(self,in_dim,activation,with_attn=False):
        super(Self_Attn,self).__init__()
        self.chanel_in = in_dim
        self.activation = activation
        self.with_attn = with_attn
        self.query_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim//8 , kernel_size= 1)
        self.key_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim//8 , kernel_size= 1)
        self.value_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim , kernel_size= 1)
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax  = nn.Softmax(dim=-1) #


    def forward(self,x):
        """
            inputs :
                x : input feature maps( B X C X W X H)
            returns :
                out : self attention value + input featuregen_face
                attention: B X N X N (N is Width*Height)
        """
        m_batchsize,C,width ,height = x.size()
        proj_query  = self.query_conv(x).view(m_batchsize,-1,width*height).permute(0,2,1) # B X CX(N)
        proj_key =  self.key_conv(x).view(m_batchsize,-1,width*height) # B X C x (*W*H)
        energy =  torch.bmm(proj_query,proj_key) # transpose check
        attention = self.softmax(energy) # BX (N) X (N)
        proj_value = self.value_conv(x).view(m_batchsize,-1,width*height) # B X C X N

        out = torch.bmm(proj_value,attention.permute(0,2,1))
        out = out.view(m_batchsize,C,width,height)

        out = self.gamma*out + x
        if self.with_attn:
            return out ,attention
        else:
            return out