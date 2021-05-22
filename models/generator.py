from __future__ import division
import torch.nn as nn
import torch.nn.functional as F
import torch
from ops import *

__all__ = ['deocnet']

class seg_decoder(nn.Module):
    def __init__(self, cn, de_conv_layers):
        super(seg_decoder, self).__init__()
        self.cn = cn
        self.de_conv_layers = de_conv_layers

        # ASPP
        self.ASPP_5 = ASPPmodule(512, 512, 8, 8, dilated_rate=[1, 2, 3], channel_down=False)
        self.ASPP_4 = ASPPmodule(512, 512, 16, 16, dilated_rate=[1, 2, 4], channel_down=False)
        self.ASPP_3 = ASPPmodule(256, 256, 32, 32, dilated_rate=[2, 4, 6], channel_down=False)

        self.seg_up_block5 = UpConvBlock(self.cn[4], self.cn[3], True, self.de_conv_layers, batch_norm=True, ASPP=self.ASPP_4)
        self.seg_up_block4 = UpConvBlock(self.cn[3], self.cn[2], True, self.de_conv_layers, batch_norm=True, ASPP=self.ASPP_3)
        self.seg_up_block3 = UpConvBlock(self.cn[2], self.cn[1], True, self.de_conv_layers, batch_norm=True)
        self.seg_up_block2 = UpConvBlock(self.cn[1], self.cn[0], True, self.de_conv_layers, batch_norm=True)
        self.seg_up_block1 = nn.Conv2d(self.cn[0], 1, 3, padding=1)


    def forward(self, blocks):

        [block1, block2, block3, block4, block5] = blocks

        Aspp_5 = self.ASPP_5(block5)
        seg_up4 = self.seg_up_block5(Aspp_5, block4)
        seg_up3 = self.seg_up_block4(seg_up4, block3)
        seg_up2 = self.seg_up_block3(seg_up3, block2)
        seg_up1 = self.seg_up_block2(seg_up2, block1)
        seg_mask = torch.sigmoid(self.seg_up_block1(seg_up1))

        return seg_mask

class parsing_decoder(nn.Module):
    def __init__(self, cn, de_conv_layers):
        super(parsing_decoder, self).__init__()
        self.cn = cn
        self.de_conv_layers = de_conv_layers

        self.ASPP_5 = ASPPmodule(512, 512, 8, 8, dilated_rate=[1, 2, 3], channel_down=False)
        self.ASPP_4 = ASPPmodule(512, 512, 16, 16, dilated_rate=[1, 2, 4], channel_down=False)
        self.ASPP_3 = ASPPmodule(256, 256, 32, 32, dilated_rate=[2, 4, 6], channel_down=False)

        self.parsing_up_block5 = Gated_UpConvBlock(self.cn[4], self.cn[3], True, self.de_conv_layers, batch_norm=True, ASPP = self.ASPP_4)
        self.parsing_up_block4 = Gated_UpConvBlock(self.cn[3], self.cn[2], True, self.de_conv_layers, batch_norm=True, ASPP = self.ASPP_3)
        self.parsing_up_block3 = Gated_UpConvBlock(self.cn[2], self.cn[2], True, self.de_conv_layers, batch_norm=True)
        self.parsing_up_block2 = Gated_UpConvBlock(self.cn[2], self.cn[2], True, self.de_conv_layers, batch_norm=True, half_flag=True)
        self.parsing_up_block1 = nn.Conv2d(self.cn[2], 7, 3, padding=1)

    def forward(self, blocks, seg_masks):

        [block1, block2, block3, block4, block5] = blocks
        [seg_mask, seg_up1, seg_up2, seg_up3] = seg_masks

        Aspp_5 = self.ASPP_5(block5)
        parsing_up4 = self.parsing_up_block5(Aspp_5, block4, seg_up3.detach())  ##channel:512+512>>>256
        parsing_up3 = self.parsing_up_block4(parsing_up4, block3, seg_up2.detach())  ##channel:256+256>>>128
        parsing_up2 = self.parsing_up_block3(parsing_up3, block2, seg_up1.detach())  ##channel:128+128>>>128
        parsing_up1 = self.parsing_up_block2(parsing_up2, block1, seg_mask.detach())  ##channel:128+64>>>128
        parsing_mask = self.parsing_up_block1(parsing_up1)

        return parsing_mask

class Detect_ParsingNet(nn.Module):
    def __init__(self, fc_dim=512, de_conv_layers=2):
        super(Detect_ParsingNet, self).__init__()

        self.down_sampler = nn.AvgPool2d(2, stride=2)
        self.cn = [3, 64, 128, 256, 512, 512]  # channel number
        self.fc_dim = fc_dim
        self.gated_cn = [cn+1 for cn in self.cn]
        self.de_conv_layers = de_conv_layers


        # Encoder modules
        self.conv_block1 = nn.Sequential(
            GatedConv(3, 64, 3, padding=1),
            GatedConv(64, 64, 3, padding=1),
        )
        self.conv_block2 = nn.Sequential(
            nn.MaxPool2d(2, 2),
            GatedConv(64, 128, 3, padding=1),
            GatedConv(128, 128, 3, padding=1),
        )
        self.conv_block3 = nn.Sequential(
            nn.MaxPool2d(2, 2),
            GatedConv(128, 256, 3, padding=1),
            GatedConv(256, 256, 3, padding=1),
            GatedConv(256, 256, 3, padding=1),
        )
        self.conv_block4 = nn.Sequential(
            nn.MaxPool2d(2, 2),
            GatedConv(256, 512, 3, padding=1),
            GatedConv(512, 512, 3, padding=1),
            GatedConv(512, 512, 3, padding=1),
        )
        self.conv_block5 = nn.Sequential(
            nn.MaxPool2d(2, 2),
            GatedConv(512, 512, 3, padding=1),
            GatedConv(512, 512, 3, padding=1),
            GatedConv(512, 512, 3, padding=1),
        )

        self.seg_decoder = seg_decoder(self.cn, self.de_conv_layers)
        self.parsing_decoder = parsing_decoder(self.cn, self.de_conv_layers)


    def forward(self, x):

        # Encoder
        block1 = self.conv_block1(x)    # 128 x 128
        block2 = self.conv_block2(block1)   # 64 x 64
        block3 = self.conv_block3(block2)   # 32 x 32
        block4 = self.conv_block4(block3)   # 16 x 16
        block5 = self.conv_block5(block4)   # 8 x 8

        blocks = [block1, block2, block3, block4, block5]

        seg_mask = self.seg_decoder(blocks)    ##seg_decoder
        seg_mask2 = F.interpolate(seg_mask, size=[64,64], mode='nearest')   # 64 x 64
        seg_mask3 = F.interpolate(seg_mask, size=[32,32], mode='nearest')   # 32 x 32
        seg_mask4 = F.interpolate(seg_mask, size=[16,16], mode='nearest')   # 16 x 16
        seg_masks = [seg_mask, seg_mask2, seg_mask3, seg_mask4]
        parsing_mask = self.parsing_decoder(blocks, seg_masks)  #parsing_decoder

        return seg_mask, parsing_mask

class ReconstructNet(nn.Module):
    def __init__(self, fc_dim=512, de_conv_layers=2):
        super(ReconstructNet, self).__init__()

        self.down_sampler = nn.AvgPool2d(2, stride=2)
        self.cn = [3, 64, 128, 256, 512, 512]  # channel number
        self.fc_dim = fc_dim
        self.gated_cn = [cn+1 for cn in self.cn]


        # Encoder modules
        self.conv_block1 = nn.Sequential(
            GatedConv(3, 64, 3, padding=1),
            GatedConv(64, 64, 3, padding=1),
        )
        self.conv_block2 = nn.Sequential(
            nn.MaxPool2d(2, 2),
            GatedConv(64, 128, 3, padding=1),
            GatedConv(128, 128, 3, padding=1),
        )
        self.conv_block3 = nn.Sequential(
            nn.MaxPool2d(2, 2),
            GatedConv(128, 256, 3, padding=1),
            GatedConv(256, 256, 3, padding=1),
            GatedConv(256, 256, 3, padding=1),
        )
        self.conv_block4 = nn.Sequential(
            nn.MaxPool2d(2, 2),
            GatedConv(256, 512, 3, padding=1),
            GatedConv(512, 512, 3, padding=1),
            GatedConv(512, 512, 3, padding=1),
        )
        self.conv_block5 = nn.Sequential(
            nn.MaxPool2d(2, 2),
            GatedConv(512, 512, 3, padding=1),
            GatedConv(512, 512, 3, padding=1),
            GatedConv(512, 512, 3, padding=1),
        )
        
        # Generation decoder modules
        self.gen_up_block6 = Gated_UpConvBlock_Gen(self.cn[5], self.cn[4], False, conv_layers=2, up_sample=False,batch_norm=True)
        self.gen_up_block5 = Gated_UpConvBlock_Gen(self.cn[4], self.cn[3], True, conv_layers=2, batch_norm=True)
        self.gen_up_block4 = Gated_UpConvBlock_Gen(self.cn[3], self.cn[3], True, conv_layers=2, batch_norm=True)

        self.feature_match_5 = Non_Local(self.cn[5])
        self.feature_match_4 = Non_Local(self.cn[3])
        self.skin_match_3 = Skin_Non_Local(self.cn[3])
        self.gen_up_block3 = Gated_UpConvBlock_Gen(self.cn[3], self.cn[2], False, conv_layers=2, batch_norm=True)
        self.gen_up_block2 = Gated_UpConvBlock_Gen(self.cn[2], self.cn[1], False, conv_layers=2, batch_norm=True)
        self.gen_up_block1 = GatedConv(self.cn[1], self.cn[0], 3, padding=1, activation=None)

    def forward(self, x, seg_mask, parsing_mask):

        # seg_mask
        seg_mask_ = F.max_pool2d(seg_mask, kernel_size=9, stride=1, padding=4)
        seg_mask = (seg_mask_ - seg_mask) * 0.5 + seg_mask
        # seg_mask = F.max_pool2d(seg_mask, kernel_size=9, stride=1, padding=4)
        seg_mask2 = F.interpolate(seg_mask, size=[64, 64], mode='nearest')  # 64 x 64
        seg_mask3 = F.interpolate(seg_mask, size=[32, 32], mode='nearest')  # 32 x 32
        seg_mask4 = F.interpolate(seg_mask, size=[16, 16], mode='nearest')  # 16 x 16
        seg_mask5 = F.interpolate(seg_mask, size=[8, 8], mode='nearest')  # 8 x 8

        # parsing_mask
        parsing_mask3 = F.interpolate(parsing_mask, size=[32, 32], mode='nearest')  # 32 x 32
        parsing_mask4 = F.interpolate(parsing_mask, size=[16, 16], mode='nearest')  # 16 x 16
        parsing_mask5 = F.interpolate(parsing_mask, size=[8, 8], mode='nearest')  # 8 x 8

        # skin_mask skin_number=1
        skin_mask3 = parsing_mask3[:, 1:2, :, :]
        # Encoder

        # 128 x 128
        block1 = self.conv_block1(x)

        # 64 x 64
        block2 = self.conv_block2(block1)

        # 32 x 32
        block3 = self.conv_block3(block2)

        # 16 x 16
        block4 = self.conv_block4(block3)

        # 8 x 8
        block5 = self.conv_block5(block4)

        # Generation Decoder
        gen_up5 = self.gen_up_block6(None, block5, seg_mask5, parsing_mask5)
        feature_match_5 = self.feature_match_5(gen_up5, seg_mask5)
        gen_up4 = self.gen_up_block5(feature_match_5, block4, seg_mask4, parsing_mask4)
        feature_match_4 = self.feature_match_4(gen_up4, seg_mask4)
        gen_up3 = self.gen_up_block4(feature_match_4, block3, seg_mask3, parsing_mask3)
        feature_match_3 = self.skin_match_3(gen_up3, seg_mask3, skin_mask3)
        gen_up2 = self.gen_up_block3(feature_match_3, seg_mask=seg_mask2)
        gen_up1 = self.gen_up_block2(gen_up2, seg_mask=seg_mask)
        gen_face = torch.tanh(self.gen_up_block1(gen_up1))

        return gen_face

class InpaintSADirciminator(nn.Module):
    def __init__(self):
        super(InpaintSADirciminator, self).__init__()
        cnum = 32
        self.discriminator_net = nn.Sequential(
            SNConvWithActivation(3, 2*cnum, 5, 1, padding=2),
            SNConvWithActivation(2*cnum, 4*cnum, 5, 2, padding=2),
            SNConvWithActivation(4*cnum, 8*cnum, 5, 2, padding=2),
            SNConvWithActivation(8*cnum, 8*cnum, 5, 2, padding=2),
            SNConvWithActivation(8*cnum, 8*cnum, 5, 2, padding=2),
            SNConvWithActivation(8*cnum, 8*cnum, 5, 1, padding=2),
            Self_Attn(8*cnum, 'relu'),
            SNConvWithActivation(8*cnum, 8*cnum, 5, 2, padding=2, activation=None),
        )
        # self.linear = nn.Linear(8*cnum*1*1, 1)

    def forward(self, input):
        x = self.discriminator_net(input)
        x = x.view((-1, 1))
        #x = self.linear(x)
        return x


def detect_parsing():
    """standard DeocNet
    """
    return Detect_ParsingNet(de_conv_layers=1)

def reconstruct():
    """standard DeocNet
    """
    return ReconstructNet(de_conv_layers=1)

def SN_PatchGAN():

    return InpaintSADirciminator()



