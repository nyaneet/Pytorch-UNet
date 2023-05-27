""" Full assembly of the parts to form the complete network """

from .unet_parts import *


class AttentionUnet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False):
        super(AttentionUnet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
    
        factor = 2 if bilinear else 1
        
        self.inc = (DoubleConv(n_channels, 64))
        self.down1 = (Down(64, 128))
        self.down2 = (Down(128, 256))
        self.down3 = (Down(256, 512))
        self.down4 = (Down(512, 1024 // factor))

        self.up1 = (AttentionUp(1024, 512 // factor, bilinear))
        self.up2 = (AttentionUp(512, 256 // factor, bilinear))
        self.up3 = (AttentionUp(256, 128 // factor, bilinear))
        self.up4 = (AttentionUp(128, 64, bilinear))
        self.outc = (OutConv(64, n_classes))
    
    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        d5 = self.up1(x5, x4)
        d4 = self.up2(d5, x3)
        d3 = self.up3(d4, x2)
        d2 = self.up4(d3, x1)
        d1 = self.outc(d2)
        return d1
    