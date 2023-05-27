""" Full assembly of the parts to form the complete network """

from .unet_parts import *


class UNet3plusCBAM(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False):
        super(UNet3plusCBAM, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.filters = [64, 128, 256, 512, 1024]

        self.inc = (DoubleConv(n_channels, self.filters[0]))
        self.down1 = (Down(self.filters[0], self.filters[1]))
        self.down2 = (Down(self.filters[1], self.filters[2]))
        self.down3 = (Down(self.filters[2], self.filters[3]))
        factor = 2 if bilinear else 1
        self.down4 = (Down(self.filters[3], self.filters[4] // factor))
        self.cbamd5 = (CBAM(self.filters[4]))

        self.cat_channels = 64
        self.cat_channels_out = self.cat_channels * 5

        # decoder 4
        self.en1d4 = (DownF(self.filters[0], self.cat_channels, 8))
        self.en2d4 = (DownF(self.filters[1], self.cat_channels, 4))
        self.en3d4 = (DownF(self.filters[2], self.cat_channels, 2))
        self.en4d4 = (Conv(self.filters[3], self.cat_channels))
        self.en5d4 = (UpF(self.filters[4], self.cat_channels, 2))
        self.catd4 = (Conv(self.cat_channels*5, self.cat_channels_out))
        self.cbamd4 = (CBAM(self.cat_channels*5))
        # decoder 3
        self.en1d3 = (DownF(self.filters[0], self.cat_channels, 4))
        self.en2d3 = (DownF(self.filters[1], self.cat_channels, 2))
        self.en3d3 = (Conv(self.filters[2], self.cat_channels))
        self.en4d3 = (UpF(self.cat_channels_out, self.cat_channels, 2))
        self.en5d3 = (UpF(self.filters[4], self.cat_channels, 4))
        self.catd3 = (Conv(self.cat_channels*5, self.cat_channels_out))
        self.cbamd3 = (CBAM(self.cat_channels*5))
        # decoder 2
        self.en1d2 = (DownF(self.filters[0], self.cat_channels, 2))
        self.en2d2 = (Conv(self.filters[1], self.cat_channels))
        self.en3d2 = (UpF(self.cat_channels_out, self.cat_channels, 2))
        self.en4d2 = (UpF(self.cat_channels_out, self.cat_channels, 4))
        self.en5d2 = (UpF(self.filters[4], self.cat_channels, 8))
        self.catd2 = (Conv(self.cat_channels*5, self.cat_channels_out))
        self.cbamd2 = (CBAM(self.cat_channels*5))
        # decoder 1
        self.en1d1 = (Conv(self.filters[0], self.cat_channels))
        self.en2d1 = (UpF(self.cat_channels_out, self.cat_channels, 2))
        self.en3d1 = (UpF(self.cat_channels_out, self.cat_channels, 4))
        self.en4d1 = (UpF(self.cat_channels_out, self.cat_channels, 8))
        self.en5d1 = (UpF(self.filters[4], self.cat_channels, 16))
        self.catd1 = (Conv(self.cat_channels*5, self.cat_channels_out))
        self.cbamd1 = (CBAM(self.cat_channels*5))
        
        self.outc = (OutConv(self.cat_channels_out, n_classes))

    def forward(self, x):
        en1 = self.inc(x)
        en2 = self.down1(en1)
        en3 = self.down2(en2)
        en4 = self.down3(en3)
        d5 = self.down4(en4)
        d5 = self.cbamd5(d5)

        en1d4 = self.en1d4(en1)
        en2d4 = self.en2d4(en2)
        en3d4 = self.en3d4(en3)
        en4d4 = self.en4d4(en4)
        en5d4 = self.en5d4(d5)
        d4 = self.catd4(torch.cat((en1d4, en2d4, en3d4, en4d4, en5d4), 1))
        d4 = self.cbamd4(d4)
        
        en1d3 = self.en1d3(en1)
        en2d3 = self.en2d3(en2)
        en3d3 = self.en3d3(en3)
        en4d3 = self.en4d3(d4)
        en5d3 = self.en5d3(d5)
        d3 = self.catd3(torch.cat((en1d3, en2d3, en3d3, en4d3, en5d3), 1))
        d3 = self.cbamd3(d3)
        
        en1d2 = self.en1d2(en1)
        en2d2 = self.en2d2(en2)
        en3d2 = self.en3d2(d3)
        en4d2 = self.en4d2(d4)
        en5d2 = self.en5d2(d5)
        d2 = self.catd2(torch.cat((en1d2, en2d2, en3d2, en4d2, en5d2), 1))
        d2 = self.cbamd2(d2)
        
        en1d1 = self.en1d1(en1)
        en2d1 = self.en2d1(d2)
        en3d1 = self.en3d1(d3)
        en4d1 = self.en4d1(d4)
        en5d1 = self.en5d1(d5)
        d1 = self.catd1(torch.cat((en1d1, en2d1, en3d1, en4d1, en5d1), 1))
        d1 = self.cbamd1(d1)
        
        logits = self.outc(d1)
        return logits