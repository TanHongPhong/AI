import torch
import torch.nn as nn
import torch.nn.functional as F
from .basic_blocks import REBNCONV, RSU4, RSU4F, RSU5, RSU6, RSU7

class U2NETP(nn.Module):
    """U²-Net-P lightweight - EXACT COPY from NCC_PIPELINE_NEW.py"""
    def __init__(self, in_ch=3, out_ch=1):
        super().__init__()
        self.stage1 = RSU4(in_ch, 16, 64)
        self.pool12 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.stage2 = RSU4(64, 16, 64)
        self.pool23 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.stage3 = RSU4(64, 16, 64)
        self.pool34 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.stage4 = RSU4F(64, 16, 64)  # FIXED: Use RSU4F not RSU4
        self.stage3d = RSU4(128, 16, 64)
        self.stage2d = RSU4(128, 16, 64)
        self.stage1d = RSU4(128, 16, 64)
        self.side1 = nn.Conv2d(64, out_ch, 3, padding=1)
        self.side2 = nn.Conv2d(64, out_ch, 3, padding=1)
        self.side3 = nn.Conv2d(64, out_ch, 3, padding=1)
        self.outconv = nn.Conv2d(3*out_ch, out_ch, 1)
    
    def forward(self, x):
        hx = x
        h1 = self.stage1(hx)
        h = self.pool12(h1)
        h2 = self.stage2(h)
        h = self.pool23(h2)
        h3 = self.stage3(h)
        h = self.pool34(h3)
        h4 = self.stage4(h)
        h4up = F.interpolate(h4, size=h3.size()[2:], mode='bilinear', align_corners=False)
        h3d = self.stage3d(torch.cat([h4up, h3], 1))
        h3dup = F.interpolate(h3d, size=h2.size()[2:], mode='bilinear', align_corners=False)
        h2d = self.stage2d(torch.cat([h3dup, h2], 1))
        h2dup = F.interpolate(h2d, size=h1.size()[2:], mode='bilinear', align_corners=False)
        h1d = self.stage1d(torch.cat([h2dup, h1], 1))
        d1 = self.side1(h1d)
        d2 = F.interpolate(self.side2(h2d), size=x.size()[2:], mode='bilinear', align_corners=False)
        d3 = F.interpolate(self.side3(h3d), size=x.size()[2:], mode='bilinear', align_corners=False)
        d0 = self.outconv(torch.cat([d1, d2, d3], 1))
        return d0

class U2NET(nn.Module):
    """U²-Net Full architecture with deep supervision - EXACT COPY from NCC_PIPELINE_NEW.py"""
    def __init__(self, in_ch=3, out_ch=1):
        super().__init__()
        # Encoder
        self.stage1 = RSU7(in_ch, 32, 64)
        self.pool12 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.stage2 = RSU6(64, 32, 128)
        self.pool23 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.stage3 = RSU5(128, 64, 256)
        self.pool34 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.stage4 = RSU4(256, 128, 512)
        self.pool45 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.stage5 = RSU4F(512, 256, 512)
        self.pool56 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.stage6 = RSU4F(512, 256, 512)
        # Decoder
        self.stage5d = RSU4F(1024, 256, 512)
        self.stage4d = RSU4(1024, 128, 256)
        self.stage3d = RSU5(512, 64, 128)
        self.stage2d = RSU6(256, 32, 64)
        self.stage1d = RSU7(128, 16, 64)
        # Side outputs
        self.side1 = nn.Conv2d(64, out_ch, 3, padding=1)
        self.side2 = nn.Conv2d(64, out_ch, 3, padding=1)
        self.side3 = nn.Conv2d(128, out_ch, 3, padding=1)
        self.side4 = nn.Conv2d(256, out_ch, 3, padding=1)
        self.side5 = nn.Conv2d(512, out_ch, 3, padding=1)
        self.side6 = nn.Conv2d(512, out_ch, 3, padding=1)
        self.outconv = nn.Conv2d(6*out_ch, out_ch, 1)
    
    def forward(self, x):
        hx = x
        h1 = self.stage1(hx)
        h = self.pool12(h1)
        h2 = self.stage2(h)
        h = self.pool23(h2)
        h3 = self.stage3(h)
        h = self.pool34(h3)
        h4 = self.stage4(h)
        h = self.pool45(h4)
        h5 = self.stage5(h)
        h = self.pool56(h5)
        h6 = self.stage6(h)
        h6up = F.interpolate(h6, size=h5.size()[2:], mode='bilinear', align_corners=False)
        h5d = self.stage5d(torch.cat([h6up, h5], 1))
        h5dup = F.interpolate(h5d, size=h4.size()[2:], mode='bilinear', align_corners=False)
        h4d = self.stage4d(torch.cat([h5dup, h4], 1))
        h4dup = F.interpolate(h4d, size=h3.size()[2:], mode='bilinear', align_corners=False)
        h3d = self.stage3d(torch.cat([h4dup, h3], 1))
        h3dup = F.interpolate(h3d, size=h2.size()[2:], mode='bilinear', align_corners=False)
        h2d = self.stage2d(torch.cat([h3dup, h2], 1))
        h2dup = F.interpolate(h2d, size=h1.size()[2:], mode='bilinear', align_corners=False)
        h1d = self.stage1d(torch.cat([h2dup, h1], 1))
        d1 = self.side1(h1d)
        d2 = F.interpolate(self.side2(h2d), size=x.size()[2:], mode='bilinear', align_corners=False)
        d3 = F.interpolate(self.side3(h3d), size=x.size()[2:], mode='bilinear', align_corners=False)
        d4 = F.interpolate(self.side4(h4d), size=x.size()[2:], mode='bilinear', align_corners=False)
        d5 = F.interpolate(self.side5(h5d), size=x.size()[2:], mode='bilinear', align_corners=False)
        d6 = F.interpolate(self.side6(h6), size=x.size()[2:], mode='bilinear', align_corners=False)
        d0 = self.outconv(torch.cat([d1, d2, d3, d4, d5, d6], 1))
        return d0

class U2NET_LITE(nn.Module):
    """U²-Net-Lite super lightweight version for mobile/embedded - EXACT COPY from NCC_PIPELINE_NEW.py"""
    def __init__(self, in_ch=3, out_ch=1):
        super().__init__()
        self.stage1 = RSU4(in_ch, 8, 32)
        self.pool12 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.stage2 = RSU4(32, 8, 32)
        self.pool23 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.stage3 = RSU4(32, 8, 32)
        # decoder
        self.stage2d = RSU4(64, 8, 32)
        self.stage1d = RSU4(64, 8, 32)
        self.side1 = nn.Conv2d(32, out_ch, 3, padding=1)
        self.side2 = nn.Conv2d(32, out_ch, 3, padding=1)
        self.outconv = nn.Conv2d(2*out_ch, out_ch, 1)
    
    def forward(self, x):
        hx = x
        h1 = self.stage1(hx)
        h = self.pool12(h1)
        h2 = self.stage2(h)
        h = self.pool23(h2)
        h3 = self.stage3(h)
        h3up = F.interpolate(h3, size=h2.size()[2:], mode='bilinear', align_corners=False)
        h2d = self.stage2d(torch.cat([h3up, h2], 1))
        h2dup = F.interpolate(h2d, size=h1.size()[2:], mode='bilinear', align_corners=False)
        h1d = self.stage1d(torch.cat([h2dup, h1], 1))
        # side output
        d1 = self.side1(h1d)
        d2 = F.interpolate(self.side2(h2d), size=x.size()[2:], mode='bilinear', align_corners=False)
        d0 = self.outconv(torch.cat([d1, d2], 1))
        return d0