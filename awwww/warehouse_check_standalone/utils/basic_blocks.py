import torch
import torch.nn as nn
import torch.nn.functional as F

class REBNCONV(nn.Module):
    def __init__(self, in_ch, out_ch, dirate=1):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 3, padding=1*dirate, dilation=dirate, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))

class RSU4(nn.Module):
    def __init__(self, in_ch, mid_ch, out_ch):
        super().__init__()
        self.rebnconvin = REBNCONV(in_ch, out_ch)
        self.rebnconv1 = REBNCONV(out_ch, mid_ch)
        self.pool1 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.rebnconv2 = REBNCONV(mid_ch, mid_ch)
        self.pool2 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.rebnconv3 = REBNCONV(mid_ch, mid_ch)
        self.rebnconv4 = REBNCONV(mid_ch, mid_ch)
        self.rebnconv3d = REBNCONV(mid_ch*2, mid_ch)
        self.rebnconv2d = REBNCONV(mid_ch*2, mid_ch)
        self.rebnconv1d = REBNCONV(mid_ch*2, out_ch)
    
    def forward(self, x):
        hx = x
        hxin = self.rebnconvin(hx)
        h1 = self.rebnconv1(hxin)
        h = self.pool1(h1)
        h2 = self.rebnconv2(h)
        h = self.pool2(h2)
        h3 = self.rebnconv3(h)
        h4 = self.rebnconv4(h3)
        h3d = self.rebnconv3d(torch.cat([h4, h3], 1))
        h3dup = F.interpolate(h3d, size=h2.size()[2:], mode='bilinear', align_corners=False)
        h2d = self.rebnconv2d(torch.cat([h3dup, h2], 1))
        h2dup = F.interpolate(h2d, size=h1.size()[2:], mode='bilinear', align_corners=False)
        h1d = self.rebnconv1d(torch.cat([h2dup, h1], 1))
        return h1d + hxin

class RSU4F(nn.Module):
    def __init__(self, in_ch, mid_ch, out_ch):
        super().__init__()
        self.rebnconvin = REBNCONV(in_ch, out_ch, dirate=1)
        self.rebnconv1 = REBNCONV(out_ch, mid_ch, dirate=1)
        self.rebnconv2 = REBNCONV(mid_ch, mid_ch, dirate=2)
        self.rebnconv3 = REBNCONV(mid_ch, mid_ch, dirate=4)
        self.rebnconv4 = REBNCONV(mid_ch, mid_ch, dirate=8)
        self.rebnconv3d = REBNCONV(mid_ch*2, mid_ch, dirate=4)
        self.rebnconv2d = REBNCONV(mid_ch*2, mid_ch, dirate=2)
        self.rebnconv1d = REBNCONV(mid_ch*2, out_ch, dirate=1)
    
    def forward(self, x):
        hx = x
        hxin = self.rebnconvin(hx)
        h1 = self.rebnconv1(hxin)
        h2 = self.rebnconv2(h1)
        h3 = self.rebnconv3(h2)
        h4 = self.rebnconv4(h3)
        h3d = self.rebnconv3d(torch.cat([h4, h3], 1))
        h2d = self.rebnconv2d(torch.cat([h3d, h2], 1))
        h1d = self.rebnconv1d(torch.cat([h2d, h1], 1))
        return h1d + hxin

class RSU5(nn.Module):
    def __init__(self, in_ch, mid_ch, out_ch):
        super().__init__()
        self.rebnconvin = REBNCONV(in_ch, out_ch)
        self.rebnconv1 = REBNCONV(out_ch, mid_ch)
        self.pool1 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.rebnconv2 = REBNCONV(mid_ch, mid_ch)
        self.pool2 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.rebnconv3 = REBNCONV(mid_ch, mid_ch)
        self.pool3 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.rebnconv4 = REBNCONV(mid_ch, mid_ch)
        self.rebnconv5 = REBNCONV(mid_ch, mid_ch)
        self.rebnconv4d = REBNCONV(mid_ch*2, mid_ch)
        self.rebnconv3d = REBNCONV(mid_ch*2, mid_ch)
        self.rebnconv2d = REBNCONV(mid_ch*2, mid_ch)
        self.rebnconv1d = REBNCONV(mid_ch*2, out_ch)
    
    def forward(self, x):
        hx = x
        hxin = self.rebnconvin(hx)
        h1 = self.rebnconv1(hxin)
        h = self.pool1(h1)
        h2 = self.rebnconv2(h)
        h = self.pool2(h2)
        h3 = self.rebnconv3(h)
        h = self.pool3(h3)
        h4 = self.rebnconv4(h)
        h5 = self.rebnconv5(h4)
        h4d = self.rebnconv4d(torch.cat([h5, h4], 1))
        h4dup = F.interpolate(h4d, size=h3.size()[2:], mode='bilinear', align_corners=False)
        h3d = self.rebnconv3d(torch.cat([h4dup, h3], 1))
        h3dup = F.interpolate(h3d, size=h2.size()[2:], mode='bilinear', align_corners=False)
        h2d = self.rebnconv2d(torch.cat([h3dup, h2], 1))
        h2dup = F.interpolate(h2d, size=h1.size()[2:], mode='bilinear', align_corners=False)
        h1d = self.rebnconv1d(torch.cat([h2dup, h1], 1))
        return h1d + hxin

class RSU6(nn.Module):
    def __init__(self, in_ch, mid_ch, out_ch):
        super().__init__()
        self.rebnconvin = REBNCONV(in_ch, out_ch)
        self.rebnconv1 = REBNCONV(out_ch, mid_ch)
        self.pool1 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.rebnconv2 = REBNCONV(mid_ch, mid_ch)
        self.pool2 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.rebnconv3 = REBNCONV(mid_ch, mid_ch)
        self.pool3 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.rebnconv4 = REBNCONV(mid_ch, mid_ch)
        self.pool4 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.rebnconv5 = REBNCONV(mid_ch, mid_ch)
        self.rebnconv6 = REBNCONV(mid_ch, mid_ch)
        self.rebnconv5d = REBNCONV(mid_ch*2, mid_ch)
        self.rebnconv4d = REBNCONV(mid_ch*2, mid_ch)
        self.rebnconv3d = REBNCONV(mid_ch*2, mid_ch)
        self.rebnconv2d = REBNCONV(mid_ch*2, mid_ch)
        self.rebnconv1d = REBNCONV(mid_ch*2, out_ch)
    
    def forward(self, x):
        hx = x
        hxin = self.rebnconvin(hx)
        h1 = self.rebnconv1(hxin)
        h = self.pool1(h1)
        h2 = self.rebnconv2(h)
        h = self.pool2(h2)
        h3 = self.rebnconv3(h)
        h = self.pool3(h3)
        h4 = self.rebnconv4(h)
        h = self.pool4(h4)
        h5 = self.rebnconv5(h)
        h6 = self.rebnconv6(h5)
        h5d = self.rebnconv5d(torch.cat([h6, h5], 1))
        h5dup = F.interpolate(h5d, size=h4.size()[2:], mode='bilinear', align_corners=False)
        h4d = self.rebnconv4d(torch.cat([h5dup, h4], 1))
        h4dup = F.interpolate(h4d, size=h3.size()[2:], mode='bilinear', align_corners=False)
        h3d = self.rebnconv3d(torch.cat([h4dup, h3], 1))
        h3dup = F.interpolate(h3d, size=h2.size()[2:], mode='bilinear', align_corners=False)
        h2d = self.rebnconv2d(torch.cat([h3dup, h2], 1))
        h2dup = F.interpolate(h2d, size=h1.size()[2:], mode='bilinear', align_corners=False)
        h1d = self.rebnconv1d(torch.cat([h2dup, h1], 1))
        return h1d + hxin

class RSU7(nn.Module):
    def __init__(self, in_ch, mid_ch, out_ch):
        super().__init__()
        self.rebnconvin = REBNCONV(in_ch, out_ch)
        self.rebnconv1 = REBNCONV(out_ch, mid_ch)
        self.pool1 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.rebnconv2 = REBNCONV(mid_ch, mid_ch)
        self.pool2 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.rebnconv3 = REBNCONV(mid_ch, mid_ch)
        self.pool3 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.rebnconv4 = REBNCONV(mid_ch, mid_ch)
        self.pool4 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.rebnconv5 = REBNCONV(mid_ch, mid_ch)
        self.pool5 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.rebnconv6 = REBNCONV(mid_ch, mid_ch)
        self.rebnconv7 = REBNCONV(mid_ch, mid_ch)
        self.rebnconv6d = REBNCONV(mid_ch*2, mid_ch)
        self.rebnconv5d = REBNCONV(mid_ch*2, mid_ch)
        self.rebnconv4d = REBNCONV(mid_ch*2, mid_ch)
        self.rebnconv3d = REBNCONV(mid_ch*2, mid_ch)
        self.rebnconv2d = REBNCONV(mid_ch*2, mid_ch)
        self.rebnconv1d = REBNCONV(mid_ch*2, out_ch)
    
    def forward(self, x):
        hx = x
        hxin = self.rebnconvin(hx)
        h1 = self.rebnconv1(hxin)
        h = self.pool1(h1)
        h2 = self.rebnconv2(h)
        h = self.pool2(h2)
        h3 = self.rebnconv3(h)
        h = self.pool3(h3)
        h4 = self.rebnconv4(h)
        h = self.pool4(h4)
        h5 = self.rebnconv5(h)
        h = self.pool5(h5)
        h6 = self.rebnconv6(h)
        h7 = self.rebnconv7(h6)
        h6d = self.rebnconv6d(torch.cat([h7, h6], 1))
        h6dup = F.interpolate(h6d, size=h5.size()[2:], mode='bilinear', align_corners=False)
        h5d = self.rebnconv5d(torch.cat([h6dup, h5], 1))
        h5dup = F.interpolate(h5d, size=h4.size()[2:], mode='bilinear', align_corners=False)
        h4d = self.rebnconv4d(torch.cat([h5dup, h4], 1))
        h4dup = F.interpolate(h4d, size=h3.size()[2:], mode='bilinear', align_corners=False)
        h3d = self.rebnconv3d(torch.cat([h4dup, h3], 1))
        h3dup = F.interpolate(h3d, size=h2.size()[2:], mode='bilinear', align_corners=False)
        h2d = self.rebnconv2d(torch.cat([h3dup, h2], 1))
        h2dup = F.interpolate(h2d, size=h1.size()[2:], mode='bilinear', align_corners=False)
        h1d = self.rebnconv1d(torch.cat([h2dup, h1], 1))
        return h1d + hxin
