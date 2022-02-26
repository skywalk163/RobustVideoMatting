import paddle
from paddle import nn
from paddle import Tensor
from paddle.nn import functional as F
from typing import Tuple, Optional

class RecurrentDecoder(nn.Layer):
    def __init__(self, feature_channels, decoder_channels):
        super().__init__()
        self.avgpool = AvgPool()
        self.decode4 = BottleneckBlock(feature_channels[3])
        self.decode3 = UpsamplingBlock(feature_channels[3], feature_channels[2], 3, decoder_channels[0])
        self.decode2 = UpsamplingBlock(decoder_channels[0], feature_channels[1], 3, decoder_channels[1])
        self.decode1 = UpsamplingBlock(decoder_channels[1], feature_channels[0], 3, decoder_channels[2])
        self.decode0 = OutputBlock(decoder_channels[2], 3, decoder_channels[3])

    def forward(self,
                s0: Tensor, f1: Tensor, f2: Tensor, f3: Tensor, f4: Tensor,
                r1: Optional[Tensor], r2: Optional[Tensor],
                r3: Optional[Tensor], r4: Optional[Tensor]):
        s1, s2, s3 = self.avgpool(s0)
        x4, r4 = self.decode4(f4, r4)
        x3, r3 = self.decode3(x4, f3, s3, r3)
        x2, r2 = self.decode2(x3, f2, s2, r2)
        x1, r1 = self.decode1(x2, f1, s1, r1)
        x0 = self.decode0(x1, s0)
        # print("x0, r1, r2, r3, r4", x0.shape, r1.shape,r2.shape,r3.shape, r4.shape)
        return x0, r1, r2, r3, r4
    

class AvgPool(nn.Layer):
    def __init__(self):
        super().__init__()
        self.avgpool = nn.AvgPool2D(2, 2, exclusive=False, ceil_mode=True)  # count_include_pad exclusive
        
    def forward_single_frame(self, s0):
        s1 = self.avgpool(s0)
        s2 = self.avgpool(s1)
        s3 = self.avgpool(s2)
        return s1, s2, s3
    
    def forward_time_series(self, s0):
        B, T = s0.shape[:2]
        s0 = s0.flatten(0, 1)
        s1, s2, s3 = self.forward_single_frame(s0)
        # s1 = s1.unflatten(0, (B, T))
        # s2 = s2.unflatten(0, (B, T))
        # s3 = s3.unflatten(0, (B, T))
        s1 = s1.reshape([B, T] + s1.shape[1:])
        s2 = s2.reshape([B, T] + s2.shape[1:])
        s3 = s3.reshape([B, T] + s3.shape[1:])
        return s1, s2, s3
    
    def forward(self, s0):
        if s0.ndim == 5:
            return self.forward_time_series(s0)
        else:
            return self.forward_single_frame(s0)


class BottleneckBlock(nn.Layer):
    def __init__(self, channels):
        super().__init__()
        self.channels = channels
        self.gru = ConvGRU(channels // 2)
        
    def forward(self, x, r: Optional[Tensor]):
        # print("a, b = x.split(self.channels // 2, axis=-3)", self.channels // 2)
        a = []
        # a, b = x.split(self.channels // 2, axis=-3) # dim
        a, b = x.split(2, axis=-3) 
        # print(len(a))
        b, r = self.gru(b, r)
        x = paddle.concat([a, b], axis=-3) # cat concat
        return x, r

    
class UpsamplingBlock(nn.Layer):
    def __init__(self, in_channels, skip_channels, src_channels, out_channels):
        super().__init__()
        self.out_channels = out_channels
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.conv = nn.Sequential(
            nn.Conv2D(in_channels + skip_channels + src_channels, out_channels, 3, 1, 1, bias_attr=False),
            nn.BatchNorm2D(out_channels),
            nn.ReLU(),
        )
        self.gru = ConvGRU(out_channels // 2)

    def forward_single_frame(self, x, f, s, r: Optional[Tensor]):
        x = self.upsample(x)
        x = x[:, :, :s.shape[2], :s.shape[3]]
        x = paddle.concat([x, f, s], axis=1) #torch.cat paddle.concat dim - axis

        x = self.conv(x)
        # a, b = x.split(self.out_channels // 2, axis=1)
        a, b = x.split(2, axis=1)
        
        b, r = self.gru(b, r)
        x = paddle.concat([a, b], axis=1)
        return x, r
    
    def forward_time_series(self, x, f, s, r: Optional[Tensor]):
        B, T, _, H, W = s.shape
        x = x.flatten(0, 1)
        f = f.flatten(0, 1)
        s = s.flatten(0, 1)
        x = self.upsample(x)
        x = x[:, :, :H, :W]
        x = paddle.concat([x, f, s], axis=1)
        x = self.conv(x)
        # x = x.unflatten(0, (B, T))
        x = x.reshape([B, T] + x.shape[1:])

        # a, b = x.split(self.out_channels // 2, axis=2)
        a, b = x.split(2, axis=2)
        b, r = self.gru(b, r)
        x = paddle.concat([a, b], axis=2)
        return x, r
    
    def forward(self, x, f, s, r: Optional[Tensor]):
        if x.ndim == 5:
            return self.forward_time_series(x, f, s, r)
        else:
            return self.forward_single_frame(x, f, s, r)


class OutputBlock(nn.Layer):
    def __init__(self, in_channels, src_channels, out_channels):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.conv = nn.Sequential(
            nn.Conv2D(in_channels + src_channels, out_channels, 3, 1, 1, bias_attr=False),
            nn.BatchNorm2D(out_channels),
            nn.ReLU(),
            nn.Conv2D(out_channels, out_channels, 3, 1, 1, bias_attr=False),
            nn.BatchNorm2D(out_channels),
            nn.ReLU(),
        )
        
    def forward_single_frame(self, x, s):
        x = self.upsample(x)
        x = x[:, :, :s.shape[2], :s.shape[3]]
        x = paddle.concat([x, s], axis=1)
        x = self.conv(x)
        return x
    
    def forward_time_series(self, x, s):
        B, T, _, H, W = s.shape
        x = x.flatten(0, 1)
        s = s.flatten(0, 1)
        x = self.upsample(x)
        x = x[:, :, :H, :W]
        x = paddle.concat([x, s], axis=1)
        x = self.conv(x)
        # x = x.unflatten(0, (B, T))
        x = x.reshape([B, T] + x.shape[1:])
        return x
    
    def forward(self, x, s):
        if x.ndim == 5:
            return self.forward_time_series(x, s)
        else:
            return self.forward_single_frame(x, s)


class ConvGRU(nn.Layer):
    def __init__(self,
                 channels: int,
                 kernel_size: int = 3,
                 padding: int = 1):
        super().__init__()
        self.channels = channels
        self.ih = nn.Sequential(
            nn.Conv2D(channels * 2, channels * 2, kernel_size, padding=padding),
            nn.Sigmoid()
        )
        self.hh = nn.Sequential(
            nn.Conv2D(channels * 2, channels, kernel_size, padding=padding),
            nn.Tanh()
        )
        
    def forward_single_frame(self, x, h):
        # print("forward_single_frame split(self.channels, axis=1)", self.channels)
        # r, z = self.ih(paddle.concat([x, h], axis=1)).split(self.channels, axis=1)
        r, z = self.ih(paddle.concat([x, h], axis=1)).split(2, axis=1)
        
        # print(r,z)
        c = self.hh(paddle.concat([x, r * h], axis=1))
        h = (1 - z) * h + z * c
        return h, h
    
    def forward_time_series(self, x, h):
        o = []
        for xt in x.unbind(axis=1): # dim to axis
            ot, h = self.forward_single_frame(xt, h)
            o.append(ot)
        o = paddle.stack(o, axis=1) # torch.stack dim-axis
        return o, h
        
    def forward(self, x, h: Optional[Tensor]):
        if h is None:
            h = paddle.zeros((x.shape[0], x.shape[-3], x.shape[-2], x.shape[-1]),
                             dtype=x.dtype)
        
        if x.ndim == 5:
            return self.forward_time_series(x, h)
        else:
            return self.forward_single_frame(x, h)


class Projection(nn.Layer):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2D(in_channels, out_channels, 1)
    
    def forward_single_frame(self, x):
        return self.conv(x)
    
    def forward_time_series(self, x):
        B, T = x.shape[:2]
        # return self.conv(x.flatten(0, 1)).unflatten(0, (B, T))
        x = self.conv(x.flatten(0, 1))
        return x.reshape([B, T] + x.shape[1:])
        
    def forward(self, x):
        if x.ndim == 5:
            return self.forward_time_series(x)
        else:
            return self.forward_single_frame(x)
    
if __name__ == "__main__":

    # 需要看看RecuRecurrentDecoder的输入信息，以便验证
    # RecurrentDecoder([64, 256, 512, 256], [128, 64, 32, 16])

    s0 = paddle.randn([2, 2, 3, 224, 224])
    f1=paddle.randn([2, 2, 64, 112, 112])
    f2=paddle.randn([2, 2, 256, 56, 56])
    f3=paddle.randn([2, 2, 512, 28, 28])
    f4=paddle.randn([2, 2, 256, 14, 14])
    r1=r2=r3=r4=None

    testmodel = RecurrentDecoder([64, 256, 512, 256], [128, 64, 32, 16])
    tmp = testmodel(s0, f1, f2, f3, f4, r1, r2, r3, r4)
    print(f"输入8个数据 其中S0的shape:{s0.shape} 输出数据长度:{len(tmp)}")
    for i in tmp:
        print(i.shape)
