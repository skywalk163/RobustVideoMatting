import paddle
from paddle import nn

class LRASPP(nn.Layer):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.aspp1 = nn.Sequential(
            nn.Conv2D(in_channels, out_channels, 1, bias_attr=False),
            nn.BatchNorm2D(out_channels),
            nn.ReLU(True)
        )
        self.aspp2 = nn.Sequential(
            nn.AdaptiveAvgPool2D(1),
            nn.Conv2D(in_channels, out_channels, 1, bias_attr=False),
            nn.Sigmoid()
        )
        
    def forward_single_frame(self, x):
        return self.aspp1(x) * self.aspp2(x)
    
    def forward_time_series(self, x):
        B, T = x.shape[:2]
        # x = self.forward_single_frame(x.flatten(0, 1)).unflatten(0, (B, T))
        x = self.forward_single_frame(x.flatten(0, 1))
        x = x.reshape([B, T]+x.shape[1:])
        
        return x
    
    def forward(self, x):
        # print("x.ndim =", x.ndim)
        if x.ndim == 5:
            # print("self.forward_time_series(x)", self.forward_time_series(x).shape)
            return self.forward_time_series(x)
        else:
            return self.forward_single_frame(x)

if __name__ == "__main__":
    a = paddle.randn((2, 2048, 14, 14))
    # print(a.max())
    testmodel = LRASPP(2048, 256)
    tmp = testmodel(a)
    print(f"输入数据shape:{a.shape} 输出数据长度:{len(tmp)}")
    for i in tmp:
        print(i.shape)
        # print(i.max(2))