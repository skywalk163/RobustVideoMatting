import paddle
from paddle import nn
from paddle.nn import functional as F
"""
Adopted from <https://github.com/wuhuikai/DeepGuidedFilter/>
"""

class DeepGuidedFilterRefiner(nn.Layer):
    def __init__(self, hid_channels=16):
        super().__init__()
        self.box_filter = nn.Conv2D(4, 4, kernel_size=3, padding=1, bias_attr=False, groups=4) # 修改bisa
        # print("box_filter", type(self.box_filter), self.box_filter.weight)
        # self.box_filter.weight.data[...] = 1 / 9
        
        # self.box_filter.weight =1/9
        # x = paddle.to_tensor(1/9, dtype="float32")
        # print("==self.box_filter.weight.shape", self.box_filter.weight.shape)
        x = paddle.full(self.box_filter.weight.shape, 1/9, dtype="float32") # shape=[4,1,3,1,]
        self.box_filter.weight = paddle.create_parameter(shape=x.shape,
                        dtype=str(x.numpy().dtype),
                        default_initializer=paddle.nn.initializer.Assign(x))
        self.conv = nn.Sequential(
            nn.Conv2D(4 * 2 + hid_channels, hid_channels, kernel_size=1, bias_attr=False),
            nn.BatchNorm2D(hid_channels),
            nn.ReLU(True),
            nn.Conv2D(hid_channels, hid_channels, kernel_size=1, bias_attr=False),
            nn.BatchNorm2D(hid_channels),
            nn.ReLU(True),
            nn.Conv2D(hid_channels, 4, kernel_size=1, bias_attr=True)
        )
        
    def forward_single_frame(self, fine_src, base_src, base_fgr, base_pha, base_hid):
        # fine_x = torch.cat([fine_src, fine_src.mean(1, keepdim=True)], dim=1) # axis
        # base_x = torch.cat([base_src, base_src.mean(1, keepdim=True)], dim=1)
        # base_y = torch.cat([base_fgr, base_pha], dim=1)
        fine_x = paddle.concat([fine_src, fine_src.mean(1, keepdim=True)], axis=1) # axis
        base_x = paddle.concat([base_src, base_src.mean(1, keepdim=True)], axis=1)
        base_y = paddle.concat([base_fgr, base_pha], axis=1)

        mean_x = self.box_filter(base_x)
        mean_y = self.box_filter(base_y)
        cov_xy = self.box_filter(base_x * base_y) - mean_x * mean_y
        var_x  = self.box_filter(base_x * base_x) - mean_x * mean_x
        
        # A = self.conv(torch.cat([cov_xy, var_x, base_hid], dim=1))
        A = self.conv(paddle.concat([cov_xy, var_x, base_hid], axis=1))
        b = mean_y - A * mean_x
        
        H, W = fine_src.shape[2:]
        A = F.interpolate(A, (H, W), mode='bilinear', align_corners=False)
        b = F.interpolate(b, (H, W), mode='bilinear', align_corners=False)
        
        out = A * fine_x + b
        # fgr, pha = out.split([3, 1], dim=1) 
        fgr = out[:, :3, ]
        pha = out[:, 3:, ]
#         print("DeepGuidedFilterRefiner forward_single_frame fgr, pha", fgr.shape, pha.shape)
        return fgr, pha
    
    def forward_time_series(self, fine_src, base_src, base_fgr, base_pha, base_hid):
        B, T = fine_src.shape[:2]
        fgr, pha = self.forward_single_frame(
            fine_src.flatten(0, 1),
            base_src.flatten(0, 1),
            base_fgr.flatten(0, 1),
            base_pha.flatten(0, 1),
            base_hid.flatten(0, 1))
        # fgr = fgr.unflatten(0, (B, T))
        fgr = fgr.reshape([B, T] + fgr.shape[1:])
        # pha = pha.unflatten(0, (B, T))
        pha = pha.reshape([B, T] + pha.shape[1:])
#         print("DeepGuidedFilterRefiner forward_time_series fgr, pha", fgr.shape, pha.shape)
        return fgr, pha
    
    def forward(self, fine_src, base_src, base_fgr, base_pha, base_hid):
        if fine_src.ndim == 5:
            # print("if fine_src.ndim == 5:")
            return self.forward_time_series(fine_src, base_src, base_fgr, base_pha, base_hid)
        else:
            # print("if fine_src.ndim != 5:")
            return self.forward_single_frame(fine_src, base_src, base_fgr, base_pha, base_hid)
