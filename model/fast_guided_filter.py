import paddle
import paddle.nn as nn 
from paddle.nn import functional as F
# from .fast_guided_filter import FastGuidedFilterRefiner
"""
Adopted from <https://github.com/wuhuikai/DeepGuidedFilter/>
"""

class FastGuidedFilterRefiner(nn.Layer):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.guilded_filter = FastGuidedFilter(1)
    
    def forward_single_frame(self, fine_src, base_src, base_fgr, base_pha):
        fine_src_gray = fine_src.mean(1, keepdim=True)
        base_src_gray = base_src.mean(1, keepdim=True)
        
        fgr, pha = self.guilded_filter(
            # torch.cat([base_src, base_src_gray], dim=1),
            # torch.cat([base_fgr, base_pha], dim=1),
            # torch.cat([fine_src, fine_src_gray], dim=1)).split([3, 1], dim=1) 
            paddle.concat([base_src, base_src_gray], axis=1),
            paddle.concat([base_fgr, base_pha], daxisim=1),
            paddle.concat([fine_src, fine_src_gray], axis=1)).split([3, 1], axis=1) 
#         print("FastGuidedFilterRefiner forward_single_frame fgr, pha", fgr.shape, pha.shape)
        return fgr, pha
    
    def forward_time_series(self, fine_src, base_src, base_fgr, base_pha):
#         print("==FastGuidedFilterRefiner fine_src, base_src, base_fgr, base_pha", fine_src, base_src, base_fgr, base_pha)
        B, T = fine_src.shape[:2]
        fgr, pha = self.forward_single_frame(
            fine_src.flatten(0, 1),
            base_src.flatten(0, 1),
            base_fgr.flatten(0, 1),
            base_pha.flatten(0, 1))
        # fgr = fgr.unflatten(0, (B, T))
        fgr = fgr.reshape([B, T] + fgr.shape[1:])
        # pha = pha.unflatten(0, (B, T))
        pha = fgr.reshape([B, T] + pha.shape[1:])
#         print("FastGuidedFilterRefiner forward_time_series fgr, pha", fgr.shape, pha.shape)
        return fgr, pha
    
    def forward(self, fine_src, base_src, base_fgr, base_pha, base_hid):
        # print("fine_src.ndim=", fine_src.ndim)
        if fine_src.ndim == 5:
            return self.forward_time_series(fine_src, base_src, base_fgr, base_pha)
        else:
            return self.forward_single_frame(fine_src, base_src, base_fgr, base_pha)


class FastGuidedFilter(nn.Layer):
    def __init__(self, r: int, eps: float = 1e-5):
        super().__init__()
        self.r = r
        self.eps = eps
        self.boxfilter = BoxFilter(r)

    def forward(self, lr_x, lr_y, hr_x):
        mean_x = self.boxfilter(lr_x)
        mean_y = self.boxfilter(lr_y)
        cov_xy = self.boxfilter(lr_x * lr_y) - mean_x * mean_y
        var_x = self.boxfilter(lr_x * lr_x) - mean_x * mean_x
        A = cov_xy / (var_x + self.eps)
        b = mean_y - A * mean_x
        A = F.interpolate(A, hr_x.shape[2:], mode='bilinear', align_corners=False)
        b = F.interpolate(b, hr_x.shape[2:], mode='bilinear', align_corners=False)
        return A * hr_x + b


class BoxFilter(nn.Layer):
    def __init__(self, r):
        super(BoxFilter, self).__init__()
        self.r = r

    def forward(self, x):
        # Note: The original implementation at <https://github.com/wuhuikai/DeepGuidedFilter/>
        #       uses faster box blur. However, it may not be friendly for ONNX export.
        #       We are switching to use simple convolution for box blur.
        kernel_size = 2 * self.r + 1
        # kernel_x = torch.full((x.data.shape[1], 1, 1, kernel_size), 1 / kernel_size, device=x.device, dtype=x.dtype)
        # kernel_y = torch.full((x.data.shape[1], 1, kernel_size, 1), 1 / kernel_size, device=x.device, dtype=x.dtype)
        kernel_x = paddle.full((x.shape[1], 1, 1, kernel_size), 1 / kernel_size, dtype=x.dtype)
        kernel_y = paddle.full((x.shape[1], 1, kernel_size, 1), 1 / kernel_size, dtype=x.dtype)
        x = F.conv2d(x, kernel_x, padding=(0, self.r), groups=x.shape[1])
        x = F.conv2d(x, kernel_y, padding=(self.r, 0), groups=x.shape[1])
#         print(x.shape)
        return x
        