
import paddle
from paddle import nn
from paddle import Tensor
from paddle.nn import functional as F
from typing import Tuple, Optional

# from .mobilenetv3 import MobileNetV3LargeEncoder
from .resnet import ResNet50Encoder
from .lraspp import LRASPP
from .decoder import RecurrentDecoder, Projection
# from .fast_guided_filter import FastGuidedFilterRefiner
from .deep_guided_filter import DeepGuidedFilterRefiner


class MattingNetwork(nn.Layer):
    def __init__(self,
                 variant: str = 'mobilenetv3',
                 refiner: str = 'deep_guided_filter',
                 pretrained_backbone: bool = False):
        super().__init__()
        assert variant in ['mobilenetv3', 'resnet50']
        assert refiner in ['fast_guided_filter', 'deep_guided_filter']
#         print(variant, refiner)
        if variant == 'mobilenetv3':
            self.backbone = MobileNetV3LargeEncoder(pretrained_backbone)
            self.aspp = LRASPP(960, 128)
            self.decoder = RecurrentDecoder([16, 24, 40, 128], [80, 40, 32, 16])
        else:
            self.backbone = ResNet50Encoder(pretrained_backbone)
            self.aspp = LRASPP(2048, 256)
            self.decoder = RecurrentDecoder([64, 256, 512, 256], [128, 64, 32, 16])
            
        self.project_mat = Projection(16, 4)
        self.project_seg = Projection(16, 1)

        if refiner == 'deep_guided_filter':
            self.refiner = DeepGuidedFilterRefiner()
        else:
            self.refiner = FastGuidedFilterRefiner()
        
    def forward(self,
                src: Tensor,
                r1: Optional[Tensor] = None,
                r2: Optional[Tensor] = None,
                r3: Optional[Tensor] = None,
                r4: Optional[Tensor] = None,
                downsample_ratio: float = 1,
                segmentation_pass: bool = False):
        
        if downsample_ratio != 1:
            src_sm = self._interpolate(src, scale_factor=downsample_ratio)
        else:
            src_sm = src
#         print("src_sm=", src_sm.shape)
        f1, f2, f3, f4 = self.backbone(src_sm)
#         print("====f1, f2, f3, f4 = self.backbone(src_sm)", f1.shape, f2.shape, f3.shape, f4.shape)
        f4 = self.aspp(f4)
#         print("f4 = self.aspp(f4)", f4.shape)
        hid, *rec = self.decoder(src_sm, f1, f2, f3, f4, r1, r2, r3, r4)
        
        if not segmentation_pass:
            # fgr_residual, pha = self.project_mat(hid).split([3, 1], dim=-3)
#             print("self.project_mat(hid).split", self.project_mat(hid).shape)
            fgr_residual, pha = self.project_mat(hid).split([3, 1], axis=-3)
            # fgr_residual = self.project_mat(hid)[:, :3, ]
            # pha = self.project_mat(hid)[:, 3:, ]

            if downsample_ratio != 1:
                fgr_residual, pha = self.refiner(src, src_sm, fgr_residual, pha, hid)
            fgr = fgr_residual + src
            fgr = fgr.clip(0., 1.)
            pha = pha.clip(0., 1.)
            return [fgr, pha, *rec]
        else:
            seg = self.project_seg(hid)
            return [seg, *rec]

    def _interpolate(self, x: Tensor, scale_factor: float):
        if x.ndim == 5:
            B, T = x.shape[:2]
            # x = F.interpolate(x.flatten(0, 1), scale_factor=scale_factor,
            #     mode='bilinear', align_corners=False, recompute_scale_factor=False)
            x = F.interpolate(x.flatten(0, 1), scale_factor=scale_factor,
                mode='bilinear', align_corners=False)
            # x = x.unflatten(0, (B, T))
            x = x.reshape([B, T] + x.shape[1:])
        else:
            # x = F.interpolate(x, scale_factor=scale_factor,
            #     mode='bilinear', align_corners=False, recompute_scale_factor=False)
            x = F.interpolate(x, scale_factor=scale_factor,
                mode='bilinear', align_corners=False)
        return x

if __name__ == "__main__":
    pass
    # # 验证通过
    # model = MattingNetwork('resnet50')
    # # a = paddle.randn((2, 24, 3, 224, 224))
    # import numpy as np
    # np.random.seed(1)
    # a = np.random.randn(2,3,244,244).astype('float32')
    # a = paddle.to_tensor(a)
    # tmp = model(a)
    # print(f"输入数据shape:{a.shape} 输出数据长度:{len(tmp)}")
    # for i in tmp:
    #     print(i.shape())
