# 手工修改几个x2paddle里面的代码，以便在AIStudio下运行
import paddle
class ReLU(paddle.nn.ReLU):
    def __init__(self, inplace=False):
        super().__init__()
        self.inplace = inplace

    def forward(self, x):
        if self.inplace:
            out = paddle.nn.functional.relu_(x)
        else:
            out = super().forward(x)
        return out

def constant_init_(param, val):
    replaced_param = paddle.create_parameter(
        shape=param.shape,
        dtype=param.dtype,
        default_initializer=paddle.nn.initializer.Assign(
            paddle.full(param.shape, val, param.dtype)))
    paddle.assign(replaced_param, param)

# 照抄x2paddle里面的paddle_dtypes，因为x2paddle有时候抽风
# -*- coding:UTF-8 -*-
# Copyright (c) 2021  PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# import paddle


def string(param):
    """ 生成字符串。
    """
    return "\'{}\'".format(param)


def check_version():
    version = paddle.__version__
    v0, v1, v2 = version.split('.')
    if not ((v0 == '0' and v1 == '0' and v2 == '0') or
            (int(v0) >= 2 and int(v1) >= 1)):
        return False
    else:
        return True


class PaddleDtypes():
    def __init__(self, is_new_version=True):
        if is_new_version:
            self.t_float16 = paddle.float16
            self.t_float32 = paddle.float32
            self.t_float64 = paddle.float64
            self.t_uint8 = paddle.uint8
            self.t_int8 = paddle.int8
            self.t_int16 = paddle.int16
            self.t_int32 = paddle.int32
            self.t_int64 = paddle.int64
            self.t_bool = paddle.bool
        else:
            self.t_float16 = "paddle.fluid.core.VarDesc.VarType.FP16"
            self.t_float32 = "paddle.fluid.core.VarDesc.VarType.FP32"
            self.t_float64 = "paddle.fluid.core.VarDesc.VarType.FP64"
            self.t_uint8 = "paddle.fluid.core.VarDesc.VarType.UINT8"
            self.t_int8 = "paddle.fluid.core.VarDesc.VarType.INT8"
            self.t_int16 = "paddle.fluid.core.VarDesc.VarType.INT16"
            self.t_int32 = "paddle.fluid.core.VarDesc.VarType.INT32"
            self.t_int64 = "paddle.fluid.core.VarDesc.VarType.INT64"
            self.t_bool = "paddle.fluid.core.VarDesc.VarType.BOOL"


is_new_version = check_version()
paddle_dtypes = PaddleDtypes(is_new_version)

# 单独写凯明初始化
# Copyright (c) 2021  PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math
from functools import reduce
import paddle
from paddle.fluid import framework
from paddle.fluid.core import VarDesc
from paddle.fluid.initializer import XavierInitializer, MSRAInitializer
from paddle.fluid.data_feeder import check_variable_and_dtype
# from x2paddle.utils import paddle_dtypes


def _calculate_fan_in_and_fan_out(var):
    dimensions = var.dim()
    if dimensions < 2:
        raise ValueError(
            "Fan in and fan out can not be computed for var with fewer than 2 dimensions"
        )
    num_input_fmaps = var.shape[0]
    num_output_fmaps = var.shape[1]
    receptive_field_size = 1
    if var.dim() > 2:
        receptive_field_size = reduce(lambda x, y: x * y, var.shape[2:])
    fan_in = num_input_fmaps * receptive_field_size
    fan_out = num_output_fmaps * receptive_field_size
    return fan_in, fan_out


def _calculate_correct_fan(var, mode):
    mode = mode.lower()
    valid_modes = ['fan_in', 'fan_out']
    if mode not in valid_modes:
        raise ValueError("Mode {} not supported, please use one of {}".format(
            mode, valid_modes))
    fan_in, fan_out = _calculate_fan_in_and_fan_out(var)
    return fan_in if mode == 'fan_in' else fan_out


def _calculate_gain(nonlinearity, param=None):
    linear_fns = [
        'linear', 'conv1d', 'conv2d', 'conv3d', 'conv_transpose1d',
        'conv_transpose2d', 'conv_transpose3d'
    ]
    if nonlinearity in linear_fns or nonlinearity == 'sigmoid':
        return 1
    elif nonlinearity == 'tanh':
        return 5.0 / 3
    elif nonlinearity == 'relu':
        return math.sqrt(2.0)
    elif nonlinearity == 'leaky_relu':
        if param is None:
            negative_slope = 0.01
        elif not isinstance(param, bool) and isinstance(
                param, int) or isinstance(param, float):
            # True/False are instances of int, hence check above
            negative_slope = param
        else:
            raise ValueError("negative_slope {} not a valid number".format(
                param))
        return math.sqrt(2.0 / (1 + negative_slope**2))
    elif nonlinearity == 'selu':
        return 3.0 / 4  # Value found empirically (https://github.com/pytorch/pytorch/pull/50664)
    else:
        raise ValueError("Unsupported nonlinearity {}".format(nonlinearity))


class KaimingNormal(MSRAInitializer):
    def __init__(self, a=0, mode='fan_in', nonlinearity='leaky_relu'):
        super(KaimingNormal, self).__init__(uniform=False, fan_in=None, seed=0)
        self.a = a
        self.mode = mode
        self.nonlinearity = nonlinearity

    def __call__(self, var, block=None):
        """Initialize the input tensor with MSRA initialization.
        Args:
            var(Tensor): Tensor that needs to be initialized.
            block(Block, optional): The block in which initialization ops
                   should be added. Used in static graph only, default None.
        Returns:
            The initialization op
        """
        block = self._check_block(block)

        assert isinstance(var, framework.Variable)
        assert isinstance(block, framework.Block)
        f_in, f_out = self._compute_fans(var)

        if self._seed == 0:
            self._seed = block.program.random_seed

        # to be compatible of fp16 initalizers
        if var.dtype == paddle_dtypes.t_float16:
            out_dtype = paddle_dtypes.t_float32
            out_var = block.create_var(
                name=unique_name.generate(".".join(
                    ['masra_init', var.name, 'tmp'])),
                shape=var.shape,
                dtype=out_dtype,
                type=VarDesc.VarType.LOD_TENSOR,
                persistable=False)
        else:
            out_dtype = var.dtype
            out_var = var

        fan = _calculate_correct_fan(var, self.mode)
        gain = _calculate_gain(self.nonlinearity, self.a)
        std = gain / math.sqrt(fan)
        op = block._prepend_op(
            type="gaussian_random",
            outputs={"Out": out_var},
            attrs={
                "shape": out_var.shape,
                "dtype": int(out_dtype),
                "mean": 0.0,
                "std": std,
                "seed": self._seed
            },
            stop_gradient=True)

        if var.dtype == VarDesc.VarType.FP16:
            block.append_op(
                type="cast",
                inputs={"X": out_var},
                outputs={"Out": var},
                attrs={"in_dtype": out_var.dtype,
                       "out_dtype": var.dtype})

        if not framework.in_dygraph_mode():
            var.op = op
        return op


def kaiming_normal_(param, a=0, mode='fan_in', nonlinearity='leaky_relu'):
    replaced_param = paddle.create_parameter(
        shape=param.shape,
        dtype=param.dtype,
        default_initializer=KaimingNormal(
            a=a, mode=mode, nonlinearity=nonlinearity))
    paddle.assign(replaced_param, param)


class XavierNormal(XavierInitializer):
    def __init__(self, gain=1.0):
        super(XavierNormal, self).__init__(
            uniform=True, fan_in=None, fan_out=None, seed=0)
        self._gain = gain

    def __call__(self, var, block=None):
        block = self._check_block(block)
        assert isinstance(block, framework.Block)
        check_variable_and_dtype(var, "Out", ["float16", "float32", "float64"],
                                 "xavier_init")

        fan_in, fan_out = _calculate_fan_in_and_fan_out(var)

        if self._seed == 0:
            self._seed = block.program.random_seed

        # to be compatible of fp16 initalizers
        if var.dtype == paddle_dtypes.t_float16:
            out_dtype = paddle_dtypes.t_float32
            out_var = block.create_var(
                name=unique_name.generate(".".join(
                    ['xavier_init', var.name, 'tmp'])),
                shape=var.shape,
                dtype=out_dtype,
                type=VarDesc.VarType.LOD_TENSOR,
                persistable=False)
        else:
            out_dtype = var.dtype
            out_var = var

        std = self._gain * math.sqrt(2.0 / float(fan_in + fan_out))
        op = block._prepend_op(
            type="uniform_random",
            inputs={},
            outputs={"Out": out_var},
            attrs={
                "shape": out_var.shape,
                "dtype": out_dtype,
                "min": 0,
                "max": std,
                "seed": self._seed
            },
            stop_gradient=True)
        if var.dtype == paddle_dtypes.t_float16:
            block.append_op(
                type="cast",
                inputs={"X": out_var},
                outputs={"Out": var},
                attrs={"in_dtype": out_var.dtype,
                       "out_dtype": var.dtype})
        if not framework.in_dygraph_mode():
            var.op = op
        return op


def xavier_normal_(param, gain=1.0):
    replaced_param = paddle.create_parameter(
        shape=param.shape,
        dtype=param.dtype,
        default_initializer=XavierNormal(gain=gain))
    paddle.assign(replaced_param, param)


class XavierUniform(XavierInitializer):
    def __init__(self, gain=1.0):
        super(XavierUniform, self).__init__(
            uniform=True, fan_in=None, fan_out=None, seed=0)
        self._gain = gain

    def __call__(self, var, block=None):
        block = self._check_block(block)
        assert isinstance(block, framework.Block)
        check_variable_and_dtype(var, "Out", ["float16", "float32", "float64"],
                                 "xavier_init")

        fan_in, fan_out = _calculate_fan_in_and_fan_out(var)

        if self._seed == 0:
            self._seed = block.program.random_seed

        # to be compatible of fp16 initalizers
        if var.dtype == paddle_dtypes.t_float16:
            out_dtype = paddle_dtypes.t_float32
            out_var = block.create_var(
                name=unique_name.generate(".".join(
                    ['xavier_init', var.name, 'tmp'])),
                shape=var.shape,
                dtype=out_dtype,
                type=VarDesc.VarType.LOD_TENSOR,
                persistable=False)
        else:
            out_dtype = var.dtype
            out_var = var

        std = self._gain * math.sqrt(2.0 / float(fan_in + fan_out))
        limit = math.sqrt(3.0) * std
        op = block._prepend_op(
            type="uniform_random",
            inputs={},
            outputs={"Out": out_var},
            attrs={
                "shape": out_var.shape,
                "dtype": out_dtype,
                "min": -limit,
                "max": limit,
                "seed": self._seed
            },
            stop_gradient=True)
        if var.dtype == paddle_dtypes.t_float16:
            block.append_op(
                type="cast",
                inputs={"X": out_var},
                outputs={"Out": var},
                attrs={"in_dtype": out_var.dtype,
                       "out_dtype": var.dtype})
        if not framework.in_dygraph_mode():
            var.op = op
        return op


def xavier_uniform_(param, gain=1.0):
    replaced_param = paddle.create_parameter(
        shape=param.shape,
        dtype=param.dtype,
        default_initializer=XavierUniform(gain=gain))
    paddle.assign(replaced_param, param)


def constant_init_(param, val):
    replaced_param = paddle.create_parameter(
        shape=param.shape,
        dtype=param.dtype,
        default_initializer=paddle.nn.initializer.Assign(
            paddle.full(param.shape, val, param.dtype)))
    paddle.assign(replaced_param, param)


def normal_init_(param, mean=0.0, std=1.0):
    replaced_param = paddle.create_parameter(
        shape=param.shape,
        dtype=param.dtype,
        default_initializer=paddle.nn.initializer.Assign(
            paddle.normal(
                mean=mean, std=std, shape=param.shape)))
    paddle.assign(replaced_param, param)


def ones_init_(param):
    replaced_param = paddle.create_parameter(
        shape=param.shape,
        dtype=param.dtype,
        default_initializer=paddle.nn.initializer.Assign(
            paddle.ones(param.shape, param.dtype)))
    paddle.assign(replaced_param, param)


def zeros_init_(param):
    replaced_param = paddle.create_parameter(
        shape=param.shape,
        dtype=param.dtype,
        default_initializer=paddle.nn.initializer.Assign(
            paddle.zeros(param.shape, param.dtype)))
    paddle.assign(replaced_param, param)

# 使用x2paddle里面的代码,完全临摹torch的ResNet，以便参数对齐
import paddle
import paddle.nn as nn
from paddle import Tensor
from paddle.utils.download import get_weights_path_from_url
from typing import Type, Any, Callable, Union, List, Optional
# from x2paddle import storch2paddle

__all__ = [
    'ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152',
    'resnext50_32x4d', 'resnext101_32x8d', 'wide_resnet50_2', 'wide_resnet101_2'
]

model_urls = {
    'resnet18':
    'https://x2paddle.bj.bcebos.com/vision/models/resnet18-pt.pdparams',
    'resnet34':
    'https://x2paddle.bj.bcebos.com/vision/models/resnet34-pt.pdparams',
    'resnet50':
    'https://x2paddle.bj.bcebos.com/vision/models/resnet50-pt.pdparams',
    'resnet101':
    'https://x2paddle.bj.bcebos.com/vision/models/resnet101-pt.pdparams',
    'resnet152':
    'https://x2paddle.bj.bcebos.com/vision/models/resnet152-pt.pdparams',
    'resnext50_32x4d':
    'https://x2paddle.bj.bcebos.com/vision/models/resnext50_32x4d-pt.pdparams',
    'resnext101_32x8d':
    'https://x2paddle.bj.bcebos.com/vision/models/resnext101_32x8d-pt.pdparams',
    'wide_resnet50_2':
    'https://x2paddle.bj.bcebos.com/vision/models/wide_resnet50_2-pt.pdparams',
    'wide_resnet101_2':
    'https://x2paddle.bj.bcebos.com/vision/models/wide_resnet101_2-pt.pdparams',
}


def conv3x3(in_planes: int,
            out_planes: int,
            stride: int=1,
            groups: int=1,
            dilation: int=1) -> nn.Conv2D:
    """3x3 convolution with padding"""
    return nn.Conv2D(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias_attr=False,
        dilation=dilation)


def conv1x1(in_planes: int, out_planes: int, stride: int=1) -> nn.Conv2D:
    """1x1 convolution"""
    return nn.Conv2D(
        in_planes, out_planes, kernel_size=1, stride=stride, bias_attr=False)


class BasicBlock(nn.Layer):
    expansion: int = 1

    def __init__(self,
                 inplanes: int,
                 planes: int,
                 stride: int=1,
                 downsample: Optional[nn.Layer]=None,
                 groups: int=1,
                 base_width: int=64,
                 dilation: int=1,
                 norm_layer: Optional[Callable[..., nn.Layer]]=None) -> None:
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError(
                'BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError(
                "Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = ReLU(True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Layer):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition"https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion: int = 4

    def __init__(self,
                 inplanes: int,
                 planes: int,
                 stride: int=1,
                 downsample: Optional[nn.Layer]=None,
                 groups: int=1,
                 base_width: int=64,
                 dilation: int=1,
                 norm_layer: Optional[Callable[..., nn.Layer]]=None) -> None:
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2D
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = ReLU(True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Layer):
    def __init__(self,
                 block: Type[Union[BasicBlock, Bottleneck]],
                 layers: List[int],
                 num_classes: int=1000,
                 zero_init_residual: bool=False,
                 groups: int=1,
                 width_per_group: int=64,
                 replace_stride_with_dilation: Optional[List[bool]]=None,
                 norm_layer: Optional[Callable[..., nn.Layer]]=None) -> None:
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2D
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, Ture]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(
                                 replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2D(
            3,
            self.inplanes,
            kernel_size=7,
            stride=2,
            padding=3,
            bias_attr=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = ReLU(True)
        self.maxpool = nn.MaxPool2D(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(
            block,
            128,
            layers[1],
            stride=2,
            dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(
            block,
            256,
            layers[2],
            stride=2,
            dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(
            block,
            512,
            layers[3],
            stride=2,
            dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2D((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.sublayers():
            if isinstance(m, nn.Conv2D):
                kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2D, nn.GroupNorm)):
                constant_init_(m.weight, 1)
                constant_init_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.sublayers():
                if isinstance(m, Bottleneck):
                    constant_init_(m.bn3.weight,
                                                0)  # type: ignore[arg-type]
                elif isinstance(m, BasicBlock):
                    constant_init_(m.bn2.weight,
                                                0)  # type: ignore[arg-type]

    def _make_layer(self,
                    block: Type[Union[BasicBlock, Bottleneck]],
                    planes: int,
                    blocks: int,
                    stride: int=1,
                    dilate: bool=False) -> nn.Sequential:
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion), )

        layers = []
        layers.append(
            block(self.inplanes, planes, stride, downsample, self.groups,
                  self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=self.dilation,
                    norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def _forward_impl(self, x: Tensor) -> Tensor:
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = paddle.flatten(x, 1)
        x = self.fc(x)

        return x

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)


def _resnet(arch: str,
            block: Type[Union[BasicBlock, Bottleneck]],
            layers: List[int],
            pretrained: bool,
            **kwargs: Any) -> ResNet:
    model = ResNet(block, layers, **kwargs)
    if pretrained:
        state_dict = paddle.load(get_weights_path_from_url(model_urls[arch]))
        model.load_dict(state_dict)
    return model


def resnet18(pretrained: bool=False, progress: bool=True,
             **kwargs: Any) -> ResNet:
    r"""ResNet-18 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet18', BasicBlock, [2, 2, 2, 2], pretrained, **kwargs)


def resnet34(pretrained: bool=False, progress: bool=True,
             **kwargs: Any) -> ResNet:
    r"""ResNet-34 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet34', BasicBlock, [3, 4, 6, 3], pretrained, **kwargs)


def resnet50(pretrained: bool=False, progress: bool=True,
             **kwargs: Any) -> ResNet:
    r"""ResNet-50 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet50', Bottleneck, [3, 4, 6, 3], pretrained, **kwargs)


def resnet101(pretrained: bool=False, progress: bool=True,
              **kwargs: Any) -> ResNet:
    r"""ResNet-101 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet101', Bottleneck, [3, 4, 23, 3], pretrained, **kwargs)


def resnet152(pretrained: bool=False, progress: bool=True,
              **kwargs: Any) -> ResNet:
    r"""ResNet-152 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet152', Bottleneck, [3, 8, 36, 3], pretrained, **kwargs)


def resnext50_32x4d(pretrained: bool=False, progress: bool=True,
                    **kwargs: Any) -> ResNet:
    r"""ResNeXt-50 32x4d model from
    `"Aggregated Residual Transformation for Deep Neural Networks" <https://arxiv.org/pdf/1611.05431.pdf>`_.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs['groups'] = 32
    kwargs['width_per_group'] = 4
    return _resnet('resnext50_32x4d', Bottleneck, [3, 4, 6, 3], pretrained,
                   **kwargs)


def resnext101_32x8d(pretrained: bool=False, progress: bool=True,
                     **kwargs: Any) -> ResNet:
    r"""ResNeXt-101 32x8d model from
    `"Aggregated Residual Transformation for Deep Neural Networks" <https://arxiv.org/pdf/1611.05431.pdf>`_.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs['groups'] = 32
    kwargs['width_per_group'] = 8
    return _resnet('resnext101_32x8d', Bottleneck, [3, 4, 23, 3], pretrained,
                   **kwargs)


def wide_resnet50_2(pretrained: bool=False, progress: bool=True,
                    **kwargs: Any) -> ResNet:
    r"""Wide ResNet-50-2 model from
    `"Wide Residual Networks" <https://arxiv.org/pdf/1605.07146.pdf>`_.
    The model is the same as ResNet except for the bottleneck number of channels
    which is twice larger in every block. The number of channels in outer 1x1
    convolutions is the same, e.g. last block in ResNet-50 has 2048-512-2048
    channels, and in Wide ResNet-50-2 has 2048-1024-2048.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs['width_per_group'] = 64 * 2
    return _resnet('wide_resnet50_2', Bottleneck, [3, 4, 6, 3], pretrained,
                   **kwargs)


def wide_resnet101_2(pretrained: bool=False, progress: bool=True,
                     **kwargs: Any) -> ResNet:
    r"""Wide ResNet-101-2 model from
    `"Wide Residual Networks" <https://arxiv.org/pdf/1605.07146.pdf>`_.
    The model is the same as ResNet except for the bottleneck number of channels
    which is twice larger in every block. The number of channels in outer 1x1
    convolutions is the same, e.g. last block in ResNet-50 has 2048-512-2048
    channels, and in Wide ResNet-50-2 has 2048-1024-2048.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs['width_per_group'] = 64 * 2
    return _resnet('wide_resnet101_2', Bottleneck, [3, 4, 23, 3], pretrained,
                   **kwargs)


import paddle
from paddle import nn
# from paddle.vision.models.resnet import BottleneckBlock
# from paddle.vision.models import ResNet
# from torchvision.models.resnet import ResNet, Bottleneck

class ResNet50Encoder(ResNet):
    def __init__(self, pretrained: bool = False):
        super().__init__(
            block=Bottleneck,
            layers=[3, 4, 6, 3],
            replace_stride_with_dilation=[False, False, True],
            norm_layer=None)
        
        if pretrained:
            # self.load_state_dict(torch.hub.load_state_dict_from_url(
            #     'https://download.pytorch.org/models/resnet50-0676ba61.pth'))
            load_weight = paddle.load("rvm_resnet50.pdparams")
            self.weight.set_value(load_weight)
            
        del self.avgpool
        del self.fc
        
    def forward_single_frame(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        f1 = x  # 1/2
        x = self.maxpool(x)
        x = self.layer1(x)
        f2 = x  # 1/4
        x = self.layer2(x)
        f3 = x  # 1/8
        x = self.layer3(x)
        x = self.layer4(x)
        f4 = x  # 1/16
        return [f1, f2, f3, f4]
    
    def forward_time_series(self, x):
        B, T = x.shape[:2]
        features = self.forward_single_frame(x.flatten(0, 1))
        
        # print("==before unflatten features", B, T, len(features), features[0].shape, features[0].shape)
        # for i,j in enumerate(features):
        #     print(i,j.shape)
        # tmpshape = [B, T] + features[0].shape[1:]
        # features = [f.unflatten(0, (B, T)) for f in features]
        features = [f.reshape([B, T] + f.shape[1:]) for f in features]
        # print("==after unflatten features", len(features), features[0].shape, features[0].shape)
        return features
    
    def forward(self, x):
        if x.ndim == 5:
            return self.forward_time_series(x)
        else:
            return self.forward_single_frame(x)
        
if __name__ == "__main__":
    a = paddle.randn((2, 3, 224, 224))
    testmodel = ResNet50Encoder()
    tmp = testmodel(a)
    print(f"输入数据shape:{a.shape} 输出数据长度:{len(tmp)}")
    for i in tmp:
        print(i.shape)
