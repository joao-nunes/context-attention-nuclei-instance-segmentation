from typing import Callable, List, Optional, Tuple, Dict
import warnings
from torch.nn import init
from torch import Tensor
from torchvision.ops.feature_pyramid_network import ExtraFPNBlock
from collections import OrderedDict
from utils import _log_api_usage_once
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import torch
from torch_geometric.nn import GATv2Conv


class ChannelAttention(nn.Module):

    def __init__(self, channel, reduction=16):

        super(ChannelAttention, self).__init__()

        self.maxpool = nn.AdaptiveMaxPool2d(1)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.se = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(channel // reduction, channel, 1, bias=False)
            )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):

        max_result = self.maxpool(x)
        avg_result = self.avgpool(x)
        max_out = self.se(max_result)
        avg_out = self.se(avg_result)
        output = self.sigmoid(max_out+avg_out)

        return output


class SpatialAttention(nn.Module):

    def __init__(self, kernel_size=5):

        super(SpatialAttention, self).__init__()
        
        self.conv1 = nn.Conv2d(
            2,
            1,
            kernel_size=kernel_size,
            padding=kernel_size//2
            )
        

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):

        max_result, _ = torch.max(
            x,
            dim=1,
            keepdim=True
            )

        avg_result = torch.mean(
            x,
            dim=1,
            keepdim=True
            )

        result = torch.cat(
            [max_result,
             avg_result],
            1
            )

        output = self.sigmoid(self.conv1(result)) 
       
        return output


class CBAMBlock(nn.Module):

    def __init__(self, channel=512, reduction=16, kernel_size=7):
        super(CBAMBlock, self).__init__()
        self.ca = ChannelAttention(channel=channel, reduction=reduction)
        #self.sa = SpatialAttention(kernel_size=kernel_size)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x):
        b, c, _, _ = x.size()
        residual = x
        out = x*self.ca(x)
        # out = out*self.sa(out)
        return out + residual


class CBAMFeaturePyramidNetwork(nn.Module):
    """
    Module that adds a FPN from on top of a set of feature maps.
    This is based on `"Feature Pyramid Network for Object Detection"
    <https://arxiv.org/abs/1612.03144>`_.
    The feature maps are currently supposed to be in increasing depth
    order.
    The input to the model is expected to be an OrderedDict[Tensor], containing
    the feature maps on top of which the FPN will be added.
    Args:
        in_channels_list (list[int]): number of channels for each feature map
        that is passed to the module
        out_channels (int): number of channels of the FPN representation
        extra_blocks (ExtraFPNBlock or None): if provided,
        extra operations will be performed. It is expected
        to take the fpn features, the original
        features and the names of the original features as input, and returns
        a new list of feature maps and their corresponding names
    Examples::
        >>> m = torchvision.ops.FeaturePyramidNetwork([10, 20, 30], 5)
        >>> # get some dummy data
        >>> x = OrderedDict()
        >>> x['feat0'] = torch.rand(1, 10, 64, 64)
        >>> x['feat2'] = torch.rand(1, 20, 16, 16)
        >>> x['feat3'] = torch.rand(1, 30, 8, 8)
        >>> # compute the FPN on top of x
        >>> output = m(x)
        >>> print([(k, v.shape) for k, v in output.items()])
        >>> # returns
        >>>   [('feat0', torch.Size([1, 5, 64, 64])),
        >>>    ('feat2', torch.Size([1, 5, 16, 16])),
        >>>    ('feat3', torch.Size([1, 5, 8, 8]))]
    """

    def __init__(
        self,
        in_channels_list: List[int],
        out_channels: int,
        extra_blocks: Optional[ExtraFPNBlock] = None,
    ):
        super().__init__()
        _log_api_usage_once(self)
        self.inner_blocks = nn.ModuleList()
        self.layer_blocks = nn.ModuleList()
        self.attention_blocks = nn.ModuleList()
        self.layer_attention_blocks = nn.ModuleList()
        for in_channels in in_channels_list:
            if in_channels == 0:
                raise ValueError("in_channels=0 is currently not supported")
            inner_block_module = nn.Conv2d(
                in_channels,
                out_channels,
                1)

            layer_block_module = nn.Conv2d(
                out_channels,
                out_channels,
                3,
                padding=1
                )
            attention_block_module = CBAMBlock(in_channels)
            self.attention_blocks.append(attention_block_module)
            self.inner_blocks.append(inner_block_module)
            self.layer_blocks.append(layer_block_module)

        # initialize parameters now to avoid modifying the initialization of top_blocks
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight, a=1)
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

        if extra_blocks is not None:
            assert isinstance(extra_blocks, ExtraFPNBlock)
        self.extra_blocks = extra_blocks

    def get_result_from_inner_blocks(self, x: Tensor, idx: int) -> Tensor:
        """
        This is equivalent to self.inner_blocks[idx](x),
        but torchscript doesn't support this yet
        """
        num_blocks = len(self.inner_blocks)
        if idx < 0:
            idx += num_blocks
        out = x
        for i, module in enumerate(self.inner_blocks):
            if i == idx:
                out = module(x)
        return out

    def get_result_from_attention_blocks(self, x: Tensor, idx: int) -> Tensor:
        """
        This is equivalent to self.attention_blocks[idx](x),
        but torchscript doesn't support this yet
        """
        num_blocks = len(self.attention_blocks)
        if idx < 0:
            idx += num_blocks
        for i, module in enumerate(self.attention_blocks):
            if i == idx:
                out = module(x)
        return out
    
    def get_result_from_layer_blocks(self, x: Tensor, idx: int) -> Tensor:
        """
        This is equivalent to self.layer_blocks[idx](x),
        but torchscript doesn't support this yet
        """
        num_blocks = len(self.layer_blocks)
        if idx < 0:
            idx += num_blocks
        out = x
        for i, module in enumerate(self.layer_blocks):
            if i == idx:
                out = module(x)
        return out

    def forward(self, x: Dict[str, Tensor]) -> Dict[str, Tensor]:
        """
        Computes the FPN for a set of feature maps.
        Args:
            x (OrderedDict[Tensor]): feature maps for each feature level.
        Returns:
            results (OrderedDict[Tensor]): feature maps after FPN layers.
                They are ordered from highest resolution first.
        """
        # unpack OrderedDict into two lists for easier handling
        names = list(x.keys())
        x = list(x.values())
        attention_map = self.get_result_from_attention_blocks(x[-1], -1)
        last_inner = self.get_result_from_inner_blocks(attention_map, -1)
        results = []
        results.append(self.get_result_from_layer_blocks(last_inner, -1))

        for idx in range(len(x) - 2, -1, -1):
            attention_map = self.get_result_from_attention_blocks(
                x[idx], idx)

            inner_lateral = self.get_result_from_inner_blocks(
                attention_map,
                idx
                )
            feat_shape = inner_lateral.shape[-2:]
            inner_top_down = F.interpolate(
                last_inner,
                size=feat_shape,
                mode="nearest")

            last_inner = inner_lateral + inner_top_down
            results.insert(
                0,
                self.get_result_from_layer_blocks(last_inner, idx))

        if self.extra_blocks is not None:
            results, names = self.extra_blocks(results, x, names)

        # make it back an OrderedDict
        out = OrderedDict([(k, v) for k, v in zip(names, results)])
        return out


class CBAMv2FeaturePyramidNetwork(nn.Module):
    """
    Module that adds a FPN from on top of a set of feature maps.
    This is based on `"Feature Pyramid Network for Object Detection"
    <https://arxiv.org/abs/1612.03144>`_.
    The feature maps are currently supposed to be in increasing depth
    order.
    The input to the model is expected to be an OrderedDict[Tensor], containing
    the feature maps on top of which the FPN will be added.
    Args:
        in_channels_list (list[int]): number of channels for each feature map
        that is passed to the module
        out_channels (int): number of channels of the FPN representation
        extra_blocks (ExtraFPNBlock or None): if provided,
        extra operations will be performed. It is expected
        to take the fpn features, the original
        features and the names of the original features as input, and returns
        a new list of feature maps and their corresponding names
    Examples::
        >>> m = torchvision.ops.FeaturePyramidNetwork([10, 20, 30], 5)
        >>> # get some dummy data
        >>> x = OrderedDict()
        >>> x['feat0'] = torch.rand(1, 10, 64, 64)
        >>> x['feat2'] = torch.rand(1, 20, 16, 16)
        >>> x['feat3'] = torch.rand(1, 30, 8, 8)
        >>> # compute the FPN on top of x
        >>> output = m(x)
        >>> print([(k, v.shape) for k, v in output.items()])
        >>> # returns
        >>>   [('feat0', torch.Size([1, 5, 64, 64])),
        >>>    ('feat2', torch.Size([1, 5, 16, 16])),
        >>>    ('feat3', torch.Size([1, 5, 8, 8]))]
    """

    def __init__(
        self,
        in_channels_list: List[int],
        out_channels: int,
        extra_blocks: Optional[ExtraFPNBlock] = None,
    ):
        super().__init__()
        _log_api_usage_once(self)
        self.inner_blocks = nn.ModuleList()
        self.layer_blocks = nn.ModuleList()
        self.attention_blocks = nn.ModuleList()
        self.layer_attention_blocks = nn.ModuleList()
        for in_channels in in_channels_list:
            if in_channels == 0:
                raise ValueError("in_channels=0 is currently not supported")
            inner_block_module = nn.Conv2d(
                in_channels,
                out_channels,
                1)

            layer_block_module = nn.Conv2d(
                out_channels,
                out_channels,
                3,
                padding=1
                )
            attention_block_module = CBAMBlock(out_channels)
            self.attention_blocks.append(attention_block_module)
            self.inner_blocks.append(inner_block_module)
            self.layer_blocks.append(layer_block_module)

        # initialize parameters now to avoid modifying the initialization of top_blocks
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight, a=1)
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

        if extra_blocks is not None:
            assert isinstance(extra_blocks, ExtraFPNBlock)
        self.extra_blocks = extra_blocks

    def get_result_from_inner_blocks(self, x: Tensor, idx: int) -> Tensor:
        """
        This is equivalent to self.inner_blocks[idx](x),
        but torchscript doesn't support this yet
        """
        num_blocks = len(self.inner_blocks)
        if idx < 0:
            idx += num_blocks
        out = x
        for i, module in enumerate(self.inner_blocks):
            if i == idx:
                out = module(x)
        return out

    def get_result_from_attention_blocks(self, x: Tensor, idx: int) -> Tensor:
        """
        This is equivalent to self.attention_blocks[idx](x),
        but torchscript doesn't support this yet
        """
        num_blocks = len(self.attention_blocks)
        if idx < 0:
            idx += num_blocks
        for i, module in enumerate(self.attention_blocks):
            if i == idx:
                out = module(x)
        return out
    
    def get_result_from_layer_blocks(self, x: Tensor, idx: int) -> Tensor:
        """
        This is equivalent to self.layer_blocks[idx](x),
        but torchscript doesn't support this yet
        """
        num_blocks = len(self.layer_blocks)
        if idx < 0:
            idx += num_blocks
        out = x
        for i, module in enumerate(self.layer_blocks):
            if i == idx:
                out = module(x)
        return out

    def forward(self, x: Dict[str, Tensor]) -> Dict[str, Tensor]:
        """
        Computes the FPN for a set of feature maps.
        Args:
            x (OrderedDict[Tensor]): feature maps for each feature level.
        Returns:
            results (OrderedDict[Tensor]): feature maps after FPN layers.
                They are ordered from highest resolution first.
        """
        # unpack OrderedDict into two lists for easier handling
        names = list(x.keys())
        x = list(x.values())
         
        last_inner = self.get_result_from_inner_blocks(x[-1], -1)
       
        results = []
        results.append( self.get_result_from_attention_blocks(
                    self.get_result_from_layer_blocks(last_inner, -1),
                    -1)
                )

        for idx in range(len(x) - 2, -1, -1):
            inner_lateral = self.get_result_from_inner_blocks(
                x[idx],
                idx
                )
            feat_shape = inner_lateral.shape[-2:]
            inner_top_down = F.interpolate(
                last_inner,
                size=feat_shape,
                mode="nearest")

            last_inner = inner_lateral + inner_top_down
            results.insert(
                0,
                self.get_result_from_attention_blocks(
                    self.get_result_from_layer_blocks(last_inner, idx),
                    idx)
                )

        if self.extra_blocks is not None:
            results, names = self.extra_blocks(results, x, names)

        # make it back an OrderedDict
        out = OrderedDict([(k, v) for k, v in zip(names, results)])
        return out

class CBAMv3FeaturePyramidNetwork(nn.Module):
    """
    Module that adds a FPN from on top of a set of feature maps.
    This is based on `"Feature Pyramid Network for Object Detection"
    <https://arxiv.org/abs/1612.03144>`_.
    The feature maps are currently supposed to be in increasing depth
    order.
    The input to the model is expected to be an OrderedDict[Tensor], containing
    the feature maps on top of which the FPN will be added.
    Args:
        in_channels_list (list[int]): number of channels for each feature map
        that is passed to the module
        out_channels (int): number of channels of the FPN representation
        extra_blocks (ExtraFPNBlock or None): if provided,
        extra operations will be performed. It is expected
        to take the fpn features, the original
        features and the names of the original features as input, and returns
        a new list of feature maps and their corresponding names
    Examples::
        >>> m = torchvision.ops.FeaturePyramidNetwork([10, 20, 30], 5)
        >>> # get some dummy data
        >>> x = OrderedDict()
        >>> x['feat0'] = torch.rand(1, 10, 64, 64)
        >>> x['feat2'] = torch.rand(1, 20, 16, 16)
        >>> x['feat3'] = torch.rand(1, 30, 8, 8)
        >>> # compute the FPN on top of x
        >>> output = m(x)
        >>> print([(k, v.shape) for k, v in output.items()])
        >>> # returns
        >>>   [('feat0', torch.Size([1, 5, 64, 64])),
        >>>    ('feat2', torch.Size([1, 5, 16, 16])),
        >>>    ('feat3', torch.Size([1, 5, 8, 8]))]
    """

    def __init__(
        self,
        in_channels_list: List[int],
        out_channels: int,
        extra_blocks: Optional[ExtraFPNBlock] = None,
    ):
        super().__init__()
        _log_api_usage_once(self)
        self.inner_blocks = nn.ModuleList()
        self.layer_blocks = nn.ModuleList()
        self.inner_attention_blocks = nn.ModuleList()
        self.attention_blocks = nn.ModuleList()
        self.layer_attention_blocks = nn.ModuleList()
        for in_channels in in_channels_list:
            if in_channels == 0:
                raise ValueError("in_channels=0 is currently not supported")
            inner_block_module = nn.Conv2d(
                in_channels,
                out_channels,
                1)

            layer_block_module = nn.Conv2d(
                out_channels,
                out_channels,
                3,
                padding=1
                )
            attention_block_module = CBAMBlock(out_channels)
            inner_attention_block_module = CBAMBlock(in_channels)
            self.attention_blocks.append(attention_block_module)
            self.inner_attention_blocks.append(inner_attention_block_module)
            
            self.inner_blocks.append(inner_block_module)
            self.layer_blocks.append(layer_block_module)

        # initialize parameters now to avoid modifying the initialization of top_blocks
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight, a=1)
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

        if extra_blocks is not None:
            assert isinstance(extra_blocks, ExtraFPNBlock)
        self.extra_blocks = extra_blocks

    def get_result_from_inner_blocks(self, x: Tensor, idx: int) -> Tensor:
        """
        This is equivalent to self.inner_blocks[idx](x),
        but torchscript doesn't support this yet
        """
        num_blocks = len(self.inner_blocks)
        if idx < 0:
            idx += num_blocks
        out = x
        for i, module in enumerate(self.inner_blocks):
            if i == idx:
                out = module(x)
        return out

    def get_result_from_attention_blocks(self, x: Tensor, idx: int) -> Tensor:
        """
        This is equivalent to self.attention_blocks[idx](x),
        but torchscript doesn't support this yet
        """
        num_blocks = len(self.attention_blocks)
        if idx < 0:
            idx += num_blocks
        for i, module in enumerate(self.attention_blocks):
            if i == idx:
                out = module(x)
        return out
    
    def get_result_from_inner_attention_blocks(self, x: Tensor, idx: int) -> Tensor:
        """
        This is equivalent to self.attention_blocks[idx](x),
        but torchscript doesn't support this yet
        """
        num_blocks = len(self.inner_attention_blocks)
        if idx < 0:
            idx += num_blocks
        for i, module in enumerate(self.inner_attention_blocks):
            if i == idx:
                out = module(x)
        return out
    
    def get_result_from_layer_blocks(self, x: Tensor, idx: int) -> Tensor:
        """
        This is equivalent to self.layer_blocks[idx](x),
        but torchscript doesn't support this yet
        """
        num_blocks = len(self.layer_blocks)
        if idx < 0:
            idx += num_blocks
        out = x
        for i, module in enumerate(self.layer_blocks):
            if i == idx:
                out = module(x)
        return out

    def forward(self, x: Dict[str, Tensor]) -> Dict[str, Tensor]:
        """
        Computes the FPN for a set of feature maps.
        Args:
            x (OrderedDict[Tensor]): feature maps for each feature level.
        Returns:
            results (OrderedDict[Tensor]): feature maps after FPN layers.
                They are ordered from highest resolution first.
        """
        # unpack OrderedDict into two lists for easier handling
        names = list(x.keys())
        x = list(x.values())
        attention_map = self.get_result_from_inner_attention_blocks(x[-1], -1)
        last_inner = self.get_result_from_inner_blocks(attention_map, -1)
        results = []
        results.append( self.get_result_from_attention_blocks(
                    self.get_result_from_layer_blocks(last_inner, -1),
                    -1)
                )

        for idx in range(len(x) - 2, -1, -1):
            attention_map = self.get_result_from_inner_attention_blocks(
                x[idx], idx)

            inner_lateral = self.get_result_from_inner_blocks(
                attention_map,
                idx
                )
            feat_shape = inner_lateral.shape[-2:]
            inner_top_down = F.interpolate(
                last_inner,
                size=feat_shape,
                mode="nearest")

            last_inner = inner_lateral + inner_top_down
            results.insert(
                0,
                self.get_result_from_attention_blocks(
                    self.get_result_from_layer_blocks(last_inner, idx),
                    idx)
                )

        if self.extra_blocks is not None:
            results, names = self.extra_blocks(results, x, names)

        # make it back an OrderedDict
        out = OrderedDict([(k, v) for k, v in zip(names, results)])
        return out

class GLSGRFeaturePyramidNetwork(nn.Module):
    """
    Module that adds a FPN from on top of a set of feature maps. This is based on
    `"Feature Pyramid Network for Object Detection" <https://arxiv.org/abs/1612.03144>`_.
    The feature maps are currently supposed to be in increasing depth
    order.
    The input to the model is expected to be an OrderedDict[Tensor], containing
    the feature maps on top of which the FPN will be added.
    Args:
        in_channels_list (list[int]): number of channels for each feature map that
            is passed to the module
        out_channels (int): number of channels of the FPN representation
        extra_blocks (ExtraFPNBlock or None): if provided, extra operations will
            be performed. It is expected to take the fpn features, the original
            features and the names of the original features as input, and returns
            a new list of feature maps and their corresponding names
        norm_layer (callable, optional): Module specifying the normalization layer to use. Default: None
    Examples::
        >>> m = torchvision.ops.FeaturePyramidNetwork([10, 20, 30], 5)
        >>> # get some dummy data
        >>> x = OrderedDict()
        >>> x['feat0'] = torch.rand(1, 10, 64, 64)
        >>> x['feat2'] = torch.rand(1, 20, 16, 16)
        >>> x['feat3'] = torch.rand(1, 30, 8, 8)
        >>> # compute the FPN on top of x
        >>> output = m(x)
        >>> print([(k, v.shape) for k, v in output.items()])
        >>> # returns
        >>>   [('feat0', torch.Size([1, 5, 64, 64])),
        >>>    ('feat2', torch.Size([1, 5, 16, 16])),
        >>>    ('feat3', torch.Size([1, 5, 8, 8]))]
    """

    def __init__(
        self,
        in_channels_list: List[int],
        out_channels: int,
        extra_blocks: Optional[ExtraFPNBlock] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        kernel_size: int = 5,
        stride: int = 5,
        node_connectivities : List[int] = [7, 16, 20, 25]
        #node_connectivities : List[int] = [3, 10, 20, 30]
    ):
        super().__init__()
        _log_api_usage_once(self)
        
        self.inner_blocks = nn.ModuleList()
        self.layer_blocks = nn.ModuleList()
        self.layer_attention_blocks = nn.ModuleList()
        self.conv_blocks = nn.ModuleList()
        self.kernel_size = kernel_size
        self.stride = stride
        
        for j, in_channels in enumerate(in_channels_list):
            if in_channels == 0:
                raise ValueError("in_channels=0 is currently not supported")
            inner_block_module = nn.Conv2d(
                in_channels,
                out_channels,
                1)

            layer_block_module = nn.Conv2d(
                out_channels,
                out_channels,
                3,
                padding=1
                )
            conv_block_module = nn.ConvTranspose2d(in_channels,
                                                   in_channels,
                                                   kernel_size,
                                                   stride)
            
            attention_block_module = GNN(in_channels, in_channels, k=node_connectivities[0])
            self.inner_blocks.append(inner_block_module)
            self.layer_blocks.append(layer_block_module)
            self.layer_attention_blocks.append(attention_block_module)
            self.conv_blocks.append(conv_block_module)


        # initialize parameters now to avoid modifying the initialization of top_blocks
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight, a=1)
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

        if extra_blocks is not None:
            assert isinstance(extra_blocks, ExtraFPNBlock)
        self.extra_blocks = extra_blocks

    def get_result_from_inner_blocks(self, x: Tensor, idx: int) -> Tensor:
        """
        This is equivalent to self.inner_blocks[idx](x),
        but torchscript doesn't support this yet
        """
        num_blocks = len(self.inner_blocks)
        if idx < 0:
            idx += num_blocks
        out = x
        for i, module in enumerate(self.inner_blocks):
            if i == idx:
                out = module(x)
        return out
    
    def get_result_from_attention_blocks(self, x: Tensor, idx: int) -> Tensor:
        """
        This is equivalent to self.inner_blocks[idx](x),
        but torchscript doesn't support this yet
        """
        num_blocks = len(self.layer_attention_blocks)
        if idx < 0:
            idx += num_blocks
        out = x
        for i, module in enumerate(self.layer_attention_blocks):
            if i == idx:
                h = module(x)
                dim = int(np.sqrt(h.shape[0]))
                h = torch.reshape(h, (dim, dim, h.shape[1])).swapdims(-1,0).unsqueeze(0)     
                out = F.relu(self.conv_blocks[i](h))
        return out

    def get_result_from_layer_blocks(self, x: Tensor, idx: int) -> Tensor:
        """
        This is equivalent to self.layer_blocks[idx](x),
        but torchscript doesn't support this yet
        """
        num_blocks = len(self.layer_blocks)
        if idx < 0:
            idx += num_blocks
        out = x
        for i, module in enumerate(self.layer_blocks):
            if i == idx:
                out = module(x)
        return out

    def forward(self, x: Dict[str, Tensor]) -> Dict[str, Tensor]:
        """
        Computes the FPN for a set of feature maps.
        Args:
            x (OrderedDict[Tensor]): feature maps for each feature level.
        Returns:
            results (OrderedDict[Tensor]): feature maps after FPN layers.
                They are ordered from highest resolution first.
        """
        # unpack OrderedDict into two lists for easier handling
        names = list(x.keys())
        x = list(x.values())

        
        attention_features = self.get_result_from_attention_blocks(x[-1], -1)      
        # remove padding
        pad = (
            round(((x[-1].size(-2)*(self.stride - 1) % self.stride))/2), # left_pad
            ((x[-1].size(-2)*(self.stride - 1) % self.stride))-round(((x[-1].size(-2)*(self.stride - 1) % self.stride))/2), # right_pad,
            round(((x[-1].size(-1)*(self.stride - 1) % self.stride))/2), # top_pad
            ((x[-1].size(-1)*(self.stride - 1) % self.stride))-round(((x[-1].size(-1)*(self.stride - 1) % self.stride))/2) # bottom_pad
            )
        _, _, h, w = attention_features.shape
        attention_features = attention_features[..., pad[2]: h - pad[3], pad[0]:w-pad[1]]
        x1 = (x[-1] * attention_features) + x[-1]
        last_inner = self.get_result_from_inner_blocks(x1, -1)

        results = []
        results.append(self.get_result_from_layer_blocks(last_inner, -1))

        for idx in range(len(x) - 2, -1, -1):
            
            attention_features = self.get_result_from_attention_blocks(x[idx], idx)
            # remove padding
            # pad x to the nearest multiple of stride
            pad = (
            round(((x[idx].size(-2)*(self.stride - 1) % self.stride))/2), # left_pad
            ((x[idx].size(-2)*(self.stride - 1) % self.stride))-round(((x[idx].size(-2)*(self.stride - 1) % self.stride))/2), # right_pad,
            round(((x[idx].size(-1)*(self.stride - 1) % self.stride))/2), # top_pad
            ((x[idx].size(-1)*(self.stride - 1) % self.stride))-round(((x[idx].size(-1)*(self.stride - 1) % self.stride))/2) # bottom_pad
            )
            _, _, h, w = attention_features.shape
            attention_features = attention_features[..., pad[2]: h - pad[3], pad[0]:w-pad[1]]
            x1 = (x[idx] * attention_features) + x[idx]
            inner_lateral = self.get_result_from_inner_blocks(x1, idx)

            feat_shape = inner_lateral.shape[-2:]
            inner_top_down = F.interpolate(last_inner, size=feat_shape, mode="nearest")
            last_inner = inner_lateral + inner_top_down
            results.insert(0, self.get_result_from_layer_blocks(last_inner, idx))

        if self.extra_blocks is not None:
            results, names = self.extra_blocks(results, x, names)

        # make it back an OrderedDict
        out = OrderedDict([(k, v) for k, v in zip(names, results)])

        return out
    
class GLSGRv2FeaturePyramidNetwork(nn.Module):
    """
    Module that adds a FPN from on top of a set of feature maps. This is based on
    `"Feature Pyramid Network for Object Detection" <https://arxiv.org/abs/1612.03144>`_.
    The feature maps are currently supposed to be in increasing depth
    order.
    The input to the model is expected to be an OrderedDict[Tensor], containing
    the feature maps on top of which the FPN will be added.
    Args:
        in_channels_list (list[int]): number of channels for each feature map that
            is passed to the module
        out_channels (int): number of channels of the FPN representation
        extra_blocks (ExtraFPNBlock or None): if provided, extra operations will
            be performed. It is expected to take the fpn features, the original
            features and the names of the original features as input, and returns
            a new list of feature maps and their corresponding names
        norm_layer (callable, optional): Module specifying the normalization layer to use. Default: None
    Examples::
        >>> m = torchvision.ops.FeaturePyramidNetwork([10, 20, 30], 5)
        >>> # get some dummy data
        >>> x = OrderedDict()
        >>> x['feat0'] = torch.rand(1, 10, 64, 64)
        >>> x['feat2'] = torch.rand(1, 20, 16, 16)
        >>> x['feat3'] = torch.rand(1, 30, 8, 8)
        >>> # compute the FPN on top of x
        >>> output = m(x)
        >>> print([(k, v.shape) for k, v in output.items()])
        >>> # returns
        >>>   [('feat0', torch.Size([1, 5, 64, 64])),
        >>>    ('feat2', torch.Size([1, 5, 16, 16])),
        >>>    ('feat3', torch.Size([1, 5, 8, 8]))]
    """

    def __init__(
        self,
        in_channels_list: List[int],
        out_channels: int,
        extra_blocks: Optional[ExtraFPNBlock] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        kernel_size: int = 5,
        stride: int = 5,
        node_connectivities : List[int] = [7, 16, 20, 25]
        #node_connectivities : List[int] = [3, 10, 20, 30]
    ):
        super().__init__()
        _log_api_usage_once(self)
        
        self.inner_blocks = nn.ModuleList()
        self.layer_blocks = nn.ModuleList()
        self.layer_attention_blocks = nn.ModuleList()
        self.conv_blocks = nn.ModuleList()
        self.kernel_size = kernel_size
        self.stride = stride
        
        for j, in_channels in enumerate(in_channels_list):
            if in_channels == 0:
                raise ValueError("in_channels=0 is currently not supported")
            inner_block_module = nn.Conv2d(
                in_channels,
                out_channels,
                1)

            layer_block_module = nn.Conv2d(
                out_channels,
                out_channels,
                3,
                padding=1
                )
            conv_block_module = nn.ConvTranspose2d(out_channels,
                                                   out_channels,
                                                   kernel_size,
                                                   stride)
            
            attention_block_module = GNN(out_channels, out_channels, k=node_connectivities[0])
            self.inner_blocks.append(inner_block_module)
            self.layer_blocks.append(layer_block_module)
            self.layer_attention_blocks.append(attention_block_module)
            self.conv_blocks.append(conv_block_module)


        # initialize parameters now to avoid modifying the initialization of top_blocks
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight, a=1)
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

        if extra_blocks is not None:
            assert isinstance(extra_blocks, ExtraFPNBlock)
        self.extra_blocks = extra_blocks

    def get_result_from_inner_blocks(self, x: Tensor, idx: int) -> Tensor:
        """
        This is equivalent to self.inner_blocks[idx](x),
        but torchscript doesn't support this yet
        """
        num_blocks = len(self.inner_blocks)
        if idx < 0:
            idx += num_blocks
        out = x
        for i, module in enumerate(self.inner_blocks):
            if i == idx:
                out = module(x)
        return out
    
    def get_result_from_attention_blocks(self, x: Tensor, idx: int) -> Tensor:
        """
        This is equivalent to self.inner_blocks[idx](x),
        but torchscript doesn't support this yet
        """
        num_blocks = len(self.layer_attention_blocks)
        if idx < 0:
            idx += num_blocks
        out = x
        for i, module in enumerate(self.layer_attention_blocks):
            if i == idx:
                h = module(x)
                dim = int(np.sqrt(h.shape[0]))
                h = torch.reshape(h, (dim, dim, h.shape[1])).swapdims(-1,0).unsqueeze(0)     
                out = F.relu(self.conv_blocks[i](h))
        return out

    def get_result_from_layer_blocks(self, x: Tensor, idx: int) -> Tensor:
        """
        This is equivalent to self.layer_blocks[idx](x),
        but torchscript doesn't support this yet
        """
        num_blocks = len(self.layer_blocks)
        if idx < 0:
            idx += num_blocks
        out = x
        for i, module in enumerate(self.layer_blocks):
            if i == idx:
                out = module(x)
        return out

    def forward(self, x: Dict[str, Tensor]) -> Dict[str, Tensor]:
        """
        Computes the FPN for a set of feature maps.
        Args:
            x (OrderedDict[Tensor]): feature maps for each feature level.
        Returns:
            results (OrderedDict[Tensor]): feature maps after FPN layers.
                They are ordered from highest resolution first.
        """
        # unpack OrderedDict into two lists for easier handling
        names = list(x.keys())
        x = list(x.values())

        last_inner = self.get_result_from_inner_blocks(x[-1], -1)
        x1 = self.get_result_from_layer_blocks(last_inner, -1)
        attention_features = self.get_result_from_attention_blocks(x1, -1)      
        # remove padding
        pad = (
            round(((x[-1].size(-2)*(self.stride - 1) % self.stride))/2), # left_pad
            ((x[-1].size(-2)*(self.stride - 1) % self.stride))-round(((x[-1].size(-2)*(self.stride - 1) % self.stride))/2), # right_pad,
            round(((x[-1].size(-1)*(self.stride - 1) % self.stride))/2), # top_pad
            ((x[-1].size(-1)*(self.stride - 1) % self.stride))-round(((x[-1].size(-1)*(self.stride - 1) % self.stride))/2) # bottom_pad
            )
        _, _, h, w = attention_features.shape
        attention_features = attention_features[..., pad[2]: h - pad[3], pad[0]:w-pad[1]]
        x2 = (x1 * attention_features) + x1
        results = []
        results.append(x2)

        for idx in range(len(x) - 2, -1, -1):
            
            inner_lateral = self.get_result_from_inner_blocks(x[idx], idx)
            feat_shape = inner_lateral.shape[-2:]
            inner_top_down = F.interpolate(last_inner, size=feat_shape, mode="nearest")
            last_inner = inner_lateral + inner_top_down
            x1 = self.get_result_from_layer_blocks(last_inner, idx)
            attention_features = self.get_result_from_attention_blocks(x1, idx)      
            # remove padding
            pad = (
            round(((x[idx].size(-2)*(self.stride - 1) % self.stride))/2), # left_pad
            ((x[idx].size(-2)*(self.stride - 1) % self.stride))-round(((x[idx].size(-2)*(self.stride - 1) % self.stride))/2), # right_pad,
            round(((x[idx].size(-1)*(self.stride - 1) % self.stride))/2), # top_pad
            ((x[idx].size(-1)*(self.stride - 1) % self.stride))-round(((x[idx].size(-1)*(self.stride - 1) % self.stride))/2) # bottom_pad
            )
            _, _, h, w = attention_features.shape
            attention_features = attention_features[..., pad[2]: h - pad[3], pad[0]:w-pad[1]]
            x2 = (x1 * attention_features) + x1
            results.insert(0, x2)

        if self.extra_blocks is not None:
            results, names = self.extra_blocks(results, x, names)

        # make it back an OrderedDict
        out = OrderedDict([(k, v) for k, v in zip(names, results)])

        return out


class GLSGRv3FeaturePyramidNetwork(nn.Module):
    """
    Module that adds a FPN from on top of a set of feature maps. This is based on
    `"Feature Pyramid Network for Object Detection" <https://arxiv.org/abs/1612.03144>`_.
    The feature maps are currently supposed to be in increasing depth
    order.
    The input to the model is expected to be an OrderedDict[Tensor], containing
    the feature maps on top of which the FPN will be added.
    Args:
        in_channels_list (list[int]): number of channels for each feature map that
            is passed to the module
        out_channels (int): number of channels of the FPN representation
        extra_blocks (ExtraFPNBlock or None): if provided, extra operations will
            be performed. It is expected to take the fpn features, the original
            features and the names of the original features as input, and returns
            a new list of feature maps and their corresponding names
        norm_layer (callable, optional): Module specifying the normalization layer to use. Default: None
    Examples::
        >>> m = torchvision.ops.FeaturePyramidNetwork([10, 20, 30], 5)
        >>> # get some dummy data
        >>> x = OrderedDict()
        >>> x['feat0'] = torch.rand(1, 10, 64, 64)
        >>> x['feat2'] = torch.rand(1, 20, 16, 16)
        >>> x['feat3'] = torch.rand(1, 30, 8, 8)
        >>> # compute the FPN on top of x
        >>> output = m(x)
        >>> print([(k, v.shape) for k, v in output.items()])
        >>> # returns
        >>>   [('feat0', torch.Size([1, 5, 64, 64])),
        >>>    ('feat2', torch.Size([1, 5, 16, 16])),
        >>>    ('feat3', torch.Size([1, 5, 8, 8]))]
    """

    def __init__(
        self,
        in_channels_list: List[int],
        out_channels: int,
        extra_blocks: Optional[ExtraFPNBlock] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        kernel_size: int = 5,
        stride: int = 5,
        node_connectivities : List[int] = [7, 16, 20, 25]
        #node_connectivities : List[int] = [3, 10, 20, 30]
    ):
        super().__init__()
        _log_api_usage_once(self)
        
        self.inner_blocks = nn.ModuleList()
        self.layer_blocks = nn.ModuleList()
        self.inner_attention_blocks = nn.ModuleList()
        self.layer_attention_blocks = nn.ModuleList()
        self.inner_conv_blocks = nn.ModuleList()
        self.layer_conv_blocks = nn.ModuleList()
        
        self.kernel_size = kernel_size
        self.stride = stride
        
        for j, in_channels in enumerate(in_channels_list):
            if in_channels == 0:
                raise ValueError("in_channels=0 is currently not supported")
            inner_block_module = nn.Conv2d(
                in_channels,
                out_channels,
                1)

            layer_block_module = nn.Conv2d(
                out_channels,
                out_channels,
                3,
                padding=1
                )
            inner_conv_block_module = nn.ConvTranspose2d(in_channels,
                                                   in_channels,
                                                   kernel_size,
                                                   stride,
                                                   )
            
            layer_conv_block_module = nn.ConvTranspose2d(out_channels,
                                                   out_channels,
                                                   kernel_size,
                                                   stride,
                                                   )
            
            inner_attention_block_module = GNN(in_channels, in_channels, k=node_connectivities[0])
            layer_attention_block_module = GNN(out_channels, out_channels, k=node_connectivities[0])
            
            self.inner_blocks.append(inner_block_module)
            self.layer_blocks.append(layer_block_module)
            
            self.inner_attention_blocks.append(inner_attention_block_module)
            self.layer_attention_blocks.append(layer_attention_block_module)
            
            self.inner_conv_blocks.append(inner_conv_block_module)
            self.layer_conv_blocks.append(layer_conv_block_module)

        # initialize parameters now to avoid modifying the initialization of top_blocks
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight, a=1)
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

        if extra_blocks is not None:
            assert isinstance(extra_blocks, ExtraFPNBlock)
        self.extra_blocks = extra_blocks

    def get_result_from_inner_blocks(self, x: Tensor, idx: int) -> Tensor:
        """
        This is equivalent to self.inner_blocks[idx](x),
        but torchscript doesn't support this yet
        """
        num_blocks = len(self.inner_blocks)
        if idx < 0:
            idx += num_blocks
        out = x
        for i, module in enumerate(self.inner_blocks):
            if i == idx:
                out = module(x)
        return out
    
    def get_result_from_layer_attention_blocks(self, x: Tensor, idx: int) -> Tensor:
        """
        This is equivalent to self.inner_blocks[idx](x),
        but torchscript doesn't support this yet
        """
        num_blocks = len(self.layer_attention_blocks)
        if idx < 0:
            idx += num_blocks
        out = x
        for i, module in enumerate(self.layer_attention_blocks):
            if i == idx:
                h = module(x)
                _,_,height,width = x.shape
                dim = (int(np.floor(height/self.stride)), 
                       int(np.floor(width/self.stride))
                       )
                h = torch.reshape(h, (dim[0], dim[1], h.shape[1])).moveaxis(-1,0).unsqueeze(0)    
                out = F.relu(self.layer_conv_blocks[i](h))
        return out
    
    def get_result_from_inner_attention_blocks(self, x: Tensor, idx: int) -> Tensor:
        """
        This is equivalent to self.inner_blocks[idx](x),
        but torchscript doesn't support this yet
        """
        num_blocks = len(self.inner_attention_blocks)
        if idx < 0:
            idx += num_blocks
        out = x
        for i, module in enumerate(self.inner_attention_blocks):
            if i == idx:
                h = module(x)
                _,_,height,width = x.shape
                dim = (height//self.stride, width//self.stride)
                h = torch.reshape(h, (dim[0], dim[1], h.shape[1])).moveaxis(-1,0).unsqueeze(0)     
                out = F.relu(self.inner_conv_blocks[i](h))
        return out

    def get_result_from_layer_blocks(self, x: Tensor, idx: int) -> Tensor:
        """
        This is equivalent to self.layer_blocks[idx](x),
        but torchscript doesn't support this yet
        """
        num_blocks = len(self.layer_blocks)
        if idx < 0:
            idx += num_blocks
        out = x
        for i, module in enumerate(self.layer_blocks):
            if i == idx:
                out = module(x)
        return out

    def forward(self, x: Dict[str, Tensor]) -> Dict[str, Tensor]:
        """
        Computes the FPN for a set of feature maps.
        Args:
            x (OrderedDict[Tensor]): feature maps for each feature level.
        Returns:
            results (OrderedDict[Tensor]): feature maps after FPN layers.
                They are ordered from highest resolution first.
        """
        # unpack OrderedDict into two lists for easier handling
        names = list(x.keys())
        x = list(x.values())
                # remove padding

        inner_attention_features = self.get_result_from_inner_attention_blocks(x[-1], -1)
        
        _, _, h, w = inner_attention_features.shape
        _,_,h_,w_ = x[-1].shape
        
        left_pad = (w_ - w) // 2
        right_pad = (w_ - w) - left_pad
        top_pad = (h_ - h) // 2
        bottom_pad = (h_ - h) - top_pad
        
        pad = (left_pad, right_pad, top_pad, bottom_pad)
        
        inner_attention_features = nn.ConstantPad2d(pad,0)(inner_attention_features)
        
        x_ = (x[-1] * inner_attention_features) + x[-1]
        last_inner = self.get_result_from_inner_blocks(x_, -1)
        x1 = self.get_result_from_layer_blocks(last_inner, -1)
        
        layer_attention_features = self.get_result_from_layer_attention_blocks(x1, -1)
        
        layer_attention_features = nn.ConstantPad2d(pad,0)(layer_attention_features)
        x2 = (x1 * layer_attention_features) + x1
        results = []
        results.append(x2)

        for idx in range(len(x) - 2, -1, -1):
      
            inner_attention_features = self.get_result_from_inner_attention_blocks(x[idx], idx)

            _, _, h, w = inner_attention_features.shape
            _,_,h_,w_ = x[idx].shape
        
            left_pad = (w_ - w) // 2
            right_pad = (w_ - w) - left_pad
            top_pad = (h_ - h) // 2
            bottom_pad = (h_ - h) - top_pad
        
            pad = (left_pad, right_pad, top_pad, bottom_pad)      
            
            inner_attention_features = nn.ConstantPad2d(pad, 0)(inner_attention_features)
            
            x_ = (x[idx] * inner_attention_features) + x[idx]
            
            inner_lateral = self.get_result_from_inner_blocks(x_, idx)
            feat_shape = inner_lateral.shape[-2:]
            inner_top_down = F.interpolate(last_inner, size=feat_shape, mode="nearest")
            last_inner = inner_lateral + inner_top_down
            x1 = self.get_result_from_layer_blocks(last_inner, idx)
            
            layer_attention_features = self.get_result_from_layer_attention_blocks(x1, idx)      
            layer_attention_features = nn.ConstantPad2d(pad,0)(layer_attention_features)
            
            x2 = (x1 * layer_attention_features) + x1
            results.insert(0, x2)

        if self.extra_blocks is not None:
            results, names = self.extra_blocks(results, x, names)

        # make it back an OrderedDict
        out = OrderedDict([(k, v) for k, v in zip(names, results)])

        return out


class RoIGAtR(nn.Module):
    """
    Standard heads for FPN-based models
    Args:
        in_channels (int): number of input channels
        representation_size (int): size of the intermediate representation
    """

    def __init__(self, in_channels, representation_size, heads=8, k=6):
        super().__init__()
        
        self.gat1 = GATv2Conv(in_channels, representation_size, heads=heads, dropout=0.45)
        self.gat2 = GATv2Conv(representation_size*heads, representation_size, heads=1, dropout=0.45)
        self.elu1 = nn.PReLU()
        self.elu2 = nn.PReLU()
        self.k = k
        
    def forward(self, x):

        adj_matrix = x @ x.T
        if x.shape[0] > self.k:
            topk, indices = torch.topk(adj_matrix, self.k)
        else:
            topk, indices = torch.topk(adj_matrix, x.shape[0])
        A = torch.zeros(adj_matrix.shape).to(x.device)
        A = A.scatter(1, indices, topk)
        edge_index = torch.nonzero(torch.triu(A)).T
        # rearrange the node pair indices to be from source to target
        edge_index = torch.stack([edge_index[1], edge_index[0]])
        # Global Graph Attention based on feature similarity
        
        h = self.gat1(x, edge_index)
        h = self.elu1(h)
        
        h = self.gat2(h, edge_index)
        h = self.elu2(h)
        
        return h

class GNN(nn.Module):
    """
    Standard heads for FPN-based models
    Args:
        in_channels (int): number of input channels
        representation_size (int): size of the intermediate representation
    """

    def __init__(self, in_channels, representation_size, kernel_size=5, stride=5, heads=8, k=3):
        super().__init__()

        self.kernel_size = kernel_size
        self.stride = stride
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=5, stride=1)
        self.fc1 = nn.Linear(in_channels*kernel_size**2, representation_size)
        self.fc2 = nn.Linear(representation_size, representation_size)
        self.fc3 = nn.Linear(2*representation_size, representation_size)
        
        self.gat1 = GATv2Conv(representation_size, representation_size, heads=heads, dropout=0.4)
        self.gat2 = GATv2Conv(representation_size*heads, representation_size, heads=1, dropout=0.4)
         
        self.gat3 = GATv2Conv(representation_size, representation_size, heads=heads, dropout=0.4)
        self.gat4 = GATv2Conv(representation_size*heads, representation_size, heads=1, dropout=0.4)
        self.pos_encoder = PositionalEncoding2d(stride=5)
        self.k = k
        
    def forward(self, x):

        # Global Sparse Relation Learner
        kernel_size, stride = self.kernel_size, self.stride

        patches = x.unfold(2, kernel_size, stride).unfold(3, kernel_size, stride)
        patches = patches.contiguous().view(patches.size(0), patches.size(1), -1, kernel_size, kernel_size)
        patches = torch.swapdims(patches, 1, 2).squeeze()
        
        embeddings = F.relu(self.conv1(patches)).squeeze()
        embeddings = F.relu(self.fc2(embeddings))
        
        # pos_embeddings = self.pos_encoder(x, d=embeddings.shape[-1])
 
        adj_matrix = embeddings @ embeddings.T
        
        #embeddings = embeddings + pos_embeddings.flatten(end_dim=-2).squeeze()

        topk, indices = torch.topk(adj_matrix, self.k)
        A = torch.zeros(adj_matrix.shape).to(x.device)
        A = A.scatter(1, indices, topk)
        edge_index = torch.nonzero(torch.triu(A)).T
        # rearrange the node pair indices to be from source to target
        edge_index = torch.stack([edge_index[1], edge_index[0]])
        # Global Graph Attention based on feature similarity
        
        h = self.gat1(embeddings, edge_index)
        h = F.elu(h)
        
        h = self.gat2(h, edge_index)
        h = F.elu(h)
        
        ## Local Sparse Relation Learner
        ## TODO: Use superpixels instead of regular patches
        _, _, height, width = x.shape
        region_centers = torch.stack(
            [torch.stack(
                [torch.tensor(i).to(torch.float),
                 torch.tensor(j).to(torch.float)
                ]) for i in range(0, (height-height%stride), stride)
             for j in range(0, (width-width%stride), stride)])
       
        dist = torch.cdist(
            region_centers,
            region_centers,
            p=2.0).to(x.device)
        
        topk_, indices_ = torch.topk(dist, self.k, largest=False)
        A_ = torch.zeros(dist.shape).to(x.device)
        A_ = A_.scatter(1, indices_, topk_)
        edge_index_ = torch.nonzero(torch.triu(A_)).T
        # rearrange the node pair indices to be from source to target
        edge_index_ = torch.stack([edge_index_[1], edge_index_[0]])
        # Local Graph Attention
        # TODO: Experiment 1: Only global attention
        # TODO: Experiment 2: Only local attention
        
        h_ = self.gat3(embeddings, edge_index_)
        h_ = F.elu(h_)
       
        h_ = self.gat4(h_, edge_index_)
        h_ = F.elu(h_)

        h = torch.cat([h, h_], dim=1)
        h = F.relu(self.fc3(h))
        h = h + h_
        return h


class PositionalEncoding2d(nn.Module):
    """
    Implements the sinusoidal positional encoding for
    non-recurrent neural networks.
    Implementation based on "Attention Is All You Need"
    :cite:`DBLP:journals/corr/VaswaniSPUJGKP17`
    """
    
    def __init__(self, n=10000, stride=5):
        
        super(PositionalEncoding2d, self).__init__()
        
        self.n = n
        self.stride = stride
        

    def forward(self, x, d=256):
        """
        Args:
            x: [batch_size, n_chan, height, width]
        """
        batch_size, _, h, w = x.shape
        inv_freq = (1.0 / (self.n ** (torch.arange(0, d // 2, 2).float() / (d // 2)))).to(x.device)
        pos_x = torch.arange(0, w-(w%self.stride), self.stride, dtype=torch.float).to(x.device)
        pos_y = torch.arange(0, h-(h%self.stride), self.stride, dtype=torch.float).to(x.device)
        sin_inp_x = torch.einsum("i,j->ij", pos_x, inv_freq)
        sin_inp_y = torch.einsum("i,j->ij", pos_y, inv_freq)
        
        emb_x = torch.cat((sin_inp_x.sin(), sin_inp_x.cos()), dim=-1).unsqueeze(0)
        emb_y = torch.cat((sin_inp_y.sin(), sin_inp_y.cos()), dim=-1).unsqueeze(1)
        
        emb = torch.zeros((h // self.stride, w // self.stride, (d // 2) * 2), device=x.device).type(
            x.type()
        )
        emb[:, :, : (d // 2)] = emb_y
        emb[:, :, (d // 2) : 2 * (d // 2)] = emb_x
        
        return emb[None, :, :, :d].repeat(batch_size, 1, 1, 1)
           