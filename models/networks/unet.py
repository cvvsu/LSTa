from distutils.command.build import build
import torch
from torch import nn
import functools


def center_crop(out, crop_h, crop_w):
    """
    crop the out tensor to the desired height and width

    Parameters:
        out (tensor) -- [N, C, H, W]
        crop_h, crop_w (int) -- desired height and width

    Returns
        tensor [N, C, crop_h, crop_w]
    """
    _, _, H, W = out.shape

    x = (H - crop_h) // 2
    y = (W - crop_w) // 2
    return out[:, :, x:x+crop_h, y:y+crop_w]


def cat_tensor(x1, x2):
    """
    cat tensors with different shapes.

    Parameters:
        x1 (tensor) -- [N, C, H1, W1]
        x2 (tensor) -- [N, C, H2, W2]
    
    H1 >= H2
    W1 >= W2
    
    Returns:
        catted tensor
    """
    if x1.shape == x2.shape:
        return torch.cat([x1, x2], dim=1)
    else:
        return torch.cat([x1, center_crop(x2, x1.shape[2], x1.shape[3])], dim=1)


def get_norm_layer(norm_type='instance'):
    """Return a normalization layer
    Parameters:
        norm_type (str) -- the name of the normalization layer: batch | instance | none
    For BatchNorm, we use learnable affine parameters and track running statistics (mean/stddev).
    For InstanceNorm, we do not use learnable affine parameters. We do not track running statistics.

    Code from https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/models/networks.py
    """
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True, track_running_stats=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=False)
    elif norm_type == 'none':
        def norm_layer(x): return nn.Identity()
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer


class SEBlock(nn.Module):
    def __init__(self, input_nc, r=16):
        super().__init__()
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.extraction = nn.Sequential(
            nn.Linear(input_nc, input_nc // r, bias=False),
            nn.ReLU(),
            nn.Linear(input_nc // r, input_nc, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        N, nc, _, _ = x.shape
        y = self.squeeze(x).view(N, nc)
        y = self.extraction(y).view(N, nc, 1, 1)
        return x * y.expand_as(x)


class ConvBlock(nn.Module):
    """
    convolutional blocks for Unet.
    """
    def __init__(self, input_nc, output_nc, up_or_down, use_act=True, use_bias=True, 
                norm_layer=nn.BatchNorm2d, kernel_size=4, stride=2, padding=1, conv_type='sconv'):
        super().__init__()
        model = []

        if up_or_down == 'down':

            if use_act:
                model += [nn.LeakyReLU(0.2, True)]

            if (conv_type == 'sconv') or (conv_type == 'seconv'):
                model += [
                    nn.Conv2d(input_nc, output_nc, kernel_size=kernel_size, stride=stride, padding=padding, bias=use_bias)
                ]
            elif conv_type == 'dconv':
                model += [
                    nn.Conv2d(input_nc, output_nc, kernel_size=kernel_size, stride=stride, padding=padding, bias=use_bias),
                    nn.Conv2d(output_nc, output_nc, kernel_size=3, stride=1, padding=1, bias=use_bias)
                ]
            else:
                raise NotImplementedError(f'Known layer type. Acceptable types are sconv, dconv, and seconv.')
        
        elif up_or_down == 'up':
            model = [
                nn.ReLU(True),
                nn.ConvTranspose2d(input_nc, output_nc, kernel_size=kernel_size, stride=stride, padding=padding, bias=use_bias, output_padding=1)
            ]

        if norm_layer is not None:
            model += [norm_layer(output_nc)]
            if conv_type == 'seconv':
                model += [SEBlock(output_nc)]
    
        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)


class UnetBlock(nn.Module):
    """
    Unet submodule with skip connections
    X -------------------identity----------------------
    |-- downsampling -- |submodule| -- upsampling --|

    Reference:
        1. https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/models/networks.py
    """
    def __init__(self, outer_nc, inner_nc, input_nc=None, submodule=None, outermost=False, 
                    innermost=False, norm_layer=nn.BatchNorm2d, use_dropout=False, 
                    kernel_size=4, stride=2, padding=1, conv_type='sconv'):
        """
        Parameters:
            outer_nc (int)      -- the number of filters in the outer conv layer
            inner_nc (int)      -- the number of filters in the inner conv layer
            input_nc (int)      -- the number of channels in input images/features
            submodule (module)  -- previously defined submodules
            outermost (bool)    -- if this module is the outermost module
            innermost (bool)    -- if this module is the innermost module
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers

        """
        super().__init__()
        self.outermost = outermost

        # when the batch normalization is applied, the bias operation has no effect on the result
        # but will cost more memory and computational resouses
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        if input_nc is None:
            input_nc = outer_nc
        
        if innermost:
            model = [
                ConvBlock(input_nc, inner_nc, 'down', use_bias=use_bias, norm_layer=None,
                            kernel_size=kernel_size, stride=stride, padding=padding, conv_type=conv_type),
                ConvBlock(inner_nc, outer_nc, 'up', use_bias=use_bias, norm_layer=norm_layer, 
                            kernel_size=kernel_size, stride=stride, padding=padding, conv_type=conv_type)
            ]
        elif outermost:
            model = [
                ConvBlock(input_nc, inner_nc, 'down', use_bias=use_bias, use_act=False, norm_layer=None, 
                            kernel_size=kernel_size, stride=stride, padding=padding, conv_type=conv_type),
                submodule,
                ConvBlock(inner_nc*2, outer_nc, 'up', norm_layer=None, use_bias=True, 
                            kernel_size=kernel_size, stride=stride, padding=padding, conv_type=conv_type)
            ]
        else:
            model = [
                ConvBlock(input_nc, inner_nc, 'down', use_bias=use_bias, norm_layer=norm_layer, 
                            kernel_size=kernel_size, stride=stride, padding=padding, conv_type=conv_type),
                submodule,
                ConvBlock(inner_nc*2, outer_nc, 'up', use_bias=use_bias, norm_layer=norm_layer, 
                            kernel_size=kernel_size, stride=stride, padding=padding, conv_type=conv_type)
            ]

            if use_dropout:
                model += [nn.Dropout(0.5)]

        self.model = nn.Sequential(*model)
    
    def forward(self, x):
        out = self.model(x)
        if self.outermost:
            return center_crop(out, x.shape[2], x.shape[3])
        else:
            return cat_tensor(x, out)


class Unet(nn.Module):
    """
    Create a Unet model.
    Reference:
        1. https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/models/networks.py
    """
    def __init__(self, input_nc, output_nc, num_downs, num_filters=64, 
                        norm_layer=nn.BatchNorm2d, use_dropout=False, 
                        kernel_size=4, stride=2, padding=1, conv_type='sconv'):
        """
        Parameters:
            input_nc (int)      -- the number of channels in input images
            output_nc (int)     -- the number of channels in output images (number of classes, etc.)
            num_downs (int)     -- number of downsamplings. Image size to [h/2^num_downs, w/2^num_downs]
            num_filters (int)   -- the number of filters 
            norm_layer (module) -- normalization layer
        
        Returns:
            An Unet model.
        
        The recursive process to construct the Unet module.
        Learned from the CycleGan code (link above).
        """
        super().__init__()
        # inner most layer
        unet_block = UnetBlock(num_filters*8, num_filters*8, input_nc=None, submodule=None, norm_layer=norm_layer, 
                                innermost=True, kernel_size=kernel_size, stride=stride, padding=padding, conv_type=conv_type)

        # add the intermediate layers
        for i in range(num_downs - 5):
            unet_block = UnetBlock(num_filters*8, num_filters*8, input_nc=None, submodule=unet_block, 
                                    norm_layer=norm_layer, use_dropout=use_dropout, kernel_size=kernel_size, 
                                    stride=stride, padding=padding, conv_type=conv_type)
        
        # gradually reduce the number of filters from num_filters*8 to num_filters
        unet_block = UnetBlock(num_filters*4, num_filters*8, input_nc=None, submodule=unet_block, 
                                norm_layer=norm_layer, kernel_size=kernel_size, stride=stride, 
                                padding=padding, conv_type=conv_type)
        unet_block = UnetBlock(num_filters*2, num_filters*4, input_nc=None, submodule=unet_block, 
                                norm_layer=norm_layer, kernel_size=kernel_size, stride=stride, 
                                padding=padding, conv_type=conv_type)
        unet_block = UnetBlock(num_filters, num_filters*2, input_nc=None, submodule=unet_block, 
                                norm_layer=norm_layer, kernel_size=kernel_size, stride=stride, 
                                padding=padding, conv_type=conv_type)
        self.model = UnetBlock(output_nc, num_filters, input_nc=input_nc, submodule=unet_block, 
                                outermost=True, norm_layer=norm_layer, kernel_size=kernel_size, stride=stride, 
                                padding=padding, conv_type=conv_type)

    def forward(self, x):
        return self.model(x)


def build_network(args):
    return Unet(
        args.input_nc, 
        args.output_nc, 
        num_downs=args.num_layers, 
        num_filters=args.num_filters, 
        norm_layer=get_norm_layer('instance'),
        conv_type=args.conv_type
    )


if __name__=='__main__':
    import argparse

    parser = argparse.ArgumentParser('Test')
    parser.add_argument('--input_nc', type=int, default=3)
    parser.add_argument('--output_nc', type=int, default=1)
    parser.add_argument('--num_layers', type=int, default=8)
    parser.add_argument('--conv_type', type=str, default='sconv')
    
    args = parser.parse_args()

    net = build_network(args)

    # print(net)

    x = torch.randn(2, 3, 313, 517)
    out = net(x)
    print(out.shape)