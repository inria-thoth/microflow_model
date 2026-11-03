import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.baselines.searaft.layer import BasicBlock, conv1x1, conv3x3


class ResNetFPN(nn.Module):
    """
    ResNet18, output resolution is 1/8.
    Each block has 2 layers.
    """
    def __init__(self, args, input_dim=3, output_dim=256, ratio=1.0, norm_layer=nn.BatchNorm2d, init_weight=False):
        super().__init__()
        # Config
        block = BasicBlock
        # block_dims = args.searaft_block_dims
        block_dims = [64, 128, 256]
        self.block_dims = block_dims
        # print(type(block_dims))
        initial_dim = args.searaft_initial_dim
        self.init_weight = init_weight
        self.input_dim = input_dim
        # Class Variable
        self.in_planes = initial_dim
        for i in range(len(block_dims)):
            block_dims[i] = int(block_dims[i] * ratio)
        # Networks
        self.conv1 = nn.Conv2d(input_dim, initial_dim, kernel_size=7, stride=2, padding=3)
        self.bn1 = norm_layer(initial_dim)
        self.relu = nn.ReLU(inplace=True)
        if args.searaft_pretrain == 'resnet34':
            n_block = [3, 4, 6]
        elif args.searaft_pretrain == 'resnet18':
            n_block = [2, 2, 2]
        else:
            raise NotImplementedError
        self.layer1 = self._make_layer(block, block_dims[0], stride=1, norm_layer=norm_layer, num=n_block[0])  # 1/2
        self.layer2 = self._make_layer(block, block_dims[1], stride=2, norm_layer=norm_layer, num=n_block[1])  # 1/4
        self.layer3 = self._make_layer(block, block_dims[2], stride=2, norm_layer=norm_layer, num=n_block[2])  # 1/8
        self.final_conv = conv1x1(block_dims[2], output_dim)
        self._init_weights(args)
        # raise Exception("j1")

    def _init_weights(self, args):

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d, nn.GroupNorm)):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

        if self.init_weight:
            # print(f"init_weight")
            from torchvision.models import resnet18, ResNet18_Weights, resnet34, ResNet34_Weights
            if args.searaft_pretrain == 'resnet18':
                pretrained_dict = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1).state_dict()
            else:
                pretrained_dict = resnet34(weights=ResNet34_Weights.IMAGENET1K_V1).state_dict()
            # print("")
            # print(f"before pretrained_dict ")
            # for k in pretrained_dict:
            #     print(k)
            model_dict = self.state_dict()
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
            # for k in model_dict:
            #     if k not in pretrained_dict:
            #         print(f"key {k} removed")
            # pretrained_dict = pretrained_dict2
            # print("")
            # print(f"after pretrained_dict")
            # for k in pretrained_dict:
            #     print(k)
            if self.input_dim == 6:
                for k, v in pretrained_dict.items():
                    if k == 'conv1.weight':
                        pretrained_dict[k] = torch.cat((v, v), dim=1)

            # print("")
            # print(f"after v2 pretrained_dict ")
            # for k in pretrained_dict:
            #     print(k)
            model_dict.update(pretrained_dict)
            self.load_state_dict(model_dict, strict=False)

    def _make_layer(self, block, dim, stride=1, norm_layer=nn.BatchNorm2d, num=2):
        layers = []
        layers.append(block(self.in_planes, dim, stride=stride, norm_layer=norm_layer))
        for i in range(num - 1):
            layers.append(block(dim, dim, stride=1, norm_layer=norm_layer))
        self.in_planes = dim
        return nn.Sequential(*layers)

    def forward(self, x):
        # ResNet Backbone
        x = self.relu(self.bn1(self.conv1(x)))
        for i in range(len(self.layer1)):
            x = self.layer1[i](x)
        for i in range(len(self.layer2)):
            x = self.layer2[i](x)
        for i in range(len(self.layer3)):
            x = self.layer3[i](x)
        # Output
        output = self.final_conv(x)
        return output

    def forward_with_midlevel(self, x):
        x = self.relu(self.bn1(self.conv1(x)))  # after conv1 32, 64, 128, 128
        for i in range(len(self.layer1) - 1):
            x = self.layer1[i](x)
        out_conv_2 = self.layer1[len(self.layer1) - 1](x)  # after layer 1, resolution 1/2 :  32, 64, 128, 128
        x = self.layer2[0](out_conv_2)
        for i in range(1, len(self.layer2) - 1):
            x = self.layer2[i](x)
        out_conv_4 = self.layer2[len(self.layer2) - 1](x)  # after layer 2, resolution 1/4 :  32, 128, 64, 64
        x = self.layer3[0](out_conv_4)
        for i in range(1, len(self.layer3) - 1):
            x = self.layer3[i](x)
        out_conv_8 = self.layer3[len(self.layer3) - 1](x)  # after layer 3, resolution 1/8 :  32, 256, 32, 32
        output = self.final_conv(out_conv_8)
        return output, out_conv_8, out_conv_4, out_conv_2
