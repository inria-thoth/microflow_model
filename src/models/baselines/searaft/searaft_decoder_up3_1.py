import numpy as np
import torch
import math
import torch.nn as nn
import torch.nn.functional as F

from src.models.baselines.searaft.update import BasicUpdateBlock
from src.models.baselines.searaft.corr import CorrBlock
from src.models.baselines.searaft.utils.utils import coords_grid, InputPadder
from src.models.baselines.searaft.extractor import ResNetFPN
from src.models.baselines.searaft.layer import conv1x1, conv3x3
from src.models.model_utils.saving_utils import apply_convention_and_save
from src.models.model_utils.warping_utils import warp_image_torch, get_inverse_flow
from src.models.model_utils.model_layers import conv, predict_flow, deconv, crop_like


# from huggingface_hub import PyTorchModelHubMixin

class SeaRAFTDecoderUp31(
    nn.Module,
    # # PyTorchModelHubMixin,
    # # optionally, you can add metadata which gets pushed to the model card
    # repo_url="https://github.com/princeton-vl/SEA-RAFT",
    # pipeline_tag="optical-flow-estimation",
    # license="bsd-3-clause",
):
    def __init__(self, args):
        super().__init__()
        self.args = args
        # self.args.searaft_max_iterations = 1
        self.max_iterations = self.args.searaft_max_iterations + 1
        print(f"sea raft self.max_iterations {self.max_iterations}")
        self.output_dim = args.searaft_dim * 2

        self.args.corr_levels = 4
        self.args.corr_radius = args.searaft_radius
        self.args.corr_channel = args.corr_levels * (args.searaft_radius * 2 + 1) ** 2
        self.cnet = ResNetFPN(args, input_dim=6, output_dim=self.output_dim, norm_layer=nn.BatchNorm2d,
                              init_weight=True)

        # conv for iter 0 results
        self.init_conv = conv3x3(self.output_dim, self.output_dim)
        # self.upsample_weight = nn.Sequential(
        #     # convex combination of 3x3 patches
        #     nn.Conv2d(args.searaft_dim, self.output_dim, 3, padding=1),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(self.output_dim, 64 * 9, 1, padding=0)
        # )
        self.flow_head = nn.Sequential(
            # flow(2) + weight(2) + log_b(2)
            nn.Conv2d(args.searaft_dim, self.output_dim, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.output_dim, 6, 3, padding=1)
        )
        if self.args.searaft_max_iterations > 0:
            self.fnet = ResNetFPN(args, input_dim=3, output_dim=self.output_dim, norm_layer=nn.BatchNorm2d,
                                  init_weight=True)
            self.update_block = BasicUpdateBlock(args, hdim=args.searaft_dim, cdim=args.searaft_dim)

        block_dims = self.cnet.block_dims
        self.mid_level_head_8 = nn.Sequential(
            nn.Conv2d(block_dims[2], self.output_dim, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.output_dim, block_dims[1] * 4, 3, padding=1)
        )

        self.mid_level_head_4 = nn.Sequential(
            nn.Conv2d(block_dims[1], self.output_dim, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.output_dim, block_dims[0] * 4, 3, padding=1)
        )

        self.mid_level_head_2 = nn.Sequential(
            nn.Conv2d(block_dims[0], self.output_dim, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.output_dim, block_dims[0] * 4, 3, padding=1)
        )

        self.pixel_shuffle = torch.nn.PixelShuffle(2)

        input_dim4 = block_dims[1] + block_dims[0] * 4 + 2
        intermediate_dim4 = block_dims[1] + block_dims[0] * 4
        self.flow_head4 = nn.Sequential(
            nn.Conv2d(input_dim4, intermediate_dim4, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(intermediate_dim4, 2 * 4, 3, padding=1)
        )

        input_dim2 = block_dims[0] + block_dims[0] * 4 + 2
        intermediate_dim2 = block_dims[0] + block_dims[0] * 4
        self.flow_head2 = nn.Sequential(
            nn.Conv2d(input_dim2, intermediate_dim2, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(intermediate_dim2, 2 * 4, 3, padding=1)
        )

        input_dim1 = block_dims[0] + 2
        intermediate_dim1 = block_dims[0]
        self.flow_head1 = nn.Sequential(
            nn.Conv2d(input_dim1, intermediate_dim1, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(intermediate_dim1, 2 * 1, 3, padding=1)
        )

        if len(self.args.searaft_pretrained_filename) > 0:
            self.load_from_pretrained_searaft(self.args.searaft_pretrained_filename)

        # # initialize weights
        # self._init_weights()

        # freeze everything but the head and decoder part
        if self.args.searaft_freeze:
            self.freeze_parameters()

    def _init_weights(self):
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

    def load_from_pretrained_searaft(self, filename):
        pretrained_dict = torch.load(filename, map_location='cpu')
        model_dict = self.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        self.load_state_dict(model_dict, strict=False)

    def freeze_parameters(self):
        print(f"before freeze")
        for name, param in self.named_parameters():
            if param.requires_grad:
                print(f"{name}: requires_grad ")

        freeze_parts = self.args.searaft_freeze_starts.split("_")
        for name, param in self.named_parameters():
            for freeze_part in freeze_parts:
                if name.startswith(freeze_part):
                    param.requires_grad = False

        print(f"after freeze")
        for name, param in self.named_parameters():
            if param.requires_grad:
                print(f"{name}: requires_grad ")

    def initialize_flow(self, img):
        """ Flow is represented as difference between two coordinate grids flow = coords2 - coords1"""
        N, C, H, W = img.shape
        coords1 = coords_grid(N, H // 8, W // 8, device=img.device)
        coords2 = coords_grid(N, H // 8, W // 8, device=img.device)
        return coords1, coords2

    def upsample_data(self, flow, info, mask):
        """ Upsample [H/8, W/8, C] -> [H, W, C] using convex combination """
        N, C, H, W = info.shape
        mask = mask.view(N, 1, 9, 8, 8, H, W)
        mask = torch.softmax(mask, dim=2)

        up_flow = F.unfold(8 * flow, [3, 3], padding=1)
        up_flow = up_flow.view(N, 2, 9, 1, 1, H, W)
        up_info = F.unfold(info, [3, 3], padding=1)
        up_info = up_info.view(N, C, 9, 1, 1, H, W)

        up_flow = torch.sum(mask * up_flow, dim=2)
        up_flow = up_flow.permute(0, 1, 4, 2, 5, 3)
        up_info = torch.sum(mask * up_info, dim=2)
        up_info = up_info.permute(0, 1, 4, 2, 5, 3)

        return up_flow.reshape(N, 2, 8 * H, 8 * W), up_info.reshape(N, C, 8 * H, 8 * W)

    def upsample_multiscale(self, flow_and_info8, out_conv_8, out_conv_4, out_conv_2):

        # apply mid-level heads to the mid-level feature maps
        mid_out_conv_8 = self.mid_level_head_8(out_conv_8)  # 32, 512, 32, 32
        mid_out_conv_4 = self.mid_level_head_4(out_conv_4)  # 32, 256, 64, 64
        mid_out_conv_2 = self.mid_level_head_2(out_conv_2)  # 32, 256, 128, 128

        # upsample mid-level feature maps for residual connections
        mid_out_conv_8_up = self.pixel_shuffle(mid_out_conv_8)  # 32, 128, 64, 64
        mid_out_conv_4_up = self.pixel_shuffle(mid_out_conv_4)  # 32, 64, 128, 128
        mid_out_conv_2_up = self.pixel_shuffle(mid_out_conv_2)  # 32, 64, 256, 256

        flow8 = flow_and_info8[:, :2]
        flow_and_info8_up = F.interpolate(flow8, scale_factor=2)  # the first is Nearest Interpolate
        concat4 = torch.cat((mid_out_conv_4, mid_out_conv_8_up, flow_and_info8_up), 1)  # 32, 386, 64, 64
        flow4       = self.flow_head4(concat4)  # 32, 8, 64, 64

        flow_and_info4_up    = self.pixel_shuffle(flow4)  # 32, 2, 128, 128
        concat2 = torch.cat((mid_out_conv_2, mid_out_conv_4_up, flow_and_info4_up), 1)  # 32, 322, 128, 128
        flow2 = self.flow_head2(concat2)  #  32, 8, 128, 128

        flow_and_info2_up = self.pixel_shuffle(flow2)  # 32, 2, 256, 256
        concat1 = torch.cat((mid_out_conv_2_up, flow_and_info2_up), 1)  # 32, 66, 256, 256
        flow1       = self.flow_head1(concat1)  # 32, 2, 256, 256

        flow1_up = F.interpolate(flow1, scale_factor=2)
        flow1_up_down = F.interpolate(flow1_up, scale_factor=0.5)

        return flow1_up_down, None

    # def forward(self, image1, image2, iters=None, flow_gt=None, test_mode=False):
    def forward(
            self, x, ptv=None, args=None, save=False, frame_labels=False, crs_meta_data=None, transform_meta_data=None,
            flow_gt=None, test_mode=False):
        """ Estimate optical flow between pair of frames """
        # print(f"x {x.shape}")
        image1 = x[:, 0]
        image2 = x[:, 1]
        image1 = torch.tile(image1[..., None], (1, 1, 3))
        image2 = torch.tile(image2[..., None], (1, 1, 3))

        image1 = image1.permute(0, 3, 1, 2)
        image2 = image2.permute(0, 3, 1, 2)
        # print(f"image1 {image1.shape}")
        # print(f"image2 {image2.shape}")
        N, _, H, W = image1.shape

        iters = self.args.searaft_max_iterations
        # if flow_gt is None:
        # flow_gt = torch.zeros(N, 2, H, W, device=image1.device)

        # image1 = 2 * (image1 / 255.0) - 1.0
        # image2 = 2 * (image2 / 255.0) - 1.0
        image1 = image1.contiguous()
        image2 = image2.contiguous()
        flow_predictions = []
        info_predictions = []

        # padding
        padder = InputPadder(image1.shape)
        image1, image2 = padder.pad(image1, image2)
        N, _, H, W = image1.shape
        dilation = torch.ones(N, 1, H // 8, W // 8, device=image1.device)

        # run the context network
        cnet_input = torch.cat([image1, image2], dim=1)
        cnet, out_conv_8, out_conv_4, out_conv_2 = self.cnet.forward_with_midlevel(cnet_input)

        cnet = self.init_conv(out_conv_8)
        net, context = torch.split(cnet, [self.args.searaft_dim, self.args.searaft_dim], dim=1)

        # init flow
        flow_update = self.flow_head(net)
        # weight_update = .25 * self.upsample_weight(net)
        flow_8x = flow_update[:, :2]
        info_8x = flow_update[:, 2:]

        flow_up, info_up = self.upsample_multiscale(flow_update, out_conv_8, out_conv_4, out_conv_2)
        # flow_up = F.interpolate(flow_8x, scale_factor=8)
        info_up = None

        flow_predictions.append(flow_up)
        info_predictions.append(info_up)

        if self.args.searaft_max_iterations > 0:
            # run the feature network
            fmap1_8x = self.fnet(image1)
            fmap2_8x = self.fnet(image2)
            corr_fn = CorrBlock(fmap1_8x, fmap2_8x, self.args)

        for itr in range(iters):
            N, _, H, W = flow_8x.shape
            # if self.args.detach_gradients:
            flow_8x = flow_8x.detach()
            coords2 = (coords_grid(N, H, W, device=image1.device) + flow_8x).detach()
            corr = corr_fn(coords2, dilation=dilation)
            net = self.update_block(net, context, corr, flow_8x)

            flow_update = self.flow_head(net)
            flow_8x = flow_8x + flow_update[:, :2]
            info_8x = flow_update[:, 2:]   # why isn't it updated?
            # upsample predictions
            # print(f"flow_8x {flow_8x.shape}")
            # print(f"info_8x {info_8x.shape}")
            flow_and_info_8x = torch.cat([flow_8x, info_8x], dim=1)
            # print(f"flow_and_info_8x {flow_and_info_8x.shape}")
            flow_up, info_up = self.upsample_multiscale(flow_and_info_8x, out_conv_8, out_conv_4, out_conv_2)
            # flow_up = F.interpolate(flow_8x, scale_factor=8)
            info_up = None
            flow_predictions.append(flow_up)
            info_predictions.append(info_up)

        for i in range(len(info_predictions)):
            flow_predictions[i] = padder.unpad(flow_predictions[i])
            # info_predictions[i] = padder.unpad(info_predictions[i])

        # info_predictions are 4, why not 3?

        # exclude invalid pixels and extremely large diplacements
        nf_predictions = []
        # if flow_gt is not None:
        #     for i in range(len(info_predictions)):
        #         # print(f"info_predictions[i] {info_predictions[i].shape}")
        #         if not self.args.searaft_use_var:
        #             var_max = var_min = 0
        #         else:
        #             var_max = self.args.searaft_var_max
        #             var_min = self.args.searaft_var_min
        #
        #         raw_b = info_predictions[i][:, 2:]
        #         log_b = torch.zeros_like(raw_b)
        #         weight = info_predictions[i][:, :2]
        #         # print(f"raw_b {raw_b.shape}")
        #         # print(f"weight {weight.shape}")
        #         # Large b Component
        #         log_b[:, 0] = torch.clamp(raw_b[:, 0], min=0, max=var_max)
        #         # Small b Component
        #         log_b[:, 1] = torch.clamp(raw_b[:, 1], min=var_min, max=0)
        #         # term2: [N, 2, m, H, W]
        #         # print(f"flow_gt {flow_gt.shape}")
        #         # print(f"flow_predictions {flow_predictions.shape}")
        #         # print(f"log_b {log_b.shape}")
        #         term2 = ((flow_gt - flow_predictions[i]).abs().unsqueeze(2)) * (torch.exp(-log_b).unsqueeze(1))
        #         # term1: [N, m, H, W]
        #         term1 = weight - math.log(2) - log_b
        #         nf_loss = torch.logsumexp(weight, dim=1, keepdim=True) - torch.logsumexp(term1.unsqueeze(1) - term2,
        #                                                                                  dim=2)
        #         nf_predictions.append(nf_loss)

        if save:
            for i in range(len(flow_predictions)):
                apply_convention_and_save(
                    frame_labels,
                    flow_predictions[i],
                    args.save_dir,
                    f"searaft_decoder_convt_{i}",
                    crs_meta_data,
                    transform_meta_data
                )

                reverse_prediction = get_inverse_flow(flow_predictions[i])
                apply_convention_and_save(
                    frame_labels,
                    reverse_prediction,
                    args.save_dir,
                    f"reverse_searaft_decoder_convt_{i}",
                    crs_meta_data,
                    transform_meta_data
                )

            reverse_prediction = get_inverse_flow(flow_predictions[-1])
            apply_convention_and_save(
                frame_labels,
                reverse_prediction,
                args.save_dir,
                f"reverse_searaft_decoder_convt_final",
                crs_meta_data,
                transform_meta_data
            )

        if test_mode:
            for iteration in range(len(flow_predictions)):
                flow_predictions[iteration] = get_inverse_flow(flow_predictions[iteration])

        if args.loss.lower() == "searaft":
            return flow_predictions, nf_predictions
        return flow_predictions, None

        #     return {'final': flow_predictions[-1], 'flow': flow_predictions, 'info': info_predictions, 'nf': nf_predictions}
        # else:
        #     return {'final': flow_predictions[-1], 'flow': flow_predictions, 'info': info_predictions, 'nf': None}
