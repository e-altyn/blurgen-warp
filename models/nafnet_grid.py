# ------------------------------------------------------------------------
# Copyright (c) 2022 megvii-model. All Rights Reserved.
# ------------------------------------------------------------------------

"""
Simple Baselines for Image Restoration

@article{chen2022simple,
  title={Simple Baselines for Image Restoration},
  author={Chen, Liangyu and Chu, Xiaojie and Zhang, Xiangyu and Sun, Jian},
  journal={arXiv preprint arXiv:2204.04676},
  year={2022}
}
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from models.arch_util import LayerNorm2d
from models.local_arch import Local_Base

from utils.synthesis import canonical_2d, make_c2w, complex_to_polar, polar_to_complex


class SimpleGate(nn.Module):
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2


class MLP(nn.Module):
    def __init__(self, c, num_poses):
        super().__init__()
        self.seq = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(c, c),
            nn.ReLU(),
            nn.Linear(c, num_poses * 6),
        )

    def forward(self, x):
        return self.seq(x)


class NAFBlock(nn.Module):
    def __init__(self, c, DW_Expand=2, FFN_Expand=2, drop_out_rate=0.0):
        super().__init__()
        dw_channel = c * DW_Expand
        self.conv1 = nn.Conv2d(
            in_channels=c,
            out_channels=dw_channel,
            kernel_size=1,
            padding=0,
            stride=1,
            groups=1,
            bias=True,
        )
        self.conv2 = nn.Conv2d(
            in_channels=dw_channel,
            out_channels=dw_channel,
            kernel_size=3,
            padding=1,
            stride=1,
            groups=dw_channel,
            bias=True,
        )
        self.conv3 = nn.Conv2d(
            in_channels=dw_channel // 2,
            out_channels=c,
            kernel_size=1,
            padding=0,
            stride=1,
            groups=1,
            bias=True,
        )

        # Simplified Channel Attention
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(
                in_channels=dw_channel // 2,
                out_channels=dw_channel // 2,
                kernel_size=1,
                padding=0,
                stride=1,
                groups=1,
                bias=True,
            ),
        )

        # SimpleGate
        self.sg = SimpleGate()

        ffn_channel = FFN_Expand * c
        self.conv4 = nn.Conv2d(
            in_channels=c,
            out_channels=ffn_channel,
            kernel_size=1,
            padding=0,
            stride=1,
            groups=1,
            bias=True,
        )
        self.conv5 = nn.Conv2d(
            in_channels=ffn_channel // 2,
            out_channels=c,
            kernel_size=1,
            padding=0,
            stride=1,
            groups=1,
            bias=True,
        )

        self.norm1 = LayerNorm2d(c)
        self.norm2 = LayerNorm2d(c)

        self.dropout1 = (
            nn.Dropout(drop_out_rate) if drop_out_rate > 0.0 else nn.Identity()
        )
        self.dropout2 = (
            nn.Dropout(drop_out_rate) if drop_out_rate > 0.0 else nn.Identity()
        )

        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)

    def forward(self, inp):
        x = inp

        x = self.norm1(x)

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.sg(x)
        x = x * self.sca(x)
        x = self.conv3(x)

        x = self.dropout1(x)

        y = inp + x * self.beta

        x = self.conv4(self.norm2(y))
        x = self.sg(x)
        x = self.conv5(x)

        x = self.dropout2(x)

        return y + x * self.gamma


class NAFNetGrid(nn.Module):
    def __init__(
        self,
        img_channel=3,
        width=8,
        middle_blk_num=1,
        enc_blk_nums=[],
        dec_blk_nums=[],
        num_poses=16,
    ):
        super().__init__()

        self.intro = nn.Conv2d(img_channel, width, 3, padding=1)
        self.ending = nn.Conv2d(width, 2 * num_poses, 3, padding=1)

        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.middle_blks = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()

        chan = width
        for num in enc_blk_nums:
            self.encoders.append(nn.Sequential(*[NAFBlock(chan) for _ in range(num)]))
            self.downs.append(nn.Conv2d(chan, 2 * chan, 2, 2))
            chan = chan * 2

        self.middle_blks = nn.Sequential(
            *[NAFBlock(chan) for _ in range(middle_blk_num)]
        )

        for num in dec_blk_nums:
            self.ups.append(
                nn.Sequential(
                    nn.Conv2d(chan, chan * 2, 1, bias=False), nn.PixelShuffle(2)
                )
            )
            chan = chan // 2
            self.decoders.append(nn.Sequential(*[NAFBlock(chan) for _ in range(num)]))

        self.padder_size = 2 ** len(self.encoders)

        self.mlp = MLP(width * self.padder_size, num_poses)

        self.num_poses = num_poses

    def forward(self, inp):
        B, C, H, W = inp.shape
        inp = self.check_image_size(inp) # pad H,W for proper pooling

        x = self.intro(inp)

        encs = []

        for encoder, down in zip(self.encoders, self.downs):
            x = encoder(x)
            encs.append(x)
            x = down(x)

        x = self.middle_blks(x)
        x_mlp = self.mlp(x)

        for decoder, up, enc_skip in zip(self.decoders, self.ups, encs[::-1]):
            x = up(x)
            x = x + enc_skip
            x = decoder(x)

        x = self.ending(x)
    
        _, _, H_out, W_out = x.shape

        r, t = x_mlp.chunk(2, dim=1)
        
        dense_pred = (
            x.reshape(B, 2, self.num_poses, H_out, W_out)
            .permute(0, 2, 3, 4, 1)[:, :, :H, :W, :]
        ) # reshape + remove padding
        
        return self.combine_features(r, t, dense_pred)
        
    def combine_features(self, r, t, dense_pred):
        B, _, H, W, _ = dense_pred.shape
        
        # [B, self.num_poses * 3] -> [B, self.num_poses, 3] -> [B * self.num_poses, 3]
        r = r.reshape(B, self.num_poses, 3).reshape(B * self.num_poses, 3)
        t = t.reshape(B, self.num_poses, 3).reshape(B * self.num_poses, 3)
        
        rigid_2d = make_c2w(r, t)  # [B * self.num_poses, 3, 4]
        
        # normalized rigid disps
        grid_2d_rigid = F.affine_grid(
            rigid_2d[:, :2, :3], 
            [B * self.num_poses, 3, H, W]
        )  # [B * self.num_poses, H, W, 2]
        grid_2d_cano = canonical_2d(H, W, device=r.device, normalized=True)  # [1, H, W, 2]
        grid_2d_cano_expanded = grid_2d_cano.expand(B * self.num_poses, -1, -1, -1)
        rigid_disps = grid_2d_rigid - grid_2d_cano_expanded  # [B * self.num_poses, H, W, 2]
        
        # amp_local, phase_local = complex_to_polar(dense_pred)
        amp_local = dense_pred[..., 0] + 1.0 # center around 1.0
        phase_local = dense_pred[..., 1]
        amp_global, phase_global = complex_to_polar(rigid_disps)
        
        amp_local = amp_local.reshape(B, self.num_poses, H, W)
        phase_local = phase_local.reshape(B, self.num_poses, H, W)
        amp_global = amp_global.reshape(B, self.num_poses, H, W)
        phase_global = phase_global.reshape(B, self.num_poses, H, W)
        
        # modulate low-dof predictions with dense ones
        final_disps = polar_to_complex(
            amp_global * amp_local,
            phase_global + phase_local,
        )  # [B, self.num_poses, H, W, 2]
        
        return final_disps

    def check_image_size(self, x):
        _, _, h, w = x.size()
        mod_pad_h = (self.padder_size - h % self.padder_size) % self.padder_size
        mod_pad_w = (self.padder_size - w % self.padder_size) % self.padder_size
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h))
        return x
