import torch
import torch.nn as nn
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from functools import partial
import math
import pywt
from torch.autograd import Function
import torch.nn.functional as F
from typing import Dict, Tuple, Union
from mmcv.cnn import build_norm_layer
# from mmcv.utils import Registry, build_from_cfg
from einops import rearrange, repeat

# BACKBONES = Registry('backbone')


###
class DWConv(nn.Module):
    def __init__(self, dim=768):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = x.transpose(1, 2).view(B, C, H, W)
        x = self.dwconv(x)
        x = x.flatten(2).transpose(1, 2)
        return x


class DWT_Function(Function):
    @staticmethod
    def forward(ctx, x, w_ll, w_lh, w_hl, w_hh):
        x = x.contiguous()
        ctx.save_for_backward(w_ll, w_lh, w_hl, w_hh)
        ctx.shape = x.shape

        dim = x.shape[1]
        x_ll = torch.nn.functional.conv2d(x, w_ll.expand(dim, -1, -1, -1), stride = 2, groups = dim)
        x_lh = torch.nn.functional.conv2d(x, w_lh.expand(dim, -1, -1, -1), stride = 2, groups = dim)
        x_hl = torch.nn.functional.conv2d(x, w_hl.expand(dim, -1, -1, -1), stride = 2, groups = dim)
        x_hh = torch.nn.functional.conv2d(x, w_hh.expand(dim, -1, -1, -1), stride = 2, groups = dim)
        x = torch.cat([x_ll, x_lh, x_hl, x_hh], dim=1)
        return x

    @staticmethod
    def backward(ctx, dx):
        if ctx.needs_input_grad[0]:
            w_ll, w_lh, w_hl, w_hh = ctx.saved_tensors
            B, C, H, W = ctx.shape
            dx = dx.view(B, 4, -1, H//2, W//2)

            dx = dx.transpose(1,2).reshape(B, -1, H//2, W//2)
            filters = torch.cat([w_ll, w_lh, w_hl, w_hh], dim=0).repeat(C, 1, 1, 1)
            dx = torch.nn.functional.conv_transpose2d(dx, filters, stride=2, groups=C)

        return dx, None, None, None, None

class IDWT_Function(Function):
    @staticmethod
    def forward(ctx, x, filters):
        ctx.save_for_backward(filters)
        ctx.shape = x.shape

        B, _, H, W = x.shape
        x = x.view(B, 4, -1, H, W).transpose(1, 2)
        C = x.shape[1]
        x = x.reshape(B, -1, H, W)
        filters = filters.repeat(C, 1, 1, 1)
        x = torch.nn.functional.conv_transpose2d(x, filters, stride=2, groups=C)
        return x

    @staticmethod
    def backward(ctx, dx):
        if ctx.needs_input_grad[0]:
            filters = ctx.saved_tensors
            filters = filters[0]
            B, C, H, W = ctx.shape
            C = C // 4
            dx = dx.contiguous()

            w_ll, w_lh, w_hl, w_hh = torch.unbind(filters, dim=0)
            x_ll = torch.nn.functional.conv2d(dx, w_ll.unsqueeze(1).expand(C, -1, -1, -1), stride = 2, groups = C)
            x_lh = torch.nn.functional.conv2d(dx, w_lh.unsqueeze(1).expand(C, -1, -1, -1), stride = 2, groups = C)
            x_hl = torch.nn.functional.conv2d(dx, w_hl.unsqueeze(1).expand(C, -1, -1, -1), stride = 2, groups = C)
            x_hh = torch.nn.functional.conv2d(dx, w_hh.unsqueeze(1).expand(C, -1, -1, -1), stride = 2, groups = C)
            dx = torch.cat([x_ll, x_lh, x_hl, x_hh], dim=1)
        return dx, None

class IDWT_2D(nn.Module):
    def __init__(self, wave):
        super(IDWT_2D, self).__init__()
        w = pywt.Wavelet(wave)
        rec_hi = torch.Tensor(w.rec_hi)
        rec_lo = torch.Tensor(w.rec_lo)

        w_ll = rec_lo.unsqueeze(0) * rec_lo.unsqueeze(1)
        w_lh = rec_lo.unsqueeze(0) * rec_hi.unsqueeze(1)
        w_hl = rec_hi.unsqueeze(0) * rec_lo.unsqueeze(1)
        w_hh = rec_hi.unsqueeze(0) * rec_hi.unsqueeze(1)

        w_ll = w_ll.unsqueeze(0).unsqueeze(1)
        w_lh = w_lh.unsqueeze(0).unsqueeze(1)
        w_hl = w_hl.unsqueeze(0).unsqueeze(1)
        w_hh = w_hh.unsqueeze(0).unsqueeze(1)
        filters = torch.cat([w_ll, w_lh, w_hl, w_hh], dim=0)
        self.register_buffer('filters', filters)
        #self.filters = self.filters.to(dtype=torch.float16)

    def forward(self, x):
        return IDWT_Function.apply(x, self.filters)


class DWT_2D(nn.Module):
    def __init__(self, wave):
        super(DWT_2D, self).__init__()
        w = pywt.Wavelet(wave)
        dec_hi = torch.Tensor(w.dec_hi[::-1])
        dec_lo = torch.Tensor(w.dec_lo[::-1])

        w_ll = dec_lo.unsqueeze(0) * dec_lo.unsqueeze(1)
        w_lh = dec_lo.unsqueeze(0) * dec_hi.unsqueeze(1)
        w_hl = dec_hi.unsqueeze(0) * dec_lo.unsqueeze(1)
        w_hh = dec_hi.unsqueeze(0) * dec_hi.unsqueeze(1)

        self.register_buffer('w_ll', w_ll.unsqueeze(0).unsqueeze(0))
        self.register_buffer('w_lh', w_lh.unsqueeze(0).unsqueeze(0))
        self.register_buffer('w_hl', w_hl.unsqueeze(0).unsqueeze(0))
        self.register_buffer('w_hh', w_hh.unsqueeze(0).unsqueeze(0))

        # self.w_ll = self.w_ll.to(dtype=torch.float16)
        # self.w_lh = self.w_lh.to(dtype=torch.float16)
        # self.w_hl = self.w_hl.to(dtype=torch.float16)
        # self.w_hh = self.w_hh.to(dtype=torch.float16)

    def forward(self, x):
        return DWT_Function.apply(x, self.w_ll, self.w_lh, self.w_hl, self.w_hh)

class PVT2FFN(nn.Module):
    def __init__(self, in_features, hidden_features):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.dwconv = DWConv(hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, in_features)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):
        x = self.fc1(x)
        x = self.dwconv(x, H, W)
        x = self.act(x)
        x = self.fc2(x)
        return x

class LeFF(nn.Module):
    def __init__(self, in_features, hidden_features, act_layer=nn.GELU,drop = 0.):
        super().__init__()
        self.project_in = nn.Conv2d(in_features, hidden_features*2, kernel_size=1)

        self.dwconv = nn.Conv2d(hidden_features*2, hidden_features*2, kernel_size=3, stride=1, padding=1, groups=hidden_features*2)

        self.project_out = nn.Conv2d(hidden_features, in_features, kernel_size=1)


    def forward(self, x):
        bs, hw, c = x.size()
        hh = int(math.sqrt(hw))
        x = rearrange(x, ' b (h w) (c) -> b c h w ', h = hh, w = hh)
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        x = rearrange(x, ' b c h w -> b (h w) c', h = hh, w = hh)
        return x


class WaveFrSA(nn.Module):
    def __init__(self, dim, num_heads, sr_ratio):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5
        self.sr_ratio = sr_ratio

        self.lf_c = dim // 4
        self.hf_c = dim - self.lf_c
        self.dwt = DWT_2D(wave='haar')
        self.idwt = IDWT_2D(wave='haar')
        # 默认缩小4倍
        self.reduce = nn.Sequential(
            nn.Conv2d(dim, dim//4, kernel_size=1, padding=0, stride=1),
            build_norm_layer(dict(type='BN', requires_grad=False), dim//4)[1],
            nn.ReLU(inplace=True),
        )

        self.filter = nn.Sequential(
            nn.Conv2d(3 * dim // 4, 3 * dim // 4, kernel_size=3, padding=1, stride=1, groups=1),
            build_norm_layer(dict(type='BN', requires_grad=False), 3 * dim // 4)[1],
            nn.ReLU(inplace=True),
        )

        self.reduce_hf = nn.Conv2d(3 * dim // 4, dim // 4, 1, 1, 0)
        # sr_ratio:缩小KV分辨率，降低计算量?(疑问处，如果将——sr_ratio设置为1，那么在上采用最后一层出现显存溢出的问题？？？？？？)
        self.kv_embed = nn.Conv2d(self.lf_c, self.lf_c, kernel_size=sr_ratio, stride=sr_ratio) if sr_ratio > 1 else nn.Identity()
        self.q = nn.Linear(self.lf_c, self.lf_c)
        self.hf_q = nn.Linear(self.lf_c, self.lf_c)
        self.kv = nn.Sequential(
            nn.LayerNorm(self.lf_c),
            nn.Linear(self.lf_c, self.lf_c * 2)
        )
        self.up = nn.Upsample(scale_factor=2)
        self.proj = nn.Linear(3 * dim//4, dim)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):

        B, N, C = x.shape
        x = x.view(B, H, W, C).permute(0, 3, 1, 2)
        x = self.reduce(x)
        q = x.flatten(2).transpose(1,2)
        q = self.q(q).reshape(B, N, self.num_heads, self.lf_c // self.num_heads).permute(0, 2, 1, 3)
        x_dwt = self.dwt(x)
        lf_feat = x_dwt[:, :self.lf_c, :, :]
        hf_feat = x_dwt[:, self.lf_c:, :, :]
        hf_out = self.filter(hf_feat)
        hf_qurey = self.reduce_hf(hf_out).flatten(2).transpose(1, 2)
        hf_q = self.hf_q(hf_qurey).reshape(B, N // 4, self.num_heads, self.lf_c // self.num_heads).permute(0, 2, 1, 3)
        idwt_out = torch.cat([lf_feat, hf_out], dim=1)
        idwt_out = self.idwt(idwt_out).flatten(2).transpose(1, 2)
        kv = self.kv_embed(lf_feat).reshape(B, self.lf_c, -1).permute(0, 2, 1)
        kv = self.kv(kv).reshape(B, -1, 2, self.num_heads, self.lf_c // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(B, N, self.lf_c)
        attn_f = (hf_q @ k.transpose(-2, -1)) * self.scale
        attn_f = attn_f.softmax(dim=-1)
        x_f = (attn_f @ v).transpose(1, 2).reshape(B, N // 4, self.lf_c)
        x_f = x_f.view(B, H // 2, W // 2, C // 4).permute(0, 3, 1, 2)
        x_f = self.up(x_f).transpose(1, 2).reshape(B, N, self.lf_c)
        x = self.proj(torch.cat([x, x_f, idwt_out], dim=-1))


        return x


class WaveFrSViT_Block(nn.Module):
    def __init__(self,
        dim,
        num_heads,
        mlp_ratio,
        drop_path=0.,
        norm_layer=nn.LayerNorm,
        sr_ratio=4,
    ):
        super().__init__()
        #self.norm = nn.LayerNorm(dim)
        self.norm1 = norm_layer(dim)
        self.norm2 = norm_layer(dim)
        self.attn = WaveFrSA(dim, num_heads, sr_ratio)
        # self.mlp = LeFF(in_features=dim, hidden_features=int(dim * mlp_ratio))
        self.mlp = PVT2FFN(in_features=dim, hidden_features=int(dim * mlp_ratio))
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):

        _, _, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = x + self.drop_path(self.attn(self.norm1(x), H, W))
        x = x + self.drop_path(self.mlp(self.norm2(x), H, W))

        x = rearrange(x, 'b (h w) c -> b c h w', h=H, w=W)

        return x

# img = torch.rand([1, 128, 8, 8])
# model = superwavevit_xw_Block(dim=128, num_heads=2, mlp_ratio=8)
# out = model(img)
# print("out:", out.shape)