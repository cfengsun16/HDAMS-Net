import torch
import torch.nn as nn
import torch.nn.functional as F
import numbers
from einops import rearrange


# ==================== FSA相关类 ====================

# SC-Attention模块实现
class SCAttention(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.conv1 = nn.Conv2d(dim, dim, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(dim, dim, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        avg_out = self.global_avg_pool(x)
        avg_out = self.conv1(avg_out)
        avg_out = self.relu(avg_out)
        avg_out = self.conv2(avg_out)
        avg_out = self.sigmoid(avg_out)
        return avg_out
        # return x * avg_out


# LKA模块实现
class LKA(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv0 = nn.Conv2d(dim, dim, kernel_size=5, padding=2, groups=dim)
        self.conv_spatial = nn.Conv2d(dim, dim, kernel_size=7, stride=1, padding=9, groups=dim, dilation=3)
        self.conv1 = nn.Conv2d(dim, dim, kernel_size=1)
        
    def forward(self, x):
        attn = self.conv0(x)
        attn = self.conv_spatial(attn)
        attn = self.conv1(attn)
        return attn
        # return x * attn


def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')


def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)


class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma + 1e-5) * self.weight


class LayerNorm(nn.Module):
    def __init__(self, dim):
        super(LayerNorm, self).__init__()
        self.body = BiasFree_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)


class FeedForward(nn.Module):
    def __init__(self, dim, bias):
        super(FeedForward, self).__init__()

        hidden_features = int(dim * 3)

        self.project_in = nn.Conv2d(dim, hidden_features * 2, kernel_size=1, bias=bias)

        self.dwconv = nn.Conv2d(hidden_features * 2, hidden_features * 2, kernel_size=3, stride=1, padding=1,
                                groups=hidden_features * 2, bias=bias)

        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.relu(x1) * x2
        x = self.project_out(x)
        return x


# Frequency-Domain Pixel Attention
class FDPA(nn.Module):
    def __init__(self, dim) -> None:
        super().__init__()

        self.conv = nn.Conv2d(dim, dim*2, 3, 1, 1)

        self.conv1 = nn.Conv2d(dim, dim, 1, 1, groups=1)
        self.conv2 = nn.Conv2d(dim, dim, 1, 1, groups=1)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x)

        x2_fft = torch.fft.fft2(x2)

        out = x1 * x2_fft

        out = torch.fft.ifft2(out, dim=(-2,-1))
        out = torch.abs(out)

        return out


class FSA_Encoder(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.norm = LayerNorm(dim)
        self.ffn = FeedForward(dim, bias=False)

        # Frequency-Domain Pixel Attention
        self.fdpa = FDPA(dim)

        # Spatial-Domain Channel Attention
        self.sc_attention = SCAttention(dim)
        
        # 3x3 DW Conv (新增)
        self.dw_conv3x3 = nn.Conv2d(dim, dim, kernel_size=3, padding=1, stride=1, groups=dim, bias=True)
        
        # LKA模块
        self.lka = LKA(dim)

    def forward(self, x):
        # 频域处理分支
        freq_out = self.fdpa(x)
        # SC-Attention处理
        sc_out = self.sc_attention(freq_out)
        sc_out = freq_out * sc_out  # 频域和SC-Attention结果相乘
        # 3x3 DW Conv分支
        dw_out = self.dw_conv3x3(x)
        # 频域和DW Conv结果相加
        combined = sc_out + dw_out

        # LKA处理
        lka_out = self.lka(combined)
        # LayerNorm
        x_norm = self.norm(lka_out)
        # FFN处理
        ffn_out = self.ffn(x_norm)
        # 第二个残差连接
        x = combined + ffn_out
        
        return x


class FSA_Decoder(nn.Module):
    def __init__(self, dim):
        super().__init__()
        # 添加上采样层
        # self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        
        self.norm = LayerNorm(dim)
        self.ffn = FeedForward(dim, bias=False)

        # Frequency-Domain Pixel Attention
        self.fdpa = FDPA(dim)

        # Spatial-Domain Channel Attention
        self.sc_attention = SCAttention(dim)
        
        # 3x3 DW Conv (新增)
        self.dw_conv3x3 = nn.Conv2d(dim, dim, kernel_size=3, padding=1, stride=1, groups=dim, bias=True)
        
        # LKA模块
        self.lka = LKA(dim)

    def forward(self, x):
        # 上采样
        # x = self.upsample(x)
        
        # 频域处理分支
        freq_out = self.fdpa(x)
        # SC-Attention处理
        sc_out = self.sc_attention(freq_out)
        sc_out = freq_out * sc_out  # 频域和SC-Attention结果相乘
        # 3x3 DW Conv分支
        dw_out = self.dw_conv3x3(x)
        # 频域和DW Conv结果相加
        combined = sc_out + dw_out

        # LKA处理
        lka_out = self.lka(combined)
        # LayerNorm
        x_norm = self.norm(lka_out)
        # FFN处理
        ffn_out = self.ffn(x_norm)
        # 第二个残差连接
        x = combined + ffn_out
        
        return x


# ==================== CAP相关类 ====================

class CALayer(nn.Module):
    def __init__(self, channel, reduction=1, bias=False):
        super().__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # feature channel downscale and upscale --> channel weight
        reduction = 1
        self.conv_du = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=bias),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=bias),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y


class AxialDW(nn.Module):
    def __init__(self, dim, mixer_kernel, dilation=1):
        super().__init__()
        h, w = mixer_kernel
        self.dw_h = nn.Conv2d(dim, dim, kernel_size=(h, 1), padding='same', groups=dim, dilation=dilation)
        self.dw_w = nn.Conv2d(dim, dim, kernel_size=(1, w), padding='same', groups=dim, dilation=dilation)
        self.CAB = CALayer(dim)

    def forward(self, x):
        x = self.CAB(x + self.dw_h(x) + self.dw_w(x))
        return x


class CAP_Encoder(nn.Module):
    """Encoding then downsampling"""

    def __init__(self, in_c, out_c, mixer_kernel=(7, 7)):
        super().__init__()
        self.dw = AxialDW(in_c, mixer_kernel=(7, 7))
        self.bn = nn.BatchNorm2d(in_c)
        self.pw = nn.Conv2d(in_c, out_c, kernel_size=1)
        self.down = nn.MaxPool2d((2, 2))
        self.act = nn.GELU()

    def forward(self, x):
        skip = self.bn(self.dw(x))
        x = self.act(self.down(self.pw(skip)))
        return x, skip


class CAP_Decoder(nn.Module):
    """Upsampling then decoding"""

    def __init__(self, in_c, out_c, mixer_kernel=(7, 7)):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2)
        self.pw = nn.Conv2d(in_c + out_c, out_c, kernel_size=1)
        self.bn = nn.BatchNorm2d(out_c)
        self.dw = AxialDW(out_c, mixer_kernel=(7, 7))
        self.act = nn.GELU()
        self.pw2 = nn.Conv2d(out_c, out_c, kernel_size=1)

    def forward(self, x, skip):
        x = self.up(x)
        x = torch.cat([x, skip], dim=1)
        x = self.act(self.pw2(self.dw(self.bn(self.pw(x)))))
        return x


class MSC_Block(nn.Module):
    def __init__(self, dim) -> None:
        super().__init__()

        ker = 31
        pad = ker // 2
        self.in_conv = nn.Sequential(
                    nn.Conv2d(dim, dim, kernel_size=1, padding=0, stride=1),
                    nn.GELU()
                    )
        self.dw_13 = nn.Conv2d(dim, dim, kernel_size=(1,ker), padding=(0,pad), stride=1, groups=dim)
        self.dw_31 = nn.Conv2d(dim, dim, kernel_size=(ker,1), padding=(pad,0), stride=1, groups=dim)
        self.dw_33 = nn.Conv2d(dim, dim, kernel_size=ker, padding=pad, stride=1, groups=dim)
        self.dw_11 = nn.Conv2d(dim, dim, kernel_size=1, padding=0, stride=1, groups=dim)
        self.act = nn.ReLU()
        
        self.out_conv = nn.Conv2d(dim, dim, kernel_size=1, padding=0, stride=1)

    def forward(self, x):
        # 效仿model.py的实现方式
        out = self.in_conv(x)
        
        # 直接从原始输入x取值进行相加，而非压缩后的out
        out = x + self.dw_13(out) + self.dw_31(out) + self.dw_33(out) + self.dw_11(out)
        out = self.act(out)
        return self.out_conv(out)


# ==================== 整合后的主网络 ====================

class HDAMS_Net(nn.Module):
    def __init__(self):
        super().__init__()

        """Encoder"""
        self.conv_in = nn.Conv2d(3, 16, kernel_size=7, padding='same')
        # 用FSA替换原来的EncoderBlock
        self.e1 = FSA_Encoder(16)  # 替换 EncoderBlock(16, 64)
        self.e1_down = nn.Sequential(
            nn.Conv2d(16, 64, kernel_size=1),  # 维度变换
            nn.MaxPool2d((2, 2))  # 下采样
        )
        self.e3 = CAP_Encoder(64, 256)

        self.b5 = MSC_Block(256)

        """Decoder"""
        self.d4 = CAP_Decoder(256, 64)
        # 用FSA替换原来的DecoderBlock
        self.d2_up = nn.Upsample(scale_factor=2)  # 上采样
        self.d2_conv = nn.Conv2d(64 + 16, 16, kernel_size=1)  # 跳跃连接融合
        self.d2 = FSA_Decoder(16)  # 替换 DecoderBlock(64, 16)
        
        self.conv_out = nn.Conv2d(16, 3, kernel_size=1)  # 输出三维RGB图片

    def forward(self, x):
        """Encoder"""
        x = self.conv_in(x)
        
        # 第一层编码 - 使用HDA
        skip1 = self.e1(x)  # HDA处理，保持16通道
        x = self.e1_down(skip1)  # 下采样并变换到64通道
        
        x, skip3 = self.e3(x)

        """BottleNeck"""
        x = self.b5(x)  # (256, H/4, W/4)

        """Decoder"""
        x = self.d4(x, skip3)
        
        # 第二层解码 - 使用HDA
        x = self.d2_up(x)  # 上采样
        x = torch.cat([x, skip1], dim=1)  # 跳跃连接
        x = self.d2_conv(x)  # 融合特征
        x = self.d2(x)  # HDA处理
        
        x = self.conv_out(x)
        return x
