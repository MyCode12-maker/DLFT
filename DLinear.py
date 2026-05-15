import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Autoformer_EncDec import series_decomp
import torch_dct as dct
from layers.Autoformer_EncDec import series_decomp
from layers.Embed import DataEmbedding_wo_pos
from layers.DWT_Decomposition import Decomposition
from layers.StandardNorm import Normalize
class BasicConv(nn.Module):
    def __init__(self, c_in, c_out, kernel_size, degree=0, stride=1, padding=0, dilation=1, groups=1, act=False,
                 bn=False, bias=False, dropout=0.):
        super(BasicConv, self).__init__()
        self.out_channels = c_out
        self.conv = nn.Conv1d(c_in, c_out, kernel_size=kernel_size, stride=stride, padding=kernel_size // 2,
                              dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm1d(c_out) if bn else None
        self.act = nn.GELU() if act else None
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        if self.bn is not None:
            x = self.bn(x)
        x = self.conv(x.transpose(-1, -2)).transpose(-1, -2)
        if self.act is not None:
            x = self.act(x)
        if self.dropout is not None:
            x = self.dropout(x)
        return x
class BasicConv2d(nn.Module):
    def __init__(self, c_in, c_out, kernel_size, stride=1, padding=None, dilation=1, groups=1, act=False, bn=False,
                 bias=False, dropout=0.):
        super(BasicConv2d, self).__init__()
        if padding is None:
            # 自动居中 padding（保持输出尺寸）
            if isinstance(kernel_size, tuple):
                padding = tuple(k // 2 for k in kernel_size)
            else:
                padding = kernel_size // 2

        self.conv = nn.Conv2d(
            in_channels=c_in,
            out_channels=c_out,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias
        )

        self.bn = nn.BatchNorm2d(c_out) if bn else None
        self.act = nn.GELU() if act else None
        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else None

    def forward(self, x):  # x: [B, C, H, W]
        if self.bn is not None:
            x = self.bn(x)
        x = self.conv(x)
        if self.act is not None:
            x = self.act(x)
        if self.dropout is not None:
            x = self.dropout(x)
        return x

import torch
import torch.nn as nn
import torch.nn.functional as F

import torch
import torch.nn as nn
import torch.nn.functional as F

import torch
import torch.nn as nn
import torch_dct as dct


import torch
import torch.nn as nn
import torch_dct as dct
class TokenMixer(nn.Module):
    def __init__(self, input_seq=[], batch_size=[], channel=[], pred_seq=[], dropout=[], factor=[], d_model=[]):
        super(TokenMixer, self).__init__()
        self.input_seq = input_seq
        self.batch_size = batch_size
        self.channel = channel
        self.pred_seq = pred_seq
        self.dropout = dropout
        self.factor = factor
        self.d_model = d_model

        self.dropoutLayer = nn.Dropout(self.dropout)
        self.layers = nn.Sequential(nn.Linear(self.input_seq, self.pred_seq * self.factor),
                                    nn.GELU(),
                                    nn.Dropout(self.dropout),
                                    nn.Linear(self.pred_seq * self.factor, self.pred_seq)
                                    )

    def forward(self, x):
        #x = x.transpose(1, 2)
        x = self.layers(x)
        #x = x.transpose(1, 2)
        return x
class Model(nn.Module):
    """
    Paper link: https://arxiv.org/pdf/2205.13504.pdf
    """

    def __init__(self, configs, individual=False):
        """
        individual: Bool, whether shared model among different variates.
        """
        super(Model, self).__init__()
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        if self.task_name == 'classification' or self.task_name == 'anomaly_detection' or self.task_name == 'imputation':
            self.pred_len = configs.seq_len
        else:
            self.pred_len = configs.pred_len

        if self.task_name == 'classification':
            self.projection = nn.Linear(
                configs.enc_in * configs.seq_len, configs.num_class)

        self.decompsition = series_decomp(configs.moving_avg)
        self.individual = individual
        self.channels = configs.enc_in

        self.W3 = nn.Parameter(torch.randn(self.seq_len, self.seq_len)* 0.02)

        self.d_models = 8
        self.dfactor = 2
        self.tfactor = 2
        self.embedding_dropout = 0.1
        self.embedding_dropout1 = 0.1

        self.patch_stride = 8
        self.patch_len = 16
        self.patch_num = int((self.seq_len - self.patch_len) / self.patch_stride + 2)
        self.patch_norm = nn.BatchNorm2d(self.channels)

        self.enc_embedding = DataEmbedding_wo_pos(1, self.d_models, configs.embed, configs.freq,
                                                  configs.dropout)

        self.embeddingMixer = nn.Sequential(nn.Linear(self.d_models, self.d_models * self.dfactor),
                                            nn.GELU(),
                                            nn.Dropout(self.embedding_dropout1),
                                            nn.Linear(self.d_models * self.dfactor, 1))

        self.embeddingMixer_P = nn.Sequential(nn.Linear(self.patch_len, self.patch_len * 4),
                                             nn.GELU(),
                                             nn.Dropout(0.1),
                                             nn.Flatten(start_dim=-2, end_dim=-1),
                                             nn.Linear(self.patch_len * 4 * self.patch_num, self.pred_len))

        self.head = nn.Sequential(nn.Flatten(start_dim=-2, end_dim=-1),
                                  nn.Linear(self.patch_num * self.patch_len, self.pred_len))


        self.alpha = nn.Parameter(torch.tensor(0.5))

        self.conv2d = BasicConv2d(self.channels, self.channels, kernel_size=3, groups=self.channels)
        self.conv1 = BasicConv(self.patch_num, self.patch_num, kernel_size=3, groups=self.patch_num)
        self.conv3 = BasicConv(self.patch_len, self.patch_len, kernel_size=3, groups=self.patch_len)

        self.fc = nn.Sequential(
            nn.Linear(self.patch_len, self.patch_len),
            nn.GELU(),
            nn.Linear(self.patch_len, self.patch_len),
            nn.Sigmoid()
        )
    def do_patching(self, x):
        x_end = x[:, :, -1:]
        x_padding = x_end.repeat(1, 1, self.patch_stride)
        x_new = torch.cat((x, x_padding), dim=-1)
        x_patch = x_new.unfold(dimension=-1, size=self.patch_len, step=self.patch_stride)
        return x_patch

    def TDT_p(self, x):

        B, T, N = x.shape

        x = x.permute(0, 2, 1).contiguous().reshape(B * N, T, 1)
        x = self.enc_embedding(x, None)

        x = x.permute(0, 2, 1)

        x_p = self.do_patching(x)
        x = self.embeddingMixer_P(x_p)
        x = x.permute(0, 2, 1)
        x = self.embeddingMixer(x)
        x = x.permute(0, 2, 1)

        x = x.reshape(B, self.channels, self.pred_len).contiguous()
        y = x.permute(0, 2, 1)

        return y
    def DCT(self, x):

        x = x.permute(0, 2, 1)
        B, N, T = x.shape
        x_d = dct.dct(x, norm='ortho') @ self.W3
        #hamming_window = torch.hamming_window(self.patch_len, periodic=True, device=x.device).reshape(1, 1, self.patch_len)  # hamming_window  hann_window
        hann_window = torch.hann_window(self.patch_len, periodic=True, device=x.device).reshape(1, 1, self.patch_len)
        x_d_p = self.do_patching(x_d)
        x_d_p = self.fc(x_d_p) * x_d_p * hann_window
        x_d_p_n = self.patch_norm(x_d_p)

        x_d_p_3 = x_d_p_n.reshape(B*N, self.patch_num,-1)
        x_d_p_3 = self.conv3(x_d_p_3)

        x_d_p_c = self.conv1(x_d_p_3.permute(0, 2, 1))
        x_d_p_c = x_d_p_c.permute(0, 2, 1) + x_d_p_3

        x_d_p_4 = x_d_p_c.reshape(B, N,self.patch_num,-1)
        x_d_p_4 = self.conv2d(x_d_p_4)

        x_d_ = self.head(x_d_p_4)
        x_d_ = dct.idct(x_d_)

        x = x_d_
        x = x.permute(0, 2, 1)

        return x

    def encoder(self, x):
        x_enc = x
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(torch.var(x, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc /= stdev

        x_d = self.DCT(x_enc)
        x_t = self.TDT_p(x_enc)
        y = self.alpha * x_t + (1-self.alpha) * x_d

        dec_out = y * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))

        return dec_out

    def forecast(self, x_enc):
        # Encoder
        return self.encoder(x_enc)

    def imputation(self, x_enc):
        # Encoder
        return self.encoder(x_enc)

    def anomaly_detection(self, x_enc):
        # Encoder
        return self.encoder(x_enc)

    def classification(self, x_enc):
        # Encoder
        enc_out = self.encoder(x_enc)
        # Output
        # (batch_size, seq_length * d_model)
        output = enc_out.reshape(enc_out.shape[0], -1)
        # (batch_size, num_classes)
        output = self.projection(output)
        return output

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            dec_out = self.forecast(x_enc)
            return dec_out[:, -self.pred_len:, :]  # [B, L, D]
        if self.task_name == 'imputation':
            dec_out = self.imputation(x_enc)
            return dec_out  # [B, L, D]
        if self.task_name == 'anomaly_detection':
            dec_out = self.anomaly_detection(x_enc)
            return dec_out  # [B, L, D]
        if self.task_name == 'classification':
            dec_out = self.classification(x_enc)
            return dec_out  # [B, N]
        return None

    def do_patching_m(self, x, patch_len=None, patch_stride=None):
        """
        x: [B, N, T]
        return: [B, N, patch_num, patch_len]
        """

        # 如果不传，用默认值（兼容原来代码）
        if patch_len is None:
            patch_len = self.patch_len
        if patch_stride is None:
            patch_stride = self.patch_stride

        # padding（保持最后一个patch完整）
        x_end = x[:, :, -1:]
        pad_len = patch_stride
        x_padding = x_end.repeat(1, 1, pad_len)

        x_new = torch.cat((x, x_padding), dim=-1)

        # unfold切patch
        x_patch = x_new.unfold(dimension=-1, size=patch_len, step=patch_stride)

        return x_patch