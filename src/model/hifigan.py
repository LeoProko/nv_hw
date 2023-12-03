import torch
from torch import nn
import torch.nn.functional as F


class Transpose(nn.Module):
    def __init__(self, dim_1: int, dim_2: int):
        super().__init__()

        self.dim_1 = dim_1
        self.dim_2 = dim_2

    def forward(self, x: torch.Tensor):
        return x.transpose(self.dim_1, self.dim_2)


class MyConv1d(nn.Conv1d):
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return nn.Conv1d.forward(self, input.transpose(-1, -2)).transpose(-1, -2)


class ResBlock(nn.Module):
    def __init__(
        self,
        n_channels: int,
        kernel_size: int,
        dilations: list[int],
        relu_leakage: float,
    ):
        super().__init__()

        self.activation = nn.LeakyReLU(relu_leakage)
        self.convs = nn.ModuleList(
            [
                nn.Sequential(
                    *[
                        nn.LeakyReLU(relu_leakage),
                        MyConv1d(
                            n_channels,
                            n_channels,
                            kernel_size=kernel_size,
                            stride=1,
                            dilation=dilation,
                            padding=dilation * (kernel_size - 1) // 2,
                        ),
                        nn.LeakyReLU(relu_leakage),
                        MyConv1d(
                            n_channels,
                            n_channels,
                            kernel_size=kernel_size,
                            stride=1,
                            dilation=1,
                            padding=(kernel_size - 1) // 2,
                        ),
                    ]
                )
                for dilation in dilations
            ]
        )

    def forward(self, x: torch.Tensor):
        for conv in self.convs:
            x = x + conv(x)
        return x


class Generator(nn.Module):
    def __init__(
        self,
        generator_channels: int,
        generator_strides: list[int],
        generator_kernel_sizes: list[int],
        resblock_kernel_sizes: list[int],
        resblock_dilation_sizes: list[list[int]],
        relu_leakage: float,
    ):
        super().__init__()

        self.in_conv = nn.utils.weight_norm(
            nn.Conv1d(
                80,
                generator_channels,
                kernel_size=7,
                stride=1,
                dilation=1,
                padding=3,
            )
        )

        self.l_relu = nn.LeakyReLU(relu_leakage)

        self.upsamples = nn.Sequential(
            *[
                nn.utils.weight_norm(
                    nn.ConvTranspose1d(
                        generator_channels // (2**i),
                        generator_channels // (2 ** (i + 1)),
                        kernel_size=kernel_size,
                        stride=stride,
                        padding=(kernel_size - stride) // 2,
                    )
                )
                for i, (stride, kernel_size) in enumerate(
                    zip(generator_strides, generator_kernel_sizes)
                )
            ]
        )

        self.num_resnet_subblocks = len(resblock_kernel_sizes)
        self.res_blocks = nn.Sequential(
            *[
                ResBlock(
                    generator_channels // (2 ** (i + 1)),
                    kernel_size,
                    dilations,
                    relu_leakage,
                )
                for i in range(len(self.upsamples))
                for kernel_size, dilations in zip(
                    resblock_kernel_sizes, resblock_dilation_sizes
                )
            ]
        )

        self.out_conv = nn.Sequential(
            *[
                nn.LeakyReLU(relu_leakage),
                nn.utils.weight_norm(
                    nn.Conv1d(
                        generator_channels // (2 ** (len(resblock_kernel_sizes) + 1)),
                        1,
                        kernel_size=7,
                        stride=1,
                        dilation=1,
                        padding=3,
                    )
                ),
                nn.Tanh(),
            ]
        )

    def forward(self, spectrogram: torch.Tensor):
        x = self.in_conv(spectrogram)

        for i in range(len(self.upsamples)):
            x = self.l_relu(x)
            x = self.upsamples[i](x)

            x = x.transpose(-1, -2)

            for j in range(self.num_resnet_subblocks):
                if j == 0:
                    x = self.res_blocks[i * self.num_resnet_subblocks + j](x)
                else:
                    x = x + self.res_blocks[i * self.num_resnet_subblocks + j](x)
            
            x = x / self.num_resnet_subblocks

            x = x.transpose(-1, -2)

        x = self.out_conv(x)

        return x


class MSDBlock(torch.nn.Module):
    def __init__(
        self,
        relu_leakage: float,
        channels: list[int],
        kernel_sizes: list[int],
        strides: list[int],
        groups: list[int],
    ):
        super().__init__()

        self.convs = nn.Sequential(
            *[
                nn.utils.weight_norm(
                    nn.Conv1d(
                        in_channels,
                        out_channels,
                        kernel_size=kernel_size,
                        stride=stride,
                        padding=kernel_size // 2,
                        groups=group,
                    )
                )
                for in_channels, out_channels, kernel_size, stride, group in zip(
                    channels, channels[1:], kernel_sizes, strides, groups
                )
            ]
        )
        self.activation = nn.LeakyReLU(relu_leakage)

    def forward(self, x: torch.Tensor):
        fm = []
        for l in self.convs:
            x = l(x)
            x = self.activation(x)
            fm.append(x)

        x = torch.flatten(x, 1, -1)

        return x, fm


class MSD(torch.nn.Module):
    def __init__(
        self,
        relu_leakage: float,
        n_blocks: int,
        channels: list[int],
        msd_kernel_sizes: list[int],
        strides: list[int],
        groups: list[int],
    ):
        super().__init__()

        self.blocks = nn.Sequential(
            *[
                MSDBlock(
                    relu_leakage,
                    channels,
                    msd_kernel_sizes,
                    strides,
                    groups,
                )
                for _ in range(n_blocks)
            ]
        )

        self.avg_pools = nn.Sequential(
            *[nn.AvgPool1d(4, 2, padding=2) for _ in range(n_blocks - 1)]
        )

    def forward(self, target_wav: torch.Tensor, generated_wav: torch.Tensor):
        target_outputs = []
        gen_outputs = []
        target_fms = []
        gen_fms = []
        for i, block in enumerate(self.blocks):
            if i != 0:
                target_wav = self.avg_pools[i - 1](target_wav)
                generated_wav = self.avg_pools[i - 1](generated_wav)
            target_output, fm_target = block(target_wav)
            gen_output, fm_gen = block(generated_wav)
            target_outputs.append(target_output)
            gen_outputs.append(gen_output)
            target_fms.append(fm_target)
            gen_fms.append(fm_gen)

        return target_outputs, gen_outputs, target_fms, gen_fms


class MPDBlock(nn.Module):
    def __init__(
        self,
        period: int,
        kernel_size: int,
        stride: int,
        channels: list[int],
        relu_leakage: float,
    ):
        super().__init__()

        self.period = period
        self.activation = nn.LeakyReLU(relu_leakage)

        self.convs = nn.Sequential(
            *[
                nn.utils.weight_norm(
                    nn.Conv2d(
                        in_channels,
                        out_channels,
                        kernel_size=(kernel_size, 1),
                        stride=(stride, 1),
                        padding=(2, 0),
                    )
                )
                for in_channels, out_channels in zip(channels, channels[1:])
            ]
            + [
                nn.utils.weight_norm(
                    nn.Conv2d(
                        channels[-1],
                        channels[-1],
                        kernel_size=(kernel_size, 1),
                        stride=1,
                        padding=(2, 0),
                    )
                ),
                nn.utils.weight_norm(
                    nn.Conv2d(
                        channels[-1], 1, kernel_size=(3, 1), stride=1, padding=(1, 0)
                    )
                ),
            ]
        )

    def forward(self, x: torch.Tensor):
        fm = []

        x = x.squeeze()
        if len(x.shape) == 1:
            x = x.unsqueeze(0)

        batch_size, audio_len = x.shape
        if audio_len % self.period != 0:
            x = F.pad(x, (0, self.period - (audio_len % self.period)), "reflect")
            audio_len = x.size(-1)

        x = x.view(batch_size, audio_len // self.period, self.period)

        for conv in self.convs:
            x = self.activation(conv(x))
            fm.append(x)

        x = torch.flatten(x, 1, -1)

        return x, fm


class MPD(torch.nn.Module):
    def __init__(
        self,
        kernel_size: int,
        stride: int,
        relu_leakage: float,
        periods: list[int],
        channels: list[int],
    ):
        super().__init__()

        self.blocks = nn.Sequential(
            *[
                MPDBlock(
                    period,
                    kernel_size,
                    stride,
                    channels,
                    relu_leakage,
                )
                for period in periods
            ]
        )

    def forward(self, target_wav: torch.Tensor, generated_wav: torch.Tensor):
        target_outputs = []
        gen_outputs = []
        target_fms = []
        gen_fms = []

        for block in self.blocks:
            target_output, target_fm = block(target_wav)
            gen_output, gen_fm = block(generated_wav)
            target_outputs.append(target_output)
            gen_outputs.append(gen_output)
            target_fms.append(target_fm)
            gen_fms.append(gen_fm)

        return target_outputs, gen_outputs, target_fms, gen_fms


class HiFiGAN(nn.Module):
    def __init__(
        self,
        generator_channels: int,
        generator_strides: list[int],
        generator_kernel_sizes: list[int],
        resblock_kernel_sizes: list[int],
        resblock_dilation_sizes: list[list[int]],
        relu_leakage: float,
        n_msd_blocks: int,
        msd_channels: list[int],
        msd_kernel_sizes: list[int],
        msd_strides: list[int],
        msd_groups: list[int],
        mpd_kernel_size: int,
        mpd_stride: int,
        mpd_periods: list[int],
        mpd_channels: list[int],
    ):
        super().__init__()

        self.generator = Generator(
            generator_channels,
            generator_strides,
            generator_kernel_sizes,
            resblock_kernel_sizes,
            resblock_dilation_sizes,
            relu_leakage,
        )
        self.msd = MSD(
            relu_leakage,
            n_msd_blocks,
            msd_channels,
            msd_kernel_sizes,
            msd_strides,
            msd_groups,
        )
        self.mpd = MPD(
            mpd_kernel_size,
            mpd_stride,
            relu_leakage,
            mpd_periods,
            mpd_channels,
        )

    def train(self):
        self.generator.train()
        self.msd.train()
        self.mpd.train()

    def eval(self):
        self.generator.train()
        self.msd.train()
        self.mpd.train()

    def forward(self):
        raise NotImplementedError
