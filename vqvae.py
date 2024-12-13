import torch
import torch.nn as nn
import torch.nn.functional as F

import torchelie.nn as tnn
from torchelie.utils import xavier


class PosEnc(nn.Module):
    def __init__(self, *shape):
        super().__init__()
        self.pos = nn.Parameter(torch.zeros(*shape))

    def forward(self, x):
        return x + self.pos


class Res(nn.Module):
    def __init__(self, mod):
        super(Res, self).__init__()
        self.go = mod

    def forward(self, x):
        return self.go(x) + x


class NormChannel(nn.Module):
    def forward(self, x):
        return F.normalize(x, dim=1)


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-8):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(dim, 1, 1))

        self.eps = eps

    def forward(self, x):
        return (
            x * torch.rsqrt(x.pow(2).mean(dim=1, keepdim=True) +
                            self.eps) * self.scale
        )


class Encoder(nn.Module):
    def __init__(self, arch, hidden=128, vq_dim=8):
        super(Encoder, self).__init__()
        layers = [
            tnn.Conv2d(3, hidden, 5),
            nn.BatchNorm2d(hidden, affine=True),
            nn.ReLU(inplace=True),
        ]

        self.start_hidden = hidden
        for layer in arch:
            ch = min(512, hidden)
            if layer == "r":
                layers.append(tnn.PreactResBlock(ch, ch))
            elif layer == "q":
                layers.append(PosEnc(ch, 8, 8))
                (
                    layers.append(
                        Res(
                            nn.Sequential(
                                RMSNorm(ch), tnn.SelfAttention2d(
                                    ch, ch // 64, ch)
                            )
                        )
                    ),
                )
                layers.append(
                    xavier(nn.Conv2d(ch, vq_dim, 3, stride=1, padding=1)))
            elif layer == "p":
                nxt_hidden = hidden * 2
                if ch >= 256:
                    (
                        layers.append(
                            Res(
                                nn.Sequential(
                                    RMSNorm(ch), tnn.SelfAttention2d(
                                        ch, ch // 64, ch)
                                )
                            )
                        ),
                    )
                layers.append(
                    xavier(nn.Conv2d(ch, min(512, nxt_hidden),
                           3, stride=2, padding=1))
                )
                layers.append(nn.BatchNorm2d(
                    min(512, nxt_hidden), affine=True))
                hidden = nxt_hidden
        self.end_hidden = hidden

        self.layers = nn.ModuleList(layers)

    def forward(self, x, y=None, ret_idx=False):
        qs = []
        idxs = []
        for m in self.layers:
            if isinstance(m, tnn.VQ):
                x, idx = m(x)
                qs.append(x.detach())
                idxs.append(idx)
            else:
                x = m(x)

        return x
        if ret_idx:
            return qs[0], idxs[0]
        else:
            return qs[0]


class Decoder(nn.Module):
    def __init__(self, arch, hidden=128, vq_dim=8):
        super(Decoder, self).__init__()
        layers = [
            xavier(nn.Conv2d(vq_dim, min(512, hidden), 3, padding=1)),
            PosEnc(min(512, hidden), 8, 8),
            Res(
                nn.Sequential(
                    RMSNorm(min(512, hidden)),
                    tnn.SelfAttention2d(
                        min(512, hidden), min(
                            512, hidden) // 64, min(512, hidden)
                    ),
                )
            ),
        ]

        for layer in arch:
            ch = min(512, hidden)
            if layer == "r":
                layers.append(tnn.PreactResBlock(ch, ch))
            elif layer == "u":
                nxt_hidden = hidden // 2
                if False:
                    layers.append(
                        nn.ConvTranspose2d(
                            hidden, hidden, 4, stride=2, padding=1)
                    )  # nn.UpsamplingNearest2d(scale_factor=2))
                elif True:
                    layers.append(nn.UpsamplingBilinear2d(scale_factor=2))
                    layers.append(
                        xavier(nn.Conv2d(ch, min(512, nxt_hidden), 1, padding=0))
                    )
                    # layers.append(nn.BatchNorm2d(nxt_hidden, affine=True))
                else:
                    layers.append(
                        xavier(
                            nn.Conv2d(ch, min(512, nxt_hidden) * 4, 1, padding=0))
                    )
                    layers.append(nn.PixelShuffle(2))
                    # layers.append(xavier(nn.Conv2d(ch // 4, min(512, nxt_hidden), 1, padding=0)))
                    layers.append(nn.BatchNorm2d(
                        min(512, nxt_hidden), affine=True))
                hidden = nxt_hidden

        layers += [
            nn.BatchNorm2d(hidden, affine=True),
            xavier(tnn.Conv2d(hidden, 3, 1)),
        ]

        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        for m in self.layers:
            x = m(x)

        return torch.sigmoid(x)


def AE(enc, dec, hidden=64, vq_dim=8, codebook=1024):
    enc = Encoder(enc, hidden, vq_dim=vq_dim)
    vq = tnn.VQ(
        vq_dim, codebook, dim=1, max_age=20, space="angular", return_indices=False
    )
    return nn.Sequential(enc, vq, Decoder(dec, enc.end_hidden, vq_dim=vq_dim))


def baseline():
    return AE("rprrprrprrrprrrq", "rrrurrrrurrrurrrurrr", hidden=32, vq_dim=32)
