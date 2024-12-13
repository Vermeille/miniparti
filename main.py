import lpips

import torchelie as tch

import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision
import torchvision.transforms as TF
from torchelie.recipes import TrainAndTest
import torchelie.callbacks as tcb

from visdom import Visdom

from .vqvae import baseline


def load_data(SZ):
    tfms = TF.Compose(
        [
            TF.Resize(SZ),
            TF.CenterCrop(SZ),
            TF.RandomHorizontalFlip(),
            TF.ToTensor(),
        ]
    )

    ds = torchvision.datasets.ImageFolder("FCUNIST/train", transform=tfms)
    dst = tch.datasets.UnlabeledImages(
        "FCUNIST/test",
        transform=TF.Compose(
            [TF.Resize(SZ), TF.CenterCrop(SZ), TF.ToTensor()]),
    )
    print("dataset size:", len(ds))
    print("test dataset size:", len(dst))
    dl = torch.utils.data.DataLoader(
        ds, batch_size=256, shuffle=True, num_workers=16, pin_memory=True
    )
    dlt = torch.utils.data.DataLoader(
        dst,
        batch_size=64,
        shuffle=False,
        num_workers=16,
        drop_last=True,
        pin_memory=True,
    )
    return dl, dlt


class AdvLoss(nn.Module):
    def __init__(self, lr):
        super().__init__()
        from torchelie.models.snres_discr import residual_patch70

        self.model = residual_patch70()
        self.opt = torch.optim.AdamW(
            self.model.parameters(), lr=lr, betas=(0.9, 0.99))

    def forward(self, fake, real):
        self.opt.zero_grad()
        f_pred = self.model(fake.detach())
        loss = F.binary_cross_entropy_with_logits(
            f_pred, torch.zeros_like(f_pred))
        loss.backward()
        f_pred = self.model(real)
        loss = F.binary_cross_entropy_with_logits(
            f_pred, torch.ones_like(f_pred))
        loss.backward()
        self.opt.step()

        f_pred = self.model(fake)
        loss = F.binary_cross_entropy_with_logits(
            f_pred, torch.ones_like(f_pred))
        return loss


def main():
    SZ = 128
    lr = 3e-4
    device = "cuda"
    epoch = 120

    ploss = lpips.LPIPS(net="vgg", lpips=False).cuda()
    m = baseline().to(device)
    print(m)
    tag = f"vqvae-{lr}-cos"
    adv_loss = AdvLoss(1e-4).to(device)
    print(sum(p.numel() for p in m.parameters()) / 1e6, "M params")

    def train_fun(batch):
        x = batch[0]
        recon = m(x * 2 - 1)
        loss = ploss.forward(recon * 2 - 1, x * 2 - 1).mean()
        adv = adv_loss(recon * 2 - 1, x * 2 - 1)
        loss += 1 * adv
        loss.backward()
        return {"loss": loss.item(), "adv_loss": adv.item()}

    @torch.no_grad()
    def test_fun(batch):
        x = batch[0]
        recon = m(x * 2 - 1)
        loss = ploss.forward(recon * 2 - 1, x * 2 - 1).mean()
        return {"recon": recon.detach(), "altered": x, "loss": loss.item()}

    dl, dlt = load_data(SZ)
    opt = torch.optim.AdamW(m.parameters(), lr=lr)
    sched = tch.lr_scheduler.CosineDecay(opt, epoch * len(dl))
    recipe = TrainAndTest(
        m, train_fun, test_fun, dl, dlt, log_every=1, test_every=20, visdom_env=None
    )

    recipe.callbacks.cbs[-1][0].vis = Visdom(
        env=tag, server="https://visdom.vermeille.fr", port=443
    )
    recipe.callbacks.cbs[-1][0].vis.close()
    recipe.callbacks.add_callbacks(
        [
            tcb.Optimizer(opt, log_lr=True, centralize_grad=True),
            tcb.LRSched(sched, step_each_batch=True, metric=None),
            tcb.Log("loss", "loss"),
            tcb.Log("adv_loss", "adv_loss"),
        ]
    )
    recipe.test_loop.callbacks.cbs[-1][0].vis = Visdom(
        env=tag, server="https://visdom.vermeille.fr", port=443
    )
    recipe.test_loop.callbacks.add_callbacks(
        [
            tcb.EpochMetricAvg("loss", False),
            tcb.Log("recon", "recon"),
            tcb.Log("altered", "altered"),
        ]
    )
    recipe.to(device)
    recipe.run(epoch)


if __name__ == "__main__":
    main()
