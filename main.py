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

from vqvae import baseline
from clf import ClassLoss
from encode_test import LabeledTestSet


def load_data(SZ):
    tfms = TF.Compose(
        [
            TF.RandomResizedCrop(SZ, scale=(0.5, 1.0), ratio=(1., 1.)),
            TF.RandomHorizontalFlip(),
            TF.ToTensor(),
        ]
    )

    ds = torchvision.datasets.ImageFolder("FCUNIST/train", transform=tfms)
    dst = LabeledTestSet(
        'reference_test.json',
        "FCUNIST/test",
        transform=TF.Compose(
            [
                TF.Resize(int(SZ * 1.17)),
                #TF.Resize(SZ),
                TF.CenterCrop(SZ), TF.ToTensor()]),
    )
    print("dataset size:", len(ds))
    print("test dataset size:", len(dst))
    dl = torch.utils.data.DataLoader(
        ds, batch_size=64, shuffle=True, num_workers=16, pin_memory=True, drop_last=True,
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
        from torchelie.models.snres_discr import res_discr_4l as D

        self.model = D().to_projection_discr(15)
        self.opt = torch.optim.AdamW(
            self.model.parameters(), lr=lr, betas=(0., 0.99))

    def forward(self, fake, real, y):
        from torchelie.loss.gan.penalty import R3
        self.opt.zero_grad()
        f_pred = self.model(fake.detach(), y)
        r_pred = self.model(real, y)
        loss = F.binary_cross_entropy_with_logits(
            r_pred - f_pred, torch.ones_like(r_pred))
        loss.backward()
        if True:
            cond_mod = (lambda x: self.model(x, torch.cat([y, y], dim=0)))
            grad_loss, grad_mag = R3(cond_mod, real, fake)
            loss = 0.0001 * grad_loss
            print("mag:", grad_mag, "R3 loss:", grad_loss.item())
            loss.backward()
        self.opt.step()

        f_pred = self.model(fake, y)
        loss = F.binary_cross_entropy_with_logits(
            f_pred - r_pred.detach(), torch.ones_like(f_pred))
        return loss


def main():
    SZ = 256
    lr = 2e-5
    device = "cuda"
    epoch = 240

    ploss = lpips.LPIPS(net="vgg", lpips=False).cuda()
    m = baseline(SZ).to(device)
    print(m)
    tag = f"vqvae-{lr}-cos"
    adv_loss = AdvLoss(2e-5).to(device)
    class_loss = ClassLoss(32, 256, 15).to(device)
    print(sum(p.numel() for p in m.parameters()) / 1e6, "M params")

    def train_fun(batch):
        x = batch[0]
        y = batch[1]
        zq = m[1](m[0](x * 2 - 1))
        recon = m[2](zq)

        class_pred = class_loss(zq)
        clf_loss = F.cross_entropy(class_pred, y)

        loss = ploss.forward(F.interpolate(recon * 2 - 1, size=SZ, mode="bilinear"), F.interpolate(x * 2 - 1, size=SZ, mode="bilinear")).mean()
        adv = adv_loss(recon * 2 - 1, x * 2 - 1, y)
        loss += clf_loss
        loss += 1 * adv
        loss.backward()
        return {"loss": loss.item(), "adv_loss": adv.item(), "clf_loss": clf_loss.item(), "class_acc": (class_pred.argmax(-1) == batch[1]).float().mean().item()}

    @torch.no_grad()
    def test_fun(batch):
        x = batch[0]
        zq = m[1](m[0](x * 2 - 1))
        recon = dec_ema(zq)

        class_pred = class_loss(zq)
        loss = ploss.forward(recon * 2 - 1, x * 2 - 1).mean()
        return {"recon": recon.detach(), "altered": x, "loss": loss.item(), "class_acc": (class_pred.argmax(-1) == batch[1]).float().mean().item()}

    dl, dlt = load_data(SZ)
    opt = torch.optim.AdamW([{"params": m[0].parameters(), "lr":1e-4}, {"params": m[1].parameters(), "lr": 1e-4}, {"params": m[2].parameters()}, {"params":class_loss.parameters(), "lr": 3e-4, "betas": (0.9, 0.999)}], lr=lr, betas=(0., 0.99), weight_decay=0.01)
    sched = tch.lr_scheduler.CosineDecay(opt, epoch * len(dl), warmup_ratio=0)
    all_models = nn.ModuleDict({
            "model": m,
            "grader": class_loss,
            "D": adv_loss.model,
        })
    #all_models.load_state_dict(torch.load("vqvae128.pth")["model"])
    tch.utils.load_state_dict_forgiving(all_models, torch.load("vqvae128.pth")["model"], fit_dst_size=True)
    import copy
    dec_ema = copy.deepcopy(m[2])
    all_models = nn.ModuleDict({
            "model": m,
            "grader": class_loss,
            "D": adv_loss.model,
            "ema": dec_ema
        })
    recipe = TrainAndTest(
        all_models, train_fun, test_fun, dl, dlt, log_every=10, test_every=10, visdom_env=None
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
            tcb.Log("clf_loss", "clf_loss"),
            tcb.Log("class_acc", "class_acc"),
            #tcb.Log("batch.0", "orig"),
            tcb.Polyak(m[2], dec_ema, 0.99)
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
            tcb.EpochMetricAvg("class_acc", False),
        ]
    )
    recipe.to(device)
    recipe.run(epoch)


if __name__ == "__main__":
    main()
