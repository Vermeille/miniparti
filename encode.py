import random
import torch
import torchvision
import torchvision.transforms as TF
from vqvae import baseline


def load_data(SZ):
    tfms = TF.Compose(
        [
            TF.Resize(SZ),
            TF.CenterCrop(SZ),
            TF.ToTensor(),
        ]
    )

    ds = torchvision.datasets.ImageFolder("FCUNIST/train", transform=tfms)
    dl = torch.utils.data.DataLoader(
        ds, batch_size=256, shuffle=False, num_workers=16, pin_memory=True
    )
    return dl


@torch.no_grad()
def main():
    import sys

    device = "cpu"
    dl = load_data(256)
    model = baseline(256)
    torch.nn.ModuleDict({"model": model}).load_state_dict(torch.load(sys.argv[1], map_location=device)[
                          "model"], strict=False)
    model.eval()
    model[1].return_indices = True

    for i, (x, y) in enumerate(dl):
        x = x.to(device)
        z = model[0](x * 2 - 1)
        _, indices = model[1](z)
        for j, (ind, yy) in enumerate(zip(indices, y)):
            filename = dl.dataset.samples[i * 256 + j][0]
            print(filename, " ".join([str(yy.item())] + [str(ii) for ii in ind.reshape(-1).tolist()]))

        z = model[0](TF.functional.hflip(x) * 2 - 1)
        _, indices = model[1](z)
        for j, (ind, yy) in enumerate(zip(indices, y)):
            filename = dl.dataset.samples[i * 256 + j][0] + "(mirror)"
            print(filename, " ".join([str(yy.item())] + [str(ii) for ii in ind.reshape(-1).tolist()]))


if __name__ == "__main__":
    main()
