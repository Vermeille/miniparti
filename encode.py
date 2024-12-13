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
        ds, batch_size=256, shuffle=True, num_workers=16, pin_memory=True
    )
    return dl


@torch.no_grad()
def main():
    import sys

    device = "cpu"
    dl = load_data(128)
    model = baseline()
    model.load_state_dict(torch.load(sys.argv[1])[
                          "model"], map_location=device)
    model.eval()
    model[1].return_indices = True

    for i, (x, _) in enumerate(dl):
        x = x.to(device)
        z = model[0](x * 2 - 1)
        _, indices = model[1](z)
        for ind in indices:
            print(" ".join(ind.view(-1).tolist()))


if __name__ == "__main__":
    main()
