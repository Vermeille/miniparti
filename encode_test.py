import json
import random
import torch
import torchvision
import torchvision.transforms as TF
from vqvae import baseline
from torchelie.datasets import UnlabeledImages


class LabeledTestSet(UnlabeledImages):
    def __init__(self, ref_file, path, transform):
        super().__init__(path, transform)
        with open(ref_file, 'r') as f:
            labels = json.load(f)
        classes = list(sorted(set(labels.values())))
        class_to_idx = {cls: i for i, cls in enumerate(classes)}
        self.labels = {k: class_to_idx[v] for k, v in labels.items()}

    def __getitem__(self, idx):
        path = self.samples[idx].split('/')[-1]
        cls = self.labels[path]
        return super().__getitem__(idx) + [cls]


def load_data(SZ):
    tfms = TF.Compose(
        [
            TF.Resize(SZ),
            TF.CenterCrop(SZ),
            TF.ToTensor(),
        ]
    )

    ds = LabeledTestSet("reference_test.json", "FCUNIST/test", transform=tfms)
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
    torch.nn.ModuleDict({"model":model}).load_state_dict(torch.load(sys.argv[1], map_location=device)[
                          "model"], strict=False)
    model.eval()
    model[1].return_indices = True

    for i, (x, y) in enumerate(dl):
        print(x.shape, y.shape)
        x = x.to(device)
        z = model[0](x * 2 - 1)
        _, indices = model[1](z)
        for j, ind in enumerate(indices):
            filename = dl.dataset.samples[i * 256 + j]
            cls = y[j].item()
            print(f"test_{i*256+j}.png", " ".join([str(cls)] + [str(ii) for ii in ind.reshape(-1).tolist()]))

        z = model[0](TF.functional.hflip(x) * 2 - 1)
        _, indices = model[1](z)
        for j, ind in enumerate(indices):
            filename = dl.dataset.samples[i * 256 + j]
            cls = y[j].item()
            print(f"test_{i*256+j}.png(mirror)", " ".join([str(cls)] + [str(ii) for ii in ind.reshape(-1).tolist()]))


if __name__ == "__main__":
    main()
