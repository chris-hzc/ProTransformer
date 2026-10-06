"""FGSM / PGD attacks on Pro-ViT (CIFAR-10).

    python image_attack.py --norm L2               # vanilla ViT
    python image_attack.py --norm MCP --gamma 4.0  # Pro-ViT
"""

import argparse

import torch
import torch.nn.functional as F
import torchvision
from torch import nn

from protransformers import ViTForImageClassification, set_pro_attention


parser = argparse.ArgumentParser(description="ProTransformer (ViT) under FGSM / PGD")

parser.add_argument("--model", type=str, default="nateraw/vit-base-patch16-224-cifar10")
parser.add_argument("--norm", type=str, default="MCP", choices=["L2", "L1", "Huber", "MCP", "HuberMCP"])
parser.add_argument("--gamma", type=float, default=4.0)
parser.add_argument("--delta", type=float, default=4.0)
parser.add_argument("--epsilon", type=float, default=1e-2)
parser.add_argument("--L", type=int, default=3)
parser.add_argument("--attack", type=str, default="pgd", choices=["fgsm", "pgd"])
parser.add_argument("--budgets", type=float, nargs="+", default=[1, 4, 8], help="L_inf budgets, in units of 1/255")
parser.add_argument("--steps", type=int, default=7, help="PGD steps")
parser.add_argument("--alpha", type=float, default=2 / 255, help="PGD step size")
parser.add_argument("--num_example", type=int, default=1000)
parser.add_argument("--batch_size", type=int, default=50)
parser.add_argument("--data_dir", type=str, default="./data")
parser.add_argument("--seed", type=int, default=0)

args = parser.parse_args()


class Classifier(nn.Module):
    """Maps raw 32x32 images in [0, 1] to logits, so attacks operate in pixel space."""

    def __init__(self, vit):
        super().__init__()
        self.vit = vit
        self.size = vit.config.image_size

    def forward(self, x):
        x = F.interpolate(x, size=(self.size, self.size), mode="bilinear", align_corners=False)
        x = (x - 0.5) / 0.5
        return self.vit(pixel_values=x).logits


def fgsm(model, x, y, eps):
    x_adv = x.clone().requires_grad_(True)
    loss = F.cross_entropy(model(x_adv), y)
    grad = torch.autograd.grad(loss, x_adv)[0]
    return (x + eps * grad.sign()).clamp(0, 1).detach()


def pgd(model, x, y, eps, alpha, steps):
    x_adv = (x + torch.empty_like(x).uniform_(-eps, eps)).clamp(0, 1)
    for _ in range(steps):
        x_adv.requires_grad_(True)
        loss = F.cross_entropy(model(x_adv), y)
        grad = torch.autograd.grad(loss, x_adv)[0]
        x_adv = x_adv.detach() + alpha * grad.sign()
        x_adv = (x + (x_adv - x).clamp(-eps, eps)).clamp(0, 1)
    return x_adv.detach()


def main():
    print(args)
    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    vit = ViTForImageClassification.from_pretrained(args.model)
    set_pro_attention(vit, norm=args.norm, L=args.L, gamma=args.gamma, delta=args.delta, epsilon=args.epsilon)
    model = Classifier(vit).to(device).eval()

    testset = torchvision.datasets.CIFAR10(
        args.data_dir, train=False, download=True, transform=torchvision.transforms.ToTensor()
    )
    subset = torch.utils.data.Subset(testset, range(min(args.num_example, len(testset))))
    loader = torch.utils.data.DataLoader(subset, batch_size=args.batch_size, shuffle=False, num_workers=2)

    budgets = [0.0] + list(args.budgets)
    correct = {b: 0 for b in budgets}
    total = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        total += len(y)
        for b in budgets:
            eps = b / 255
            if b == 0:
                x_adv = x
            elif args.attack == "fgsm":
                x_adv = fgsm(model, x, y, eps)
            else:
                x_adv = pgd(model, x, y, eps, args.alpha, args.steps)
            with torch.no_grad():
                correct[b] += (model(x_adv).argmax(-1) == y).sum().item()
        print(f"[{total}/{len(subset)}] " + "  ".join(f"{b:g}/255: {100 * correct[b] / total:.2f}%" for b in budgets))

    print(f"\n{args.attack.upper()} robust accuracy ({args.norm}, {total} examples)")
    for b in budgets:
        print(f"  budget {'clean' if b == 0 else f'{b:g}/255':>8s}: {100 * correct[b] / total:.2f}%")


if __name__ == "__main__":
    main()
