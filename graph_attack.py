"""Adaptive PGD topology attack (Xu et al., 2019) on Pro-GAT (Cora / Citeseer node classification).

A GAT is trained once per split on the clean graph. ProAttention is then plugged into its neighbor aggregation
without retraining, and both the vanilla GAT and Pro-GAT are attacked adaptively (global evasion: the attacker
flips up to `budget * #edges` edges to minimize the tanh-margin over the test nodes).

    python graph_attack.py --norm MCP --gamma 4.0 --L 3 --budgets 0.05 0.1 0.2
"""

import argparse
import math
import os
import urllib.request

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn.functional as F
from scipy.sparse.csgraph import connected_components
from sklearn.model_selection import train_test_split
from torch import nn
from tqdm import trange

from protransformers import ProAttention, set_pro_attention


parser = argparse.ArgumentParser(description="ProTransformer (GAT) under adaptive PGD topology attack")

parser.add_argument("--data", type=str, default="cora", choices=["cora", "citeseer"])
parser.add_argument("--data_dir", type=str, default="./data")
parser.add_argument("--norm", type=str, default="MCP", choices=["L2", "L1", "Huber", "MCP", "HuberMCP"])
parser.add_argument("--gamma", type=float, default=4.0)
parser.add_argument("--delta", type=float, default=4.0)
parser.add_argument("--epsilon", type=float, default=1e-2)
parser.add_argument("--L", type=int, default=3)
parser.add_argument("--budgets", type=float, nargs="+", default=[0.05, 0.1, 0.2], help="Fraction of edges to flip")
parser.add_argument("--num_splits", type=int, default=5)
parser.add_argument("--hidden", type=int, default=8)
parser.add_argument("--heads", type=int, default=8)
parser.add_argument("--dropout", type=float, default=0.6)
parser.add_argument("--lr", type=float, default=5e-3)
parser.add_argument("--weight_decay", type=float, default=5e-4)
parser.add_argument("--epochs", type=int, default=500)
parser.add_argument("--attack_iters", type=int, default=200)
parser.add_argument("--seed", type=int, default=0)

args = parser.parse_args()

# Graphs and 10/10/80 splits as in "Are Defenses for Graph Neural Networks Robust?" (Mujkanovic et al., 2022)
DATA_URL = "https://raw.githubusercontent.com/LoadingByte/are-gnn-defenses-robust/master/data/{}.npz"
SPLIT_SEEDS = [1534, 2021, 1323, 1535, 1698]


# ----------------------------------------------------------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------------------------------------------------------


def load_graph(name, data_dir):
    path = os.path.join(data_dir, f"{name}.npz")
    if not os.path.exists(path):
        os.makedirs(data_dir, exist_ok=True)
        urllib.request.urlretrieve(DATA_URL.format(name), path)

    with np.load(path, allow_pickle=True) as loader:
        loader = dict(loader)

    def csr(prefix):
        return sp.csr_matrix(
            (loader[f"{prefix}_data"], loader[f"{prefix}_indices"], loader[f"{prefix}_indptr"]), loader[f"{prefix}_shape"]
        )

    # Symmetrize, drop self loops, keep the largest connected component
    A = csr("adj")
    A = A - sp.diags(A.diagonal())
    A = A + A.T
    A[A > 1] = 1
    _, components = connected_components(A)
    lcc = np.nonzero(components == np.bincount(components).argmax())[0]

    A = torch.tensor(A[lcc][:, lcc].todense(), dtype=torch.float32)
    X = torch.tensor(csr("attr")[lcc].todense(), dtype=torch.float32)
    y = torch.tensor(loader["labels"][lcc], dtype=torch.int64)
    return A, X, y


def make_split(y, seed):
    idx = np.arange(len(y))
    idx_trval, idx_test = train_test_split(idx, train_size=0.2, stratify=y, random_state=seed)
    idx_train, idx_val = train_test_split(idx_trval, train_size=0.5, stratify=y[idx_trval], random_state=seed)
    return torch.tensor(idx_train), torch.tensor(idx_val), torch.tensor(idx_test)


# ----------------------------------------------------------------------------------------------------------------------
# Pro-GAT
# ----------------------------------------------------------------------------------------------------------------------


class GATLayer(nn.Module):
    """Dense multi-head graph attention whose neighbor aggregation is a ProAttention `ProAttention`."""

    def __init__(self, in_dim, out_dim, heads, dropout, concat=True):
        super().__init__()
        self.heads, self.out_dim, self.concat = heads, out_dim, concat
        self.lin = nn.Linear(in_dim, heads * out_dim, bias=False)
        self.att_src = nn.Parameter(torch.empty(heads, 1, out_dim))
        self.att_dst = nn.Parameter(torch.empty(heads, 1, out_dim))
        self.bias = nn.Parameter(torch.zeros(heads * out_dim if concat else out_dim))
        self.dropout = nn.Dropout(dropout)
        self.pro_attention = ProAttention()
        nn.init.xavier_uniform_(self.lin.weight)
        nn.init.xavier_uniform_(self.att_src)
        nn.init.xavier_uniform_(self.att_dst)

    def forward(self, adj, x):
        N = x.shape[0]
        h = self.lin(x).view(N, self.heads, self.out_dim).transpose(0, 1)  # (heads, N, out_dim)

        # e_ij = LeakyReLU(a_dst . h_i + a_src . h_j)
        e = (h * self.att_dst).sum(-1, keepdim=True) + (h * self.att_src).sum(-1).unsqueeze(1)
        e = F.leaky_relu(e, 0.2)

        # Softmax over neighbors, weighted by the (possibly relaxed) adjacency: identical to GAT for a binary `adj`, and
        # differentiable w.r.t. adding edges as well as removing them
        attention = adj * torch.exp(e - e.amax(-1, keepdim=True))
        attention = self.dropout(attention / attention.sum(-1, keepdim=True))

        out = self.pro_attention(attention, h)  # vanilla GAT: attention @ h
        out = out.transpose(0, 1).reshape(N, -1) if self.concat else out.mean(0)
        return out + self.bias


class GAT(nn.Module):
    def __init__(self, in_dim, hidden, num_classes, heads, dropout):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.conv1 = GATLayer(in_dim, hidden, heads, dropout, concat=True)
        self.conv2 = GATLayer(hidden * heads, num_classes, 1, dropout, concat=False)

    def forward(self, adj, x):
        adj = adj + torch.eye(adj.shape[0], device=adj.device)  # self loops
        x = F.elu(self.conv1(adj, self.dropout(x)))
        return self.conv2(adj, self.dropout(x))


def accuracy(logits, y):
    return (logits.argmax(-1) == y).float().mean().item()


def train(model, A, X, y, idx_train, idx_val):
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    best_val, best_state = -1, None
    for _ in trange(args.epochs, desc="train", leave=False):
        model.train()
        optimizer.zero_grad()
        F.cross_entropy(model(A, X)[idx_train], y[idx_train]).backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val = accuracy(model(A, X)[idx_val], y[idx_val])
        if val > best_val:
            best_val, best_state = val, {k: v.clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    return model.eval()


# ----------------------------------------------------------------------------------------------------------------------
# PGD topology attack (Xu et al., 2019), global evasion
# ----------------------------------------------------------------------------------------------------------------------


def margin(logits, y):
    true = logits.gather(1, y[:, None]).squeeze(1)
    other = logits.scatter(1, y[:, None], float("-inf")).amax(1)
    return true - other


def symmetric(flip):
    triu = flip.triu(diagonal=1)
    return triu + triu.T


def perturb(A, flip):
    return A + flip * (1 - 2 * A)


def pgd_attack(model, A, X, y, idx_test, budget, iterations=200, base_lr=0.1, grad_clip=1.0, samples=100):
    def loss_fn(flip_sym):
        return margin(model(perturb(A, flip_sym), X)[idx_test], y[idx_test]).tanh().mean()

    flip = torch.zeros_like(A, requires_grad=True)
    for it in trange(iterations, desc="attack", leave=False):
        grad = torch.autograd.grad(loss_fn(symmetric(flip)), flip)[0]
        grad_norm = grad.norm()
        if grad_norm > grad_clip:
            grad = grad * grad_clip / grad_norm

        with torch.no_grad():
            flip -= base_lr * budget / math.sqrt(it + 1) * grad
            if flip.clamp(0, 1).sum() <= budget:
                flip.clamp_(0, 1)
            else:  # project onto {0 <= flip <= 1, sum(flip) <= budget} by bisection
                top, bot = flip.max().item(), (flip.min() - 1).clamp_min(0).item()
                mu = (top + bot) / 2
                while (top - bot) / 2 > 1e-5:
                    used = (flip - mu).clamp(0, 1).sum()
                    if used == budget:
                        break
                    bot, top = (mu, top) if used > budget else (bot, mu)
                    mu = (top + bot) / 2
                flip.sub_(mu).clamp_(0, 1)

    # Random sampling of a binary perturbation within budget
    flip = flip.detach().triu(diagonal=1)
    best_loss, best_flip = float("inf"), torch.zeros_like(flip)
    with torch.no_grad():
        tries = 0
        while tries < samples:
            sample = flip.bernoulli()
            if sample.sum() <= budget:
                tries += 1
                loss = loss_fn(symmetric(sample)).item()
                if loss < best_loss:
                    best_loss, best_flip = loss, sample
    return perturb(A, symmetric(best_flip))


def evaluate(model, A, X, y, idx_test, budget_edges):
    if budget_edges == 0:
        A_adv = A
    else:
        A_adv = pgd_attack(model, A, X, y, idx_test, budget_edges, iterations=args.attack_iters)
    with torch.no_grad():
        return accuracy(model(A_adv, X)[idx_test], y[idx_test])


def main():
    print(args)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    A, X, y = load_graph(args.data, args.data_dir)
    A, X, y = A.to(device), X.to(device), y.to(device)
    num_edges = int(A.sum().item() // 2)
    print(f"{args.data}: {A.shape[0]} nodes, {num_edges} edges, {X.shape[1]} features, {int(y.max()) + 1} classes")

    budgets = [0.0] + list(args.budgets)
    variants = {"GAT": dict(norm="L2"), f"Pro-GAT ({args.norm})": dict(
        norm=args.norm, L=args.L, gamma=args.gamma, delta=args.delta, epsilon=args.epsilon
    )}
    results = {name: {b: [] for b in budgets} for name in variants}

    for split in range(args.num_splits):
        torch.manual_seed(args.seed + split)
        idx_train, idx_val, idx_test = make_split(y.cpu().numpy(), SPLIT_SEEDS[split % len(SPLIT_SEEDS)])

        model = GAT(X.shape[1], args.hidden, int(y.max()) + 1, args.heads, args.dropout).to(device)
        model = train(model, A, X, y, idx_train, idx_val)

        for name, params in variants.items():
            set_pro_attention(model, **params)  # plug-and-play: same trained weights
            for b in budgets:
                results[name][b].append(100 * evaluate(model, A, X, y, idx_test, int(b * num_edges)))
            print(f"[split {split}] {name:18s} " + "  ".join(f"{b:.0%}: {results[name][b][-1]:.1f}" for b in budgets))

    print(f"\nAdaptive PGD (global evasion) on {args.data}, {args.num_splits} splits")
    print(f"{'Model':18s} " + " ".join(f"{f'{b:.0%}':>13s}" for b in budgets))
    for name in variants:
        row = " ".join(f"{np.mean(r):6.1f} ± {np.std(r):4.1f}" for r in results[name].values())
        print(f"{name:18s} {row}")


if __name__ == "__main__":
    main()
