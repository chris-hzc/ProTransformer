"""ProAttention: plug-and-play robust attention (ProTransformer, NeurIPS 2024).

Vanilla attention computes ``Z = A @ V``, the solution of a weighted least-squares (WLS) token estimator.
``ProAttention`` replaces it with a robust estimator (L1 / Huber / MCP / Huber-MCP penalty) solved by a few
Newton-IRLS steps:

    D = cdist(Z, V);  W = rho'(D) / D;  Z = normalize(W * A, p=1) @ V

Every attention layer patched in this package owns a ``pro_attention`` module. With ``norm="L2"`` (the default) it
reduces exactly to vanilla attention, so pretrained weights behave as usual until ProAttention is switched on with
:func:`set_pro_attention`.
"""

import torch
from torch import nn

ROBUST_NORMS = ("L2", "L1", "Huber", "MCP", "HuberMCP")


class ProAttention(nn.Module):
    def __init__(self, L=3, norm="L2", epsilon=1e-2, gamma=4.0, t=1.0, delta=4.0):
        super().__init__()
        self.L = L
        self.norm = norm
        self.epsilon = epsilon
        self.gamma = gamma
        self.t = t
        self.delta = delta

    @property
    def enabled(self):
        return self.norm != "L2" and self.L > 0

    def forward(self, A, V):
        M = torch.matmul(A, V)

        if not self.enabled:
            return M

        for _ in range(self.L):
            dist = torch.cdist(M.float(), V.float())

            if self.norm == "L1":
                w = 1 / (dist + self.epsilon)

            elif self.norm == "MCP":
                w = 1 / (dist + self.epsilon) - 1 / self.gamma
                w[w < self.epsilon] = self.epsilon

            elif self.norm == "Huber":
                w = self.delta / (dist + self.epsilon)
                w[w > 1.0] = 1.0

            elif self.norm == "HuberMCP":
                w = self.delta / (self.gamma - self.delta) * (self.gamma / (dist + self.epsilon) - 1)
                w = w.clamp(min=self.epsilon, max=1.0)

            else:
                raise ValueError(f"Unknown robust norm {self.norm!r}, expected one of {ROBUST_NORMS}")

            ww = w.to(A.dtype) * A

            ww_norm = nn.functional.normalize(ww, p=1, dim=-1)

            M = (1.0 - self.t) * M + self.t * torch.matmul(ww_norm, V)

        return M


def set_pro_attention(model, norm="MCP", L=3, gamma=4.0, epsilon=1e-2, delta=4.0, t=1.0):
    """Turn every attention layer of ``model`` into ProAttention (in place) and return the model.

    Use ``norm="L2"`` to restore vanilla attention.
    """
    if norm not in ROBUST_NORMS:
        raise ValueError(f"Unknown robust norm {norm!r}, expected one of {ROBUST_NORMS}")
    if norm == "HuberMCP" and not gamma > delta:
        raise ValueError(f"HuberMCP needs gamma > delta, got gamma={gamma}, delta={delta}")

    modules = [m for m in model.modules() if isinstance(m, ProAttention)]
    if not modules:
        raise ValueError(f"{type(model).__name__} has no ProAttention layers; is it loaded from `protransformers`?")

    attn_implementation = getattr(getattr(model, "config", None), "_attn_implementation", "eager")
    if norm != "L2" and attn_implementation != "eager":
        raise ValueError(
            f"ProAttention needs the explicit attention weights, but the model uses {attn_implementation!r}. "
            'Load it with `from_pretrained(..., attn_implementation="eager")`.'
        )

    for m in modules:
        m.norm, m.L, m.gamma, m.epsilon, m.delta, m.t = norm, L, gamma, epsilon, delta, t
    return model


# Backward-compatible aliases
RobustSum = ProAttention
set_robust_params = set_pro_attention
