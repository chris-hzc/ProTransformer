<div align="center">

# 🛡️ ProTransformer

### Robustify Transformers via Plug-and-Play Paradigm

**[Zhichao Hou](https://github.com/chris-hzc)<sup>1</sup> · Weizhi Gao<sup>1</sup> · Yuchen Shen<sup>2</sup> · Feiyi Wang<sup>3</sup> · Xiaorui Liu<sup>1,✉</sup>**

<sup>1</sup>North Carolina State University &nbsp;&nbsp; <sup>2</sup>Carnegie Mellon University &nbsp;&nbsp; <sup>3</sup>Oak Ridge National Laboratory

<img src="https://img.shields.io/badge/NeurIPS-2024-6f42c1?style=for-the-badge" alt="NeurIPS 2024">
<a href="https://arxiv.org/abs/2410.23182"><img src="https://img.shields.io/badge/arXiv-2410.23182-b31b1b?style=for-the-badge&logo=arxiv&logoColor=white" alt="arXiv"></a>
<a href="https://arxiv.org/pdf/2410.23182"><img src="https://img.shields.io/badge/Paper-PDF-1f6feb?style=for-the-badge&logo=adobeacrobatreader&logoColor=white" alt="PDF"></a>
<br>
<img src="https://img.shields.io/badge/Python-3.10-3776AB?logo=python&logoColor=white" alt="Python 3.10">
<img src="https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white" alt="PyTorch">
<img src="https://img.shields.io/badge/🤗%20Transformers-4.40-FFD21E" alt="Transformers">
<img src="https://img.shields.io/badge/TextAttack-attacks-2ea44f" alt="TextAttack">
<a href="https://github.com/chris-hzc/ProTransformer/stargazers"><img src="https://img.shields.io/github/stars/chris-hzc/ProTransformer?style=social" alt="GitHub stars"></a>

<p>
<b>No retraining. No fine-tuning. Just swap the attention.</b><br>
<i>A 4-line robust attention layer that plugs into any pretrained transformer — language, vision, or graph.</i>
</p>

</div>

<p align="center">
  <img src="./figures/textattack.png" width="61.5%" />
  <img src="./figures/protransformer.png" width="35%" />
</p>

---

## 📰 News

- **[2024]** 🎉 ProTransformer is accepted to **NeurIPS 2024**!
- **[2024.10]** 📄 Paper is released on [arXiv](https://arxiv.org/abs/2410.23182) and the code is open-sourced.

## ✨ Highlights

| | |
|:--|:--|
| 🔌 **Plug-and-Play** | Drop **ProAttention** into an already-trained transformer with **one function call** — no extra training or fine-tuning required. |
| 🧮 **Principled** | Attention ≡ a *weighted least-squares* token estimator. We replace it with a **robust** estimator (ℓ₁ / Huber / MCP) solved by a **Newton-IRLS** algorithm with a loss-descent guarantee. |
| ⚡ **Simple & Efficient** | ~4 core lines of PyTorch; converges in **K ≤ 3** iterations. |
| 🌍 **Universal** | Works across tasks, attacks, backbones (BERT, RoBERTa, ALBERT, DistilBERT, T5, LLaMA, Vicuna, ViT, GAT) and domains (text, image, graph). |

### 📈 Robustness gains at a glance

Without any further fine-tuning, ProTransformer improves the robustness of vanilla models by:

| Setting | Model | Improvement |
|:--|:--|:--:|
| TextFooler attack | BERT · ALBERT · DistilBERT · RoBERTa | **+19.5% · +28.3% · +16.1% · +11.4%** |
| Prompt attack | T5 · LLaMA | **+24.8% · +17.8%** |
| Jailbreak attack | Vicuna | **+10.4%** (avg.) |

…and it also shows strong robustness in the **vision** and **graph** domains.

---

## 🧠 Method

#### 1. Attention is a weighted least-squares (WLS) estimator

Each output token aggregates the value vectors with attention weights, which is exactly the solution of a weighted least-squares problem:

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="./figures/eq_wls_dark.svg">
    <img src="./figures/eq_wls_light.svg" alt="z = argmin_z sum_j a_j ||v_j - z||^2 = sum_j a_j v_j">
  </picture>
</p>

The quadratic penalty lets a few **adversarially perturbed tokens dominate** the output — the root of the vulnerability.

#### 2. Robust token estimator

We replace the quadratic loss with a robust penalty $\rho$ (ℓ₁, Huber, MCP, Huber-MCP):

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="./figures/eq_robust_dark.svg">
    <img src="./figures/eq_robust_light.svg" alt="z = argmin_z sum_j a_j rho(||v_j - z||)">
  </picture>
</p>

#### 3. Newton-IRLS → ProAttention

Optimizing a convex localized upper bound with a Newton step yields a closed-form **re-weighted attention**:

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="./figures/eq_irls_dark.svg">
    <img src="./figures/eq_irls_light.svg" alt="z^(k+1) = sum_j a_j w_j v_j / sum_j a_j w_j,  w_j = rho'(||v_j - z||) / (2 ||v_j - z||)">
  </picture>
</p>

with guaranteed descent $\mathcal{L}(z^{(k+1)}) \le \mathcal{L}(z^{(k)})$. Outlier tokens with large residuals are automatically **down-weighted** (and fully removed beyond $\gamma$ for MCP).

### 🔧 ProAttention in 4 lines (MCP)

```python
D = torch.cdist(Z, V)                         # pairwise distance
W = torch.clip(1 / D - 1 / gamma, min=0)      # MCP weights
W = normalize(W * A, p=1, dim=-1)             # re-weight attention
Z = torch.matmul(W, V)                        # update
```

<details>
<summary><b>📜 Full implementation (<code>protransformers/pro_attention.py</code>)</b></summary>

```python
class ProAttention(nn.Module):
    def __init__(self, L=3, norm="L2", epsilon=1e-2, gamma=4.0, t=1.0, delta=4.0):
        super().__init__()
        self.L, self.norm, self.epsilon = L, norm, epsilon
        self.gamma, self.t, self.delta = gamma, t, delta

    def forward(self, A, V):
        M = torch.matmul(A, V)                      # vanilla attention output
        if self.norm == "L2":
            return M

        for _ in range(self.L):                     # Newton-IRLS iterations
            dist = torch.cdist(M, V)

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

            ww_norm = nn.functional.normalize(w * A, p=1, dim=-1)
            M = (1.0 - self.t) * M + self.t * torch.matmul(ww_norm, V)

        return M
```

</details>

---

## 🧩 Model Zoo

Every model below is a drop-in replacement for its 🤗 Transformers counterpart and loads the **original pretrained weights**. ProAttention is **off by default** (`norm="L2"` = vanilla attention) and switched on with one call.

| Domain | ProTransformer | Base class in `protransformers` | Patched attention |
|:--|:--|:--|:--|
| 📝 Text | **Pro-BERT** | `BertForSequenceClassification` | `BertSelfAttention` |
| 📝 Text | **Pro-RoBERTa** | `RobertaForSequenceClassification` | `RobertaSelfAttention` |
| 📝 Text | **Pro-ALBERT** | `AlbertForSequenceClassification` | `AlbertAttention` |
| 📝 Text | **Pro-DistilBERT** | `DistilBertForSequenceClassification` | `MultiHeadSelfAttention` |
| 🖼️ Vision | **Pro-ViT** | `ViTForImageClassification` | `ViTSelfAttention` |
| 🤖 LLM | **Pro-LLaMA** · **Pro-Vicuna** | `LlamaForCausalLM` | `LlamaAttention` (eager) |
| 🤖 LLM | **Pro-T5** | `T5ForConditionalGeneration` | `T5Attention` |
| 🕸️ Graph | **Pro-GAT** | `GAT` in `graph_attack.py` (dense) | `GATLayer` |

---

## 🚀 Quick Start

### 0️⃣ Installation

```bash
git clone https://github.com/chris-hzc/ProTransformer.git
cd ProTransformer

conda create -n protransformer python=3.10 -y
conda activate protransformer
pip install -r requirements.txt
```

> [!NOTE]
> `./protransformers` is a modified copy of 🤗 Transformers (v4.40) that contains ProAttention. It lives side by side with the pip-installed `transformers` (still used for tokenizers and by TextAttack). Run the scripts from the repo root so that `protransformers` is importable.

### 1️⃣ Plug ProAttention into any model in 3 lines

```python
from protransformers import AutoModelForSequenceClassification, set_pro_attention

model = AutoModelForSequenceClassification.from_pretrained("textattack/bert-base-uncased-ag-news")
set_pro_attention(model, norm="MCP", gamma=4.0, L=3)   # 🛡️ now it's a ProTransformer — no training needed
# set_pro_attention(model, norm="L2")                   # ↩️ back to vanilla attention
```

> [!TIP]
> For LLMs, load with `attn_implementation="eager"`: ProAttention needs the explicit attention matrix, so SDPA / FlashAttention are not supported.

<details open>
<summary><b>⚙️ ProAttention hyperparameters</b> (shared by all scripts)</summary>

<br>

| Argument | Description |
|:--|:--|
| `--norm` | Robust penalty ρ: `L2` (vanilla), `L1`, `Huber`, `MCP`, `HuberMCP` |
| `--L` | Number of Newton-IRLS iterations *K* (default `3`) |
| `--gamma` | MCP threshold γ — residuals beyond γ are fully down-weighted |
| `--delta` | Huber threshold δ |
| `--epsilon` | Numerical stability constant (default `1e-2`) |

</details>

### 2️⃣ Pro-BERT family under classic text attacks

```bash
# Pro-BERT (MCP) under TextFooler on AG News
python text_attack.py --backbone bert --norm MCP --gamma 4.0 --L 3 --data ag-news --attack tf

# Vanilla BERT baseline
python text_attack.py --backbone bert --norm L2 --data ag-news --attack tf

# Other backbones: roberta | albert | distilbert
python text_attack.py --backbone albert --norm MCP --gamma 4.0 --attack tf
```

| Argument | Default | Description |
|:--|:--:|:--|
| `--backbone` | `bert` | `bert` · `roberta` · `albert` · `distilbert` (pretrained [TextAttack](https://huggingface.co/textattack) checkpoints) |
| `--data` | `ag-news` | Dataset, e.g. `ag-news`, `imdb` |
| `--attack` | `tf` | `tf` TextFooler · `tb` TextBugger · `dwb` DeepWordBug · `pwws` PWWS · `ba` BERT-Attack |
| `--num_example` | `50` | Number of examples to attack |
| `--max_length` | `64` | Max tokenized input length |

### 3️⃣ Pro-ViT under FGSM / PGD

```bash
# Vanilla ViT vs. Pro-ViT on CIFAR-10, PGD (7 steps, α = 2/255) with budgets 1/255, 4/255, 8/255
python image_attack.py --norm L2
python image_attack.py --norm MCP --gamma 4.0

# FGSM, custom budgets
python image_attack.py --norm MCP --attack fgsm --budgets 1 2 4 8
```

Uses [`nateraw/vit-base-patch16-224-cifar10`](https://huggingface.co/nateraw/vit-base-patch16-224-cifar10) by default (`--model` to change); CIFAR-10 is downloaded to `--data_dir`.

### 4️⃣ Pro-GAT under adaptive PGD topology attack

A GAT is trained once per split on the clean graph; ProAttention is then plugged into its neighbor aggregation **without retraining**, and both models are attacked adaptively with the PGD topology attack of [Xu et al. (2019)](https://arxiv.org/abs/1906.04214) (global evasion, budget = fraction of edges flipped).

```bash
# GAT vs. Pro-GAT on Cora, 5 splits, budgets 5% / 10% / 20%
python graph_attack.py --data cora --norm MCP --gamma 4.0 --L 3 --budgets 0.05 0.1 0.2

# Citeseer, larger budgets
python graph_attack.py --data citeseer --norm MCP --gamma 4.0 --budgets 0.1 0.2 0.3 0.4
```

Graphs and 10/10/80 splits follow [Mujkanovic et al. (2022)](https://github.com/LoadingByte/are-gnn-defenses-robust); the `.npz` files are downloaded to `--data_dir` automatically.

### 5️⃣ Chat with Pro-Vicuna / Pro-LLaMA / Pro-T5

Prints the vanilla and the ProTransformer answer side by side:

```bash
# Pro-Vicuna
python llm_chat.py --model lmsys/vicuna-7b-v1.5 --norm Huber --delta 0.1 \
                   --prompt "Give me two tips for staying focused while studying."

# Pro-LLaMA (interactive session)
python llm_chat.py --model meta-llama/Llama-2-7b-chat-hf --norm MCP --gamma 4.0

# Pro-T5
python llm_chat.py --model google/flan-t5-large --norm MCP --gamma 4.0
```

### 6️⃣ Pro-Vicuna / Pro-LLaMA under jailbreak attacks

Evaluates the attack success rate (ASR ↓) on [AdvBench](https://github.com/llm-attacks/llm-attacks) harmful behaviors with refusal-prefix matching. Bring your own adversarial suffixes (e.g. [GCG](https://github.com/llm-attacks/llm-attacks) suffixes transferred from a surrogate model), one per line:

```bash
# Vanilla Vicuna
python llm_jailbreak.py --model lmsys/vicuna-7b-v1.5 --norm L2 --suffix_file suffixes.txt

# Pro-Vicuna (Huber, δ = 0.1)
python llm_jailbreak.py --model lmsys/vicuna-7b-v1.5 --norm Huber --delta 0.1 \
                        --suffix_file suffixes.txt --output pro_vicuna.csv
```

| Argument | Default | Description |
|:--|:--:|:--|
| `--model` | `lmsys/vicuna-7b-v1.5` | Any LLaMA-architecture chat model |
| `--behaviors` | AdvBench | CSV with a `goal` column |
| `--num_behaviors` | `100` | Number of behaviors to evaluate |
| `--suffix` / `--suffix_file` | – | Adversarial suffix(es) appended to every behavior |
| `--output` | – | Save all responses to CSV |

---

## 📊 Experimental Results

<p align="center">
  <img src="./figures/results.png" width="100%" />
</p>

Pro-BERT (MCP) is competitive with adversarial-training defenses (FreeLB, PGD, MixADA, TA-VAT) **without any training**, and combining it with adversarial training (**Pro-BERT (MCP) + AT**) sets the best results across all four attacks. See the [paper](https://arxiv.org/abs/2410.23182) for results on LLMs (T5, LLaMA, Vicuna), jailbreaks, ViT, and GAT.

---

## 📁 Repository Structure

```
ProTransformer/
├── text_attack.py          # 2️⃣ Classic text attacks on Pro-BERT / RoBERTa / ALBERT / DistilBERT
├── image_attack.py         # 3️⃣ FGSM / PGD on Pro-ViT
├── graph_attack.py         # 4️⃣ Adaptive PGD topology attack on Pro-GAT
├── llm_chat.py             # 5️⃣ Chat with Pro-Vicuna / Pro-LLaMA / Pro-T5
├── llm_jailbreak.py        # 6️⃣ Jailbreak ASR of Pro-Vicuna / Pro-LLaMA
├── requirements.txt
├── protransformers/        # Modified 🤗 Transformers (v4.40) with ProAttention
│   ├── pro_attention.py    #   ← ProAttention + set_pro_attention
│   └── models/{bert,roberta,albert,distilbert,vit,llama,t5}/   # patched attention layers
└── figures/
```

---

## 📖 Citation

If you find ProTransformer useful in your research, please consider citing our paper and giving this repo a ⭐:

```bibtex
@article{hou2024protransformer,
  title={Protransformer: Robustify transformers via plug-and-play paradigm},
  author={Hou, Zhichao and Gao, Weizhi and Shen, Yuchen and Wang, Feiyi and Liu, Xiaorui},
  journal={Advances in Neural Information Processing Systems},
  volume={37},
  pages={137557--137609},
  year={2024}
}
```

## 🙏 Acknowledgements

This codebase builds upon [🤗 Transformers](https://github.com/huggingface/transformers) and [TextAttack](https://github.com/QData/TextAttack). We thank the authors for their great work.

## 📬 Contact

For questions, please open an [issue](https://github.com/chris-hzc/ProTransformer/issues) or contact Zhichao Hou ([zhou4@ncsu.edu](mailto:zhou4@ncsu.edu)).

<div align="center">

⭐ **If this project helps you, please star it!** ⭐

</div>
