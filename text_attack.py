"""Classic text attacks (TextFooler, TextBugger, DeepWordBug, PWWS, BERT-Attack) on Pro-BERT / Pro-RoBERTa /
Pro-ALBERT / Pro-DistilBERT.

    python text_attack.py --backbone bert --norm MCP --gamma 4.0 --L 3 --data ag-news --attack tf
"""

import argparse

import textattack
import transformers
from textattack import AttackArgs, Attacker
from textattack.datasets import HuggingFaceDataset
from textattack.models.wrappers import HuggingFaceModelWrapper

import protransformers
from protransformers import set_pro_attention


parser = argparse.ArgumentParser(description="ProTransformer under classic text attacks")

parser.add_argument("--norm", type=str, default="L2", choices=["L2", "L1", "Huber", "MCP", "HuberMCP"])
parser.add_argument("--gamma", type=float, default=4.0)
parser.add_argument("--epsilon", type=float, default=1e-2)
parser.add_argument("--delta", type=float, default=4.0)
parser.add_argument("--L", type=int, default=3)
parser.add_argument("--t", type=float, default=1.0)
parser.add_argument("--num_example", type=int, default=50)
parser.add_argument("--attack", type=str, default="tf", choices=["tf", "ba", "tb", "dwb", "pwws"])
parser.add_argument("--max_length", type=int, default=64)
parser.add_argument("--data", type=str, default="ag-news")
parser.add_argument("--backbone", type=str, default="bert", choices=["bert", "roberta", "distilbert", "albert"])

args = parser.parse_args()


# Pretrained TextAttack checkpoints on the HuggingFace Hub
CHECKPOINTS = {
    "bert": ("BertForSequenceClassification", "textattack/bert-base-uncased-{}"),
    "roberta": ("RobertaForSequenceClassification", "textattack/roberta-base-{}"),
    "distilbert": ("DistilBertForSequenceClassification", "textattack/distilbert-base-uncased-{}"),
    "albert": ("AlbertForSequenceClassification", "textattack/albert-base-v2-{}"),
}


class ProModelWrapper(HuggingFaceModelWrapper):
    """TextAttack's HuggingFace wrapper, minus the type check against the pip-installed `transformers`."""

    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer


def get_model_and_tokenizer():
    model_class, checkpoint = CHECKPOINTS[args.backbone]
    checkpoint = checkpoint.format(args.data)
    model = getattr(protransformers, model_class).from_pretrained(checkpoint)
    tokenizer = transformers.AutoTokenizer.from_pretrained(checkpoint)  # tokenizers are untouched by ProTransformer
    tokenizer.model_max_length = args.max_length
    return model, tokenizer


def get_dataset():
    if args.data == "ag-news":
        data_name = "ag_news"
    elif args.data == "mnli":
        data_name = "multi_nli"
    else:
        data_name = args.data

    if args.data == "mnli":
        dataset = HuggingFaceDataset(data_name, None, "validation_mismatched")
    elif args.data in ["sms_spam"]:
        dataset = HuggingFaceDataset(data_name, None, "train")
    elif args.data in ["rte", "cola"]:
        dataset = HuggingFaceDataset("glue", data_name, "validation")
    else:
        dataset = HuggingFaceDataset(data_name, None, "test")
    return dataset


def get_attack(model_wrapper):
    recipes = {
        "tf": textattack.attack_recipes.TextFoolerJin2019,
        "ba": textattack.attack_recipes.BERTAttackLi2020,
        "tb": textattack.attack_recipes.TextBuggerLi2018,
        "dwb": textattack.attack_recipes.DeepWordBugGao2018,
        "pwws": textattack.attack_recipes.PWWSRen2019,
    }
    return recipes[args.attack].build(model_wrapper)


def main():
    print(args)

    model, tokenizer = get_model_and_tokenizer()
    set_pro_attention(model, norm=args.norm, L=args.L, gamma=args.gamma, epsilon=args.epsilon, delta=args.delta, t=args.t)

    model_wrapper = ProModelWrapper(model, tokenizer)
    dataset = get_dataset()
    attack = get_attack(model_wrapper)

    attacker = Attacker(attack, dataset, AttackArgs(num_examples=args.num_example))
    attacker.attack_dataset()

    print("-----end------")


if __name__ == "__main__":
    main()
