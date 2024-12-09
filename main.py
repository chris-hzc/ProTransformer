import argparse
import time
import textattack

import transformers
from textattack.models.wrappers import HuggingFaceModelWrapper
from textattack.datasets import HuggingFaceDataset
from textattack import Attacker
from textattack import AttackArgs


parser = argparse.ArgumentParser(description='Train classification network')

parser.add_argument('--norm',type=str, default='L2')
parser.add_argument('--gamma',type=float, default=4.0)
parser.add_argument('--epsilon',type=float, default=1e-2)
parser.add_argument('--delta',type=float, default=4.0)
parser.add_argument('--L',type=int, default=3)
parser.add_argument('--t',type=float, default=1.0)
parser.add_argument('--num_example',type=int, default=50)
parser.add_argument('--num_replace',type=int, default=50)
parser.add_argument('--attack',type=str, default='tf')
parser.add_argument('--max_length',type=int, default=64)
parser.add_argument('--data',type=str, default='ag-news')
parser.add_argument('--rho',type=float, default=1.0) #max percentange
parser.add_argument('--cos',type=float, default=0.5) #max percentange
parser.add_argument('--threshold',type=float, default=0.840845057)

parser.add_argument('--backbone',type=str, default='bert')


args = parser.parse_args()



def load_params(args, model):
        
    for layer in model.bert.encoder.layer:      
        layer.attention.self.robust_sum.L = args.L
        layer.attention.self.robust_sum.norm = args.norm
        layer.attention.self.robust_sum.gamma = args.gamma
        layer.attention.self.robust_sum.epsilon = args.epsilon
        layer.attention.self.robust_sum.t = args.t
        layer.attention.self.robust_sum.delta = args.delta
    
    return model


def main():
    print(args)
    model = transformers.models.bert.modeling_bert.BertForSequenceClassification.from_pretrained(f"textattack/{args.backbone}-base-uncased-{args.data}")
    model = load_params(args, model)
    tokenizer = transformers.AutoTokenizer.from_pretrained(f"textattack/{args.backbone}-base-uncased-{args.data}")
    tokenizer.model_max_length=args.max_length
    
    model_wrapper = HuggingFaceModelWrapper(model, tokenizer)
        
    dataset = HuggingFaceDataset("ag_news", None, "test")
    
    attack = textattack.attack_recipes.TextFoolerJin2019.build(model_wrapper)

    attack_args = AttackArgs(num_examples=args.num_example)

    attacker = Attacker(attack, dataset, attack_args)
    
    attack_results = attacker.attack_dataset()

    print("-----end------")



if __name__ == '__main__':
    main()
    
