import argparse
import time
import textattack
import torch.nn as nn

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

def load_params(args, model):
    if args.backbone == 'bert':
        layers = model.bert.encoder.layer
    elif args.backbone == 'roberta':
        layers = model.roberta.encoder.layer
    elif args.backbone == 'distilbert':
        layers = model.distilbert.transformer.layer
    elif args.backbone == 'albert':
        layers = model.albert.encoder.albert_layer_groups[0].albert_layers
        
    for layer in layers:
        
        if args.backbone in ['distilbert','albert']:
            layer.attention.robust_sum.L = args.L
            layer.attention.robust_sum.norm = args.norm
            layer.attention.robust_sum.gamma = args.gamma
            layer.attention.robust_sum.epsilon = args.epsilon
            layer.attention.robust_sum.delta = args.delta
        else:        
            layer.attention.self.robust_sum.L = args.L
            layer.attention.self.robust_sum.norm = args.norm
            layer.attention.self.robust_sum.gamma = args.gamma
            layer.attention.self.robust_sum.epsilon = args.epsilon
            layer.attention.self.robust_sum.t = args.t
            layer.attention.self.robust_sum.delta = args.delta
    
    return model


def get_model():
    if args.backbone == 'bert':
        model = transformers.models.bert.modeling_bert.BertForSequenceClassification.from_pretrained(f"textattack/{args.backbone}-base-uncased-{args.data}")
    elif args.backbone == 'roberta':
        model = transformers.models.roberta.modeling_roberta.RobertaForSequenceClassification.from_pretrained(f"textattack/{args.backbone}-base-{args.data}")
    elif args.backbone == 'distilbert':
        model = transformers.models.distilbert.modeling_distilbert.DistilBertForSequenceClassification.from_pretrained(f"textattack/distilbert-base-uncased-{args.data}")
    elif args.backbone == 'albert':
        model = transformers.models.albert.modeling_albert.AlbertForSequenceClassification.from_pretrained(f"textattack/albert-base-v2-{args.data}")
    return model


def get_tokenizer():
    if args.backbone == 'bert':
        tokenizer = transformers.AutoTokenizer.from_pretrained(f"textattack/{args.backbone}-base-uncased-{args.data}")
    elif args.backbone == 'roberta':
        tokenizer = transformers.AutoTokenizer.from_pretrained(f"textattack/{args.backbone}-base-{args.data}")
    elif args.backbone == 'distilbert':
        tokenizer = transformers.AutoTokenizer.from_pretrained(f"textattack/{args.backbone}-base-uncased-{args.data}")
    elif args.backbone == 'albert':
        tokenizer = transformers.AutoTokenizer.from_pretrained(f"textattack/albert-base-v2-{args.data}")
    return tokenizer


def get_dataset():
    if args.data=='ag-news':
        data_name = 'ag_news'
    elif args.data =='mnli':
        data_name = 'multi_nli'
    else:
        data_name =args.data
        
    if args.data == 'mnli':
        dataset = HuggingFaceDataset(data_name, None, "validation_mismatched")
    elif args.data in ['sms_spam']:
        dataset = HuggingFaceDataset(data_name, None, "train")
    elif args.data in ['rte','cola']:
        dataset = HuggingFaceDataset('glue', data_name, "validation")
    else:
        dataset = HuggingFaceDataset(data_name, None, "test")
    return dataset


def get_attack(model_wrapper):
    if args.attack == 'tf':
        attack = textattack.attack_recipes.TextFoolerJin2019.build(model_wrapper)
    elif args.attack == 'ba':
        attack = textattack.attack_recipes.BERTAttackLi2020.build(model_wrapper)
    elif args.attack == 'tb':
        attack = textattack.attack_recipes.TextBuggerLi2018.build(model_wrapper)
    elif args.attack == 'dwb':
        attack = textattack.attack_recipes.DeepWordBugGao2018.build(model_wrapper)
    elif args.attack == 'pwws':
        attack = textattack.attack_recipes.PWWSRen2019.build(model_wrapper)
    return attack

def main():
    print(args)

    model = get_model()
    model = load_params(args, model)
    tokenizer = get_tokenizer()
    tokenizer.model_max_length=args.max_length
    
    model_wrapper = HuggingFaceModelWrapper(model, tokenizer)
        
    dataset = get_dataset()
    
    attack = get_attack(model_wrapper)

    attack_args = AttackArgs(num_examples=args.num_example)

    attacker = Attacker(attack, dataset, attack_args)
    
    attack_results = attacker.attack_dataset()

    print("-----end------")



if __name__ == '__main__':
    main()
    
