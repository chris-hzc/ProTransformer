"""Chat with Pro-Vicuna / Pro-LLaMA / Pro-T5, side by side with the vanilla model.

    python llm_chat.py --model lmsys/vicuna-7b-v1.5 --norm Huber --delta 0.1 --prompt "What is a robust estimator?"
    python llm_chat.py --model meta-llama/Llama-2-7b-chat-hf --norm MCP --gamma 4.0
    python llm_chat.py --model google/flan-t5-large --norm MCP --gamma 4.0
"""

import argparse

import torch
import transformers

from protransformers import AutoConfig, AutoModelForCausalLM, AutoModelForSeq2SeqLM, set_pro_attention


VICUNA_SYSTEM = (
    "A chat between a curious user and an artificial intelligence assistant. "
    "The assistant gives helpful, detailed, and polite answers to the user's questions."
)


def add_robust_args(parser):
    parser.add_argument("--model", type=str, default="lmsys/vicuna-7b-v1.5")
    parser.add_argument("--norm", type=str, default="Huber", choices=["L2", "L1", "Huber", "MCP", "HuberMCP"])
    parser.add_argument("--gamma", type=float, default=4.0)
    parser.add_argument("--delta", type=float, default=0.1)
    parser.add_argument("--epsilon", type=float, default=1e-2)
    parser.add_argument("--L", type=int, default=3)
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--dtype", type=str, default="float16", choices=["float16", "bfloat16", "float32"])


def load_llm(name, dtype="float16"):
    """Load a causal LM (LLaMA / Vicuna) or a seq2seq LM (T5) with eager attention, as ProAttention requires."""
    config = AutoConfig.from_pretrained(name)
    model_class = AutoModelForSeq2SeqLM if config.is_encoder_decoder else AutoModelForCausalLM
    model = model_class.from_pretrained(
        name, torch_dtype=getattr(torch, dtype), attn_implementation="eager", device_map="auto"
    ).eval()

    tokenizer = transformers.AutoTokenizer.from_pretrained(name)  # tokenizers are untouched by ProTransformer
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.unk_token or tokenizer.eos_token
    return model, tokenizer


def build_prompt(tokenizer, model, user_message):
    if model.config.is_encoder_decoder:
        return user_message
    if tokenizer.chat_template is not None:
        messages = [{"role": "user", "content": user_message}]
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    # Vicuna v1.1+ template
    return f"{VICUNA_SYSTEM} USER: {user_message} ASSISTANT:"


@torch.no_grad()
def generate(model, tokenizer, prompts, max_new_tokens=128):
    inputs = tokenizer(prompts, return_tensors="pt", padding=True, add_special_tokens=model.config.is_encoder_decoder)
    inputs = inputs.to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False, pad_token_id=tokenizer.pad_token_id)
    if not model.config.is_encoder_decoder:
        outputs = outputs[:, inputs["input_ids"].shape[1] :]
    return tokenizer.batch_decode(outputs, skip_special_tokens=True)


def main():
    parser = argparse.ArgumentParser(description="Chat with a ProTransformer LLM")
    add_robust_args(parser)
    parser.add_argument("--prompt", type=str, default=None, help="Single prompt; omit for an interactive session")
    args = parser.parse_args()
    print(args)

    model, tokenizer = load_llm(args.model, args.dtype)
    robust = dict(norm=args.norm, L=args.L, gamma=args.gamma, delta=args.delta, epsilon=args.epsilon)

    def answer(message):
        prompt = build_prompt(tokenizer, model, message)
        for name, params in [("Vanilla", dict(norm="L2")), (f"Pro ({args.norm})", robust)]:
            set_pro_attention(model, **params)
            print(f"\n\033[1m[{name}]\033[0m {generate(model, tokenizer, [prompt], args.max_new_tokens)[0].strip()}")

    if args.prompt is not None:
        answer(args.prompt)
        return

    while True:
        try:
            message = input("\n\033[1mUser:\033[0m ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if message:
            answer(message)


if __name__ == "__main__":
    main()
