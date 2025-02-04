import random
from typing import Iterable

import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache
import torch

# Model name from the HuggingFace hub
# model_name = "deepseek-ai/DeepSeek-R1" # Requires over 700 GBs
# model_name = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B" # Too much memory?
model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"


print("Loading model...")
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name, torch_dtype=torch.bfloat16, device_map="cuda"
)
print("... model loaded")

_, _start_think_token, end_think_token = tokenizer.encode("<think></think>")
min_thinking_tokens = 128
replacements = ["\nWait, but", "\nHmm", "\nSo"]


@torch.inference_mode
def _reasoning_effort(question: str, min_thinking_tokens: int) -> Iterable[str]:
    tokens = tokenizer.apply_chat_template(
        [
            {"role": "user", "content": question},
            {"role": "assistant", "content": "<think>\n"},
        ],
        continue_final_message=True,
        return_tensors="pt",
    )
    tokens = tokens.to(model.device)
    kv = DynamicCache()
    n_thinking_tokens = 0

    yield tokenizer.decode(list(tokens[0]))
    while True:
        out = model(input_ids=tokens, past_key_values=kv, use_cache=True)
        next_token = torch.multinomial(
            torch.softmax(out.logits[0, -1, :], dim=-1), 1
        ).item()
        kv = out.past_key_values

        if (
            next_token in (end_think_token, model.config.eos_token_id)
            and n_thinking_tokens < min_thinking_tokens
        ):
            replacement = random.choice(replacements)
            yield replacement
            replacement_tokens = tokenizer.encode(replacement)
            n_thinking_tokens += len(replacement_tokens)
            tokens = torch.tensor([replacement_tokens]).to(tokens.device)
        elif next_token == model.config.eos_token_id:
            break
        else:
            yield tokenizer.decode([next_token])
            n_thinking_tokens += 1
            tokens = torch.tensor([[next_token]]).to(tokens.device)


def process_prompt(prompt: str) -> Iterable[str]:
    print(f"\n-- Processing prompt: '{prompt}' --\n\n")
    for chunk in _reasoning_effort(prompt, min_thinking_tokens):
        yield chunk
    print(f"\n\n-- Finished processing prompt: '{prompt}' --\n\n")
