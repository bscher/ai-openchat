from threading import Thread
from typing import Optional, Iterable
import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer

# Model name from the HuggingFace hub
#model_name = "deepseek-ai/DeepSeek-R1" # Requires over 700 GBs
#model_name = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B" # Too much memory?
model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"

def chat_llama3_8b(
        message: str, 
        history: Optional[list] = None, 
        temperature: float = 0.6, 
        max_new_tokens: int = 100
    ) -> Iterable[str]:
    """
    Generate a streaming response using the llama3-32B model.
    Args:
        message (str): The input message.
        temperature (float): The temperature for generating the response.
        max_new_tokens (int): The maximum number of new tokens to generate.
    Returns:
        Iterable[str]: The generated response, in parts.
    """
    # Load the tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map="cuda", trust_remote_code=True)
    terminators = [
        tokenizer.eos_token_id,
        tokenizer.convert_tokens_to_ids("<|eot_id|>")
    ]

    conversation = []
    for user, assistant in (history or []):
        conversation.extend([{"role": "user", "content": user}, {"role": "assistant", "content": assistant}])
    conversation.append({"role": "user", "content": message})

    input_ids = tokenizer.apply_chat_template(conversation, return_tensors="pt").to(model.device)
    
    streamer = TextIteratorStreamer(tokenizer, timeout=10.0, skip_prompt=True, skip_special_tokens=True)
        
    model.generate(
        input_ids=input_ids,
        streamer=streamer,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=temperature,
        eos_token_id=terminators
    )

    outputs = []
    for text in streamer:
        outputs.append(text)
        #print(outputs)
        yield "".join(outputs)
        
def generate_text(prompt: str) -> str:

    generated_text = "".join(output_part for output_part in chat_llama3_8b(prompt))

    print("-- Generated Text --")
    print(generated_text)
    print("--------------------")
