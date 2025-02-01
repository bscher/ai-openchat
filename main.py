#!/usr/bin/env python3

import src

def patch():
    # For backwards compatibility since some remote code on Hub still rely on these variables.
    from packaging import version

    torch = src.prompter.torch
    pytorch_utils = src.prompter.transformers.pytorch_utils
    parsed_torch_version_base = version.parse(version.parse(torch.__version__).base_version)

    pytorch_utils.is_torch_greater_or_equal_than_2_0 = \
        parsed_torch_version_base >= version.parse("2.0")
    pytorch_utils.is_torch_greater_or_equal_than_1_13 = \
        parsed_torch_version_base >= version.parse("1.13")
    pytorch_utils.is_torch_greater_or_equal_than_1_12 = \
        parsed_torch_version_base >= version.parse("1.12")
    
def user_chat_exchange():
    patch()

    while True:
        user_prompt = input("Prompt: ")
        if not user_prompt:
            return
        print("\n\n")
        for prompt_part in src.prompter.process_prompt(user_prompt):
            print(prompt_part, end="", flush=True)

def main():
    user_chat_exchange()

if __name__ == '__main__':
    exit(main())