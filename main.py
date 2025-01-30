#!/usr/bin/env python3

import src

def patch():
    from packaging import version

    # For backwards compatibility since some remote code on Hub still rely on these variables.
    torch = src.prompter.torch
    pytorch_utils = src.prompter.transformers.pytorch_utils
    parsed_torch_version_base = version.parse(version.parse(torch.__version__).base_version)

    pytorch_utils.is_torch_greater_or_equal_than_2_0 = \
        parsed_torch_version_base >= version.parse("2.0")
    pytorch_utils.is_torch_greater_or_equal_than_1_13 = \
        parsed_torch_version_base >= version.parse("1.13")
    pytorch_utils.is_torch_greater_or_equal_than_1_12 = \
        parsed_torch_version_base >= version.parse("1.12")
    
def main():
    patch()
    print(src.prompter.generate_text("Once upon a time, upon the hilltops of "))


if __name__ == '__main__':
    exit(main())