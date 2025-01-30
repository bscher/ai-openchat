#!/usr/bin/env python3

import transformers
import transformers.models
    
def main():
    print(transformers.utils.hub.get_all_cached_files())


if __name__ == '__main__':
    exit(main())