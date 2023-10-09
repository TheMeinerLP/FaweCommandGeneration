# //replace ACACIA_FENCE BAMBOO, "I want to change all ACACIA_FENCE blocks to BAMBOO. What should I type?"
from pathlib import Path


if __name__ == '__main__':
    file_path = Path('materials.txt')
    materials = file_path.read_text().split('\n')
    for old_block in materials:
        for new_block in materials:
            if old_block == new_block:
                continue
            # print(F'//replace {old_block} {new_block}, "I want to change all {old_block} blocks to {new_block}. What should I type?"')
            print(F'//replace {old_block} {new_block}, "Replace all {old_block} blocks in the selection with {new_block} blocks."')

# python3 material-generation.py > material-prompts.csv