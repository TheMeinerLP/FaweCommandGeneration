from pathlib import Path
import pandas as pd
import os
from transformers import AutoTokenizer, AutoModelForCausalLM


def generate_blocks():
    file_path = Path('materials.txt')
    materials = file_path.read_text().split('\n')
    datas = []
    for old_block in materials:
        for new_block in materials:
            if old_block == new_block:
                continue
            new_data = {
                'command': F'//replace {old_block} {new_block}',
                'question': F"Replace all {old_block} blocks in the selection with {new_block} blocks."
            }
            datas.append(new_data)
    frame = pd.DataFrame.from_records(datas)
    frame.to_csv('blocks_train_raw.csv', index=False, header=True)


def cleanup_data():
    frame = pd.read_csv('blocks_train_raw.csv')
    sys_prompt = "A chat."
    tokenizer = AutoTokenizer.from_pretrained("Open-Orca/Mistral-7B-OpenOrca", max_new_tokens=512)
    model = AutoModelForCausalLM.from_pretrained("Open-Orca/Mistral-7B-OpenOrca", trust_remote_code=True,
                                                 device_map='auto')
    datas = []
    for index, row in frame.head(5).iterrows():
        question = row['question']
        prompt = F'Improve following question that looks more like a player question and replace caps locked words with more heterogenous one: {question}'
        prefix = "<|im_start|>"
        suffix = "<|im_end|>\n"
        sys_format = prefix + "system\n" + sys_prompt + suffix
        user_format = prefix + "user\n" + prompt + suffix
        assistant_format = prefix + "assistant\n"
        input_text = sys_format + user_format + assistant_format
        inputs = tokenizer(input_text, return_tensors="pt").to("cuda")
        outputs = model.generate(**inputs, max_new_tokens=512, pad_token_id=tokenizer.eos_token_id)
        ai_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        splitted_ai_text = ai_text.split('\n')
        last_line = splitted_ai_text.pop()
        new_data = {
            'command': row['command'],
            'question': last_line.strip()
        }
        datas.append(new_data)
    frame = pd.DataFrame.from_records(datas)
    frame.to_csv('blocks_train.csv', index=False, header=True)

script_path = os.path.abspath(__file__)
os.chdir(os.path.dirname(script_path))
# generate_blocks()
cleanup_data()
