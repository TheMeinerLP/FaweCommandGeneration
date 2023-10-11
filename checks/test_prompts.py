import json
from pathlib import Path
from transformers import pipeline

script_config = json.loads(Path("../config/script_config.json").read_text("utf-8"))
pipe = pipeline("text2text-generation", model=F"{script_config['output_dir']}", device=0)

print(pipe("make a curve deformation with a intensity of 61")[0]['generated_text'])
print(pipe("Make my structure 55 times smaller")[0]['generated_text'])
print(pipe("Kill all monsters in radius of 100 blocks")[0]['generated_text'])