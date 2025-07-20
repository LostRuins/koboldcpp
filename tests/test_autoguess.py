"""
Test that the AutoGuess feature picks the correct model for every template.
Also checks that every template is being tested so that when new AutoGuess additions are made, this test fails unless an accompanying test is included.
"""
import os
import sys
import requests
import json


# Map an AutoGuess name to a HuggingFace model ID
# THIS LIST MUST BE UPDATED WHEN A NEW MODEL IS ADDED
AUTOGUESS_MAPPING = {
    "ChatML (Phi 4)": "microsoft/phi-4",
    "ChatML (Qwen 2.5 based)": "Qwen/Qwen2.5-0.5B-Instruct",
    "ChatML (Kimi)": "moonshotai/Kimi-K2-Instruct",
    "Google Gemma 2": "Efficient-Large-Model/gemma-2-2b-it",
    "Google Gemma 3": "scb10x/typhoon2.1-gemma3-12b",
    "Google Gemma 3n": "lmstudio-community/gemma-3n-E4B-it-MLX-bf16",
    "Llama 3.x": "Steelskull/L3.3-Shakudo-70b",
    "Llama 4": "meta-llama/Llama-4-Scout-17B-16E-Instruct",
    "Mistral V7 (with system prompt)": "Doctor-Shotgun/MS3.2-24B-Magnum-Diamond",
    "Mistral V3": "mistralai/Mistral-7B-Instruct-v0.3",
    "GLM-4": "THUDM/glm-4-9b-chat-hf",
    "Phi 3.5": "microsoft/Phi-3.5-mini-instruct",
    "Phi 4 (mini)": "microsoft/Phi-4-mini-instruct",
    "Cohere (Aya Expanse 32B based)": "CohereLabs/aya-expanse-32b",
    "DeepSeek V2.5": "deepseek-ai/DeepSeek-V2.5",
    "Jamba": "ai21labs/Jamba-tiny-dev",
    "Dots": "rednote-hilab/dots.llm1.inst",
    "RWKV World": "fla-hub/rwkv7-1.5B-world",
    "Mistral (Generic)": "mistralai/Mistral-Nemo-Instruct-2407",
    "ChatML (Generic)": "NewEden/Gemma-27B-chatml",
}

# User may be running this test from ./ or from ../ -- we want to be in ./ (i.e. tests)
if os.path.exists("tests"):
    os.chdir("tests")

with open("../kcpp_adapters/AutoGuess.json") as f:
    autoguess = json.load(f)

def get_tokenizer_config_for_huggingface_model_id(huggingface_model_id: str):
    fname = f"gated-tokenizers/tokenizer_configs/{huggingface_model_id.replace('/','_')}.json"
    if os.path.exists(fname):
        with open(fname) as f:
            return json.load(f)

    for filename in ["tokenizer_config.json", "chat_template.json"]:
        url = f"https://huggingface.co/{huggingface_model_id}/resolve/main/{filename}"
        response = requests.get(url)
        if response.status_code == 200:
            v = json.loads(response.text)
            if 'chat_template' in v:
                return v
    raise ValueError(f"Failed to fetch tokenizer config for {huggingface_model_id}.")

def match_chat_template_to_adapter(chat_template: str|list) -> tuple[str, str|None]|None:
    # Additional code in tester not present in application: support for multiple chat templates, and use default if present
    sub_template: str|None = None
    if isinstance(chat_template, list):
        found = False
        for template in chat_template:
            # {"name": .., "template": ...}
            if template['name'] == "default":
                sub_template = "default"
                chat_template = template['template']
                found = True
                break
        if not found:
            # We pick the first template if no default is present
            sub_template = chat_template[0]['name']
            chat_template = chat_template[0]['template']
    if chat_template != "":
        for entry in autoguess:
            if all(s in chat_template for s in entry['search']):
                return entry['name'], sub_template

failures = 0
seen = set()
namefmt = "{name:<" + str(max(len(name) for name in AUTOGUESS_MAPPING.keys())) + "}"
hmifmt = "{huggingface_model_id:<" + str(max(len(huggingface_model_id) for huggingface_model_id in AUTOGUESS_MAPPING.values())) + "}"
for name, huggingface_model_id in AUTOGUESS_MAPPING.items():
    seen.add(name)
    if huggingface_model_id == "***UNKNOWN***":
        print(namefmt.format(name=name) + " = " + namefmt.format(name="***UNKNOWN***") + " : PENDING")
        continue
    tokenizer_config = get_tokenizer_config_for_huggingface_model_id(huggingface_model_id)
    assert 'chat_template' in tokenizer_config
    matched = match_chat_template_to_adapter(tokenizer_config['chat_template'])
    if matched is None:
        matched, sub_template = "MISSING MAPPING", None
    else:
        matched, sub_template = matched
    sub_template = f"[{sub_template}]" if sub_template else ""
    print(namefmt.format(name=name) + " = " + namefmt.format(name=matched) + " : " + ("OK     " if name == matched else "FAILURE") + " " + hmifmt.format(huggingface_model_id=huggingface_model_id) + " " + sub_template)
    failures += name != matched

for entry in autoguess:
    if entry['name'] not in seen:
        print(namefmt.format(name=entry['name']) + "   MISSING MAPPING")
        failures += 1

if failures > 0:
    print(f"There were {failures} failure(s)!")
    sys.exit(1)
