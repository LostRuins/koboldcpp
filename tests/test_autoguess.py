"""
Test that the AutoGuess feature picks the correct model for every template.
Also checks that every template is being tested so that when new AutoGuess additions are made, this test fails unless an accompanying test is included.
"""
import os
import sys
import jinja2
import requests
import json
from transformers import AutoTokenizer


# Map an AutoGuess name to a HuggingFace model ID
# THIS LIST MUST BE UPDATED WHEN A NEW MODEL IS ADDED
AUTOGUESS_MAPPING = {
    "ChatML (Phi 4)": "microsoft/phi-4",
    "ChatML (Qwen 2.5 based)": "Qwen/Qwen2.5-0.5B-Instruct",
    "ChatML (Kimi)": "moonshotai/Kimi-K2-Instruct",
    "Google Gemma 2": "Efficient-Large-Model/gemma-2-2b-it",
    "Google Gemma 3": "google/gemma-3-4b-it",
    "Google Gemma 3n": "lmstudio-community/gemma-3n-E4B-it-MLX-bf16",
    "Llama 3.x": "Steelskull/L3.3-Shakudo-70b",
    "Llama 4": "nvidia/Llama-4-Scout-17B-16E-Instruct-FP8",
    "Mistral Tekken": "Doctor-Shotgun/MS3.2-24B-Magnum-Diamond",
    "Mistral Non-Tekken": "mistralai/Mistral-7B-Instruct-v0.3",
    "GLM-4": "THUDM/glm-4-9b-chat-hf",
    "Phi 3.5": "microsoft/Phi-3.5-mini-instruct",
    "Phi 4 (mini)": "microsoft/Phi-4-mini-instruct",
    "Cohere (Aya Expanse 32B based)": "CohereLabs/aya-expanse-32b",
    "DeepSeek V2.5": "deepseek-ai/DeepSeek-V2.5",
    "Jamba": "ai21labs/Jamba-tiny-dev",
    "Dots": "rednote-hilab/dots.llm1.inst",
    "RWKV World": "fla-hub/rwkv7-1.5B-world",
    "OpenAI Harmony": "openai/gpt-oss-120b",
    "Mistral (Generic)": "mistralai/Mistral-Nemo-Instruct-2407",
    "ChatML (Generic)": "NewEden/Gemma-27B-chatml",
}

AUTOGUESS_SKIP_ADAPTER_TESTS = {
    "Mistral Non-Tekken": {"system"},   # Poor system support
    "Mistral (Generic)": {"system"},    # Poor system support
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

    fname = f"gated-tokenizers/tokenizer_configs/{huggingface_model_id.replace('/','_')}/tokenizer_config.json"
    if os.path.exists(fname):
        with open(fname) as f:
            return json.load(f)

    for filename in ["tokenizer_config.json", "chat_template.json", "chat_template.jinja"]:
        url = f"https://huggingface.co/{huggingface_model_id}/resolve/main/{filename}"
        response = requests.get(url)
        if response.status_code == 200:
            if url.endswith(".jinja"):
                return {"chat_template": response.text}
            v = json.loads(response.text)
            if 'chat_template' in v:
                return v
    raise ValueError(f"Failed to fetch tokenizer config for {huggingface_model_id}.")

def get_tokenizer_for_huggingface_model_id(huggingface_model_id: str):
    dname = f"gated-tokenizers/tokenizer_configs/{huggingface_model_id.replace('/','_')}"
    if os.path.exists(dname):
        return AutoTokenizer.from_pretrained(dname, trust_remote_code=True)
    return AutoTokenizer.from_pretrained(huggingface_model_id, trust_remote_code=True)

def match_chat_template_to_adapter(chat_template: str|list) -> tuple[dict, str|None]|None:
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
                return entry, sub_template

def test_tokenizer_with_adapter(tokenizer, adapter: dict[str, str], skip: set) -> tuple[bool, str|None]:
    """
    See if the adapter correctly reflects the tokenizer chat template.
    """
    def adapter_wrap(role, content):
        return adapter[f"{role}_start"] + content + adapter[f"{role}_end"]
    def system(content):        return adapter_wrap("system", content)
    def user(content):          return adapter_wrap("user", content)
    def assistant(content):     return adapter_wrap("assistant", content)
    def templ(rolelist):
        return tokenizer.apply_chat_template(rolelist, tokenize=False)

    try:
        # We skip system checks if user and system are identical, or if in skip
        if "system" not in skip and user("x") != system("x"):
            # Test system
            expect = system("SyS-tEm")
            templated = templ([{"role": "system", "content": "SyS-tEm"}, {"role": "user", "content": "user"}])
            if expect not in templated:
                return False, f"system role missing expected fragment\n\tadapter:  {expect.replace("\n", "\\n")}\n\ttokenizer: {templated.replace("\n", "\\n")}"

        # Test user/asst/user
        expect = [
            user("user_1"),
            assistant("asst_1"),
            user("user_2")
        ]
        templated = templ([
            {"role":"user", "content": "user_1"},
            {"role":"assistant", "content": "asst_1"},
            {"role":"user", "content": "user_2"},
        ])
        rem = templated
        for sub in expect:
            if sub not in rem:
                return False, f"missing expected fragment\n\tadapter:  {sub.replace("\n", "\\n")}\n\ttokenizer: {rem.replace("\n", "\\n")}"
            rem = rem.split(sub, 1)[1]
    except jinja2.exceptions.TemplateError as e:
        return False, f"template error: {e}"
    return True, None

filter = sys.argv[1] if len(sys.argv) > 1 else None

failures = 0
seen = set()
namefmt = "{name:<" + str(max(len(name) for name in AUTOGUESS_MAPPING.keys())) + "}"
hmifmt = "{huggingface_model_id:<" + str(max(len(huggingface_model_id) for huggingface_model_id in AUTOGUESS_MAPPING.values())) + "}"
for name, huggingface_model_id in AUTOGUESS_MAPPING.items():
    if filter and filter not in name:
        continue
    seen.add(name)
    if huggingface_model_id == "***UNKNOWN***":
        print(namefmt.format(name=name) + " = " + namefmt.format(name="***UNKNOWN***") + " : PENDING")
        continue
    tokenizer_config = get_tokenizer_config_for_huggingface_model_id(huggingface_model_id)
    assert 'chat_template' in tokenizer_config
    match = match_chat_template_to_adapter(tokenizer_config['chat_template'])
    if match is None:
        matched, sub_template, adapter = "MISSING", None, None
    else:
        match, sub_template = match
        matched = match['name']
        adapter = match['adapter']
    sub_template = f"[{sub_template}]" if sub_template else ""
    adaptercheck, reason = False, '?'
    if name == matched:
        assert adapter
        tokenizer = get_tokenizer_for_huggingface_model_id(huggingface_model_id)
        adaptercheck, reason = test_tokenizer_with_adapter(tokenizer, adapter, AUTOGUESS_SKIP_ADAPTER_TESTS.get(name, set()))
    print(namefmt.format(name=name) + " = " + namefmt.format(name=matched) + " : " + ("OK     " if adaptercheck and name == matched else reason if not adaptercheck else "FAILURE") + " " + hmifmt.format(huggingface_model_id=huggingface_model_id) + " " + sub_template)
    failures += name != matched or not adaptercheck

if filter is None:
    for entry in autoguess:
        if entry['name'] not in seen:
            print(namefmt.format(name=entry['name']) + "   MISSING MAPPING")
            failures += 1

if failures > 0:
    print(f"There were {failures} failure(s)!")
    sys.exit(1)
