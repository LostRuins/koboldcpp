"""
Test that the AutoGuess feature picks the correct model for every template.
Also checks that every template is being tested so that when new AutoGuess additions are made, this test fails unless an accompanying test is included.
The strict check renders whole conversations with each template and compares them byte-for-byte with the prompts
koboldcpp builds from the AutoGuess.json and AutoGuess-NoThink.json adapters.
"""
import os
import sys
import jinja2
import jinja2.ext
import requests
import json
from datetime import datetime
from jinja2.sandbox import ImmutableSandboxedEnvironment
from transformers import AutoTokenizer


# Map an AutoGuess name to a HuggingFace model ID
# THIS LIST MUST BE UPDATED WHEN A NEW MODEL IS ADDED
AUTOGUESS_MAPPING = {
    "ChatML (Phi 4)": "microsoft/phi-4",
    "ChatML (Qwen 2.5 based)": "Qwen/Qwen2.5-0.5B-Instruct",
    "ChatML (Kimi)": "moonshotai/Kimi-K2-Instruct",
    "ChatML (Kimi Alternate)": "moonshotai/Moonlight-16B-A3B-Instruct",
    "Google Gemma 2": "Efficient-Large-Model/gemma-2-2b-it",
    "Google Gemma 3": "mlx-community/gemma-3-4b-it-bf16",
    "Google Gemma 3n": "lmstudio-community/gemma-3n-E4B-it-MLX-bf16",
    "Google Gemma 4 (26B and 31B)": "google/gemma-4-31B-it",
    "Google Gemma 4 (E2B and E4B)": "google/gemma-4-E4B-it",
    "Llama 3.x": "Steelskull/L3.3-Shakudo-70b",
    "Llama 4": "lmstudio-community/Llama-4-Scout-17B-16E-MLX-text-8bit",
    "Granite 4": "ibm-granite/granite-4.0-micro",
    "Mistral Tekken": "Doctor-Shotgun/MS3.2-24B-Magnum-Diamond",
    "Mistral Non-Tekken": "mistralai/Mistral-7B-Instruct-v0.3",
    "GLM-5.2/5.3": "zai-org/GLM-5.3-Flash",
    "GLM-4.7": "zai-org/GLM-4.7",
    "GLM-4.5/4.6": "zai-org/GLM-4.6",
    "GLM-4": "THUDM/glm-4-9b-chat-hf",
    "Phi 3.5": "microsoft/Phi-3.5-mini-instruct",
    "Phi 4 (mini)": "microsoft/Phi-4-mini-instruct",
    "Cohere": "mlx-community/aya-expanse-32b-8bit",
    "Spark X2.5": "XHToken/Spark-X2.5-4B",
    "DeepSeek V2.5": "deepseek-ai/DeepSeek-V2.5",
    "Jamba": "ai21labs/Jamba-tiny-dev",
    "Dots": "rednote-hilab/dots.llm1.inst",
    "OpenAI Harmony": "openai/gpt-oss-120b",
    "Seed OSS": "ByteDance-Seed/Seed-OSS-36B-Instruct",
    "RWKV World": "fla-hub/rwkv7-1.5B-world",
    "Mistral (Generic)": "mistralai/Mistral-Nemo-Instruct-2407",
    "ChatML (MiMo)": "XiaomiMiMo/MiMo-V2.6-Flash-RL",
    "ChatML (Generic)": "NewEden/Gemma-27B-chatml",
    "Apriel 1.6": "ServiceNow-AI/Apriel-1.6-15b-Thinker",
    "Apertus": "swiss-ai/Apertus-8B-Instruct-2509",
    "MiniMax": "MiniMaxAI/MiniMax-M2.7",
    "Muse Glimmer": "meta-models/Muse-Glimmer-30B",
    "Ling 3.x": "inclusionAI/Ling-3.0-flash",
    "Ling 2.0": "inclusionAI/Ling-flash-2.0",
}

# "system" skips system role checks, "no_system" skips the strict check without a system message, "two_users" skips the
# strict check with two user messages in a row, "nothink" skips the AutoGuess-NoThink.json checks, "default_system"
# allows the template's default system prompt to be missing when there is no system message
AUTOGUESS_SKIP_ADAPTER_TESTS = {
    "Mistral Non-Tekken": {"system"},   # Poor system support
    "Mistral (Generic)": {"system"},    # Poor system support
    "OpenAI Harmony": {"system", "default_system"},  # Template always starts with a system block containing today's date
    "Muse Glimmer": {"default_system"},    # Default system prompt contains today's date
    "Apertus": {"default_system"},         # Default system prompt contains today's date
    "Mistral Tekken": {"default_system"},  # Default system prompt contains today's date
    "ChatML (Qwen 2.5 based)": {"default_system"},  # Also matches Qwen3 and QwQ, which have no default system prompt
    "ChatML (Kimi)": {"default_system"},   # Also matches Kimi-Linear, which has no default system prompt
    "Granite 4": {"default_system"},       # Also matches Granite 3, whose default system prompt contains today's date
    "MiniMax": {"default_system"},         # Default system prompt names one model version (MiniMax-M2.7)
    "Llama 3.x": {"default_system"},       # Also matches Llama 3.0 (no default system prompt) and 3.1/3.3 (knowledge cutoff block)
    "Jamba": {"default_system"},           # Default system turn is just a space; other Jamba templates are gated, so unchecked
}

# Every other file in kcpp_adapters must be identical to an AutoGuess.json or AutoGuess-NoThink.json adapter, so that it
# is covered by their checks. Files listed here are checked against the template of an AutoGuess model instead
STANDALONE_ADAPTER_TESTS = {
    "Gemma-4-Thinking.json": ("Google Gemma 4 (26B and 31B)", {"enable_thinking": True}),
}
# Generic formats with no model template to check against
STANDALONE_WITHOUT_TEMPLATE = {"Alpaca.json", "Llama-2-Chat.json", "Metharme.json", "Vicuna.json"}

# User may be running this test from ./ or from ../ -- we want to be in ./ (i.e. tests)
if os.path.exists("tests"):
    os.chdir("tests")

with open("../kcpp_adapters/AutoGuess.json") as f:
    autoguess = json.load(f)
with open("../kcpp_adapters/AutoGuess-NoThink.json") as f:
    autoguess_nothink = json.load(f)

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
    # never run tokenizer code from model repos; tokenizers that need it fail here and use ChatTemplateRenderer instead
    if os.path.exists(dname):
        return AutoTokenizer.from_pretrained(dname, trust_remote_code=False)
    return AutoTokenizer.from_pretrained(huggingface_model_id, trust_remote_code=False)

def load_tokenizer(huggingface_model_id: str, tokenizer_config: dict, sub_template: str|None):
    """
    Returns the tokenizer, or a jinja2 renderer of its chat template if transformers can't load it, plus a note saying so.
    """
    try:
        return get_tokenizer_for_huggingface_model_id(huggingface_model_id), ""
    except Exception as e:
        return ChatTemplateRenderer(tokenizer_config, sub_template), f" (tokenizer failed to load: {type(e).__name__}, rendered template with jinja2)"

class IgnoreGenerationTags(jinja2.ext.Extension):
    # transformers marks assistant text with {% generation %} blocks; render their contents unchanged
    tags = {"generation"}
    def parse(self, parser):
        parser.stream.skip(1)
        return parser.parse_statements(("name:endgeneration",), drop_needle=True)

class ChatTemplateRenderer:
    """
    Fallback for tokenizers the installed transformers cannot load (e.g. Gemma 4 and GLM-5.3 need transformers 5).
    Renders the chat template the same way transformers' apply_chat_template does.
    """
    def __init__(self, tokenizer_config: dict, sub_template: str|None):
        chat_template = tokenizer_config['chat_template']
        if isinstance(chat_template, list):
            chat_template = next(t['template'] for t in chat_template if t['name'] == sub_template)
        def tojson(x, ensure_ascii=False, indent=None, separators=None, sort_keys=False):
            return json.dumps(x, ensure_ascii=ensure_ascii, indent=indent, separators=separators, sort_keys=sort_keys)
        def raise_exception(message):
            raise jinja2.exceptions.TemplateError(message)
        env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True, extensions=[IgnoreGenerationTags, jinja2.ext.loopcontrols])
        env.filters["tojson"] = tojson
        env.globals["raise_exception"] = raise_exception
        env.globals["strftime_now"] = lambda format: datetime.now().strftime(format)
        self.template = env.from_string(chat_template)
        # special tokens such as bos_token may be plain strings or {"content": ...} objects. tokens only listed in
        # special_tokens_map.json are left out, which can drop a leading BOS but doesn't affect the fragment checks
        self.special_tokens = {k: v["content"] if isinstance(v, dict) else v for k, v in tokenizer_config.items()
                               if k.endswith("_token") and (isinstance(v, str) or (isinstance(v, dict) and "content" in v))}
        self.bos_token = self.special_tokens.get("bos_token")

    def apply_chat_template(self, conversation, tokenize=False, add_generation_prompt=False, **kwargs):
        return self.template.render(messages=conversation, add_generation_prompt=add_generation_prompt, **self.special_tokens, **kwargs)

def match_chat_template_to_adapter(chat_template: str|list, adapter_list: list[dict] = autoguess) -> tuple[dict, str|None]|None:
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
        for entry in adapter_list:
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

STRICT_TURNS = [
    {"role": "user", "content": "user_1"},
    {"role": "assistant", "content": "asst_1"},
    {"role": "user", "content": "user_2"},
]

def strict_test_tokenizer_with_adapter(tokenizer, adapter: dict[str, str], skip: set, **template_kwargs) -> list[str]:
    """
    Render whole conversations, generation prompt included, and compare them byte-for-byte with the prompt koboldcpp
    builds from the adapter: chat_start, then default_system if there is no system message, then each message wrapped in
    its role's start/end, then assistant_gen.
    """
    def has_system(messages):
        return any(m["role"] == "system" for m in messages)
    def build(messages):
        return (adapter.get("chat_start", "")
                + ("" if has_system(messages) else adapter.get("default_system", ""))
                + "".join(adapter[f"{m['role']}_start"] + m["content"] + adapter[f"{m['role']}_end"] for m in messages)
                + adapter.get("assistant_gen", adapter["assistant_start"]))
    def strip_backend_prefixes(text):
        # koboldcpp adds the BOS token itself, so it doesn't belong to the adapter
        if tokenizer.bos_token and text.startswith(tokenizer.bos_token):
            text = text[len(tokenizer.bos_token):]
        return text

    cases = {"no_system": STRICT_TURNS, "two_users": [{"role": "user", "content": "user_1"}, {"role": "user", "content": "user_2"}]}
    if adapter["system_start"] + adapter["system_end"] != adapter["user_start"] + adapter["user_end"]:
        cases["system"] = [{"role": "system", "content": "SyS-tEm"}] + STRICT_TURNS
    errors = []
    for case, messages in cases.items():
        if case in skip:
            continue
        try:
            templated = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, **template_kwargs)
        except jinja2.exceptions.TemplateError as e:
            # some templates refuse consecutive user messages, so there is nothing to compare against
            if case != "two_users":
                errors.append(f"{case}: template error: {e}")
            continue
        templated = strip_backend_prefixes(templated)
        expect = strip_backend_prefixes(build(messages))
        # without a system message some templates add a default system prompt, which may only be missing for entries
        # skipped with "default_system"
        allow_missing_default = not has_system(messages) and "default_system" in skip
        if templated != expect and not (allow_missing_default and templated.endswith(expect)):
            errors.append(f"{case}: full render differs\n\tadapter:   {expect.replace("\n", "\\n")}\n\ttokenizer: {templated.replace("\n", "\\n")}")
    return errors

# What koboldcpp passes to jinja templates when thinking is disabled
NOTHINK_KWARGS = {"enable_thinking": False, "reasoning_strength": "none"}

def template_has_thinking_switch(tokenizer) -> bool:
    try:
        return (tokenizer.apply_chat_template(STRICT_TURNS, tokenize=False, add_generation_prompt=True)
                != tokenizer.apply_chat_template(STRICT_TURNS, tokenize=False, add_generation_prompt=True, **NOTHINK_KWARGS))
    except jinja2.exceptions.TemplateError:
        return False

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
    adaptercheck, reason = False, '?'
    strict_errors = []
    renderer_note = ""
    if name == matched:
        assert adapter
        tokenizer, renderer_note = load_tokenizer(huggingface_model_id, tokenizer_config, sub_template)
        skip =AUTOGUESS_SKIP_ADAPTER_TESTS.get(name, set())
        adaptercheck, reason = test_tokenizer_with_adapter(tokenizer, adapter, skip)
        strict_errors = strict_test_tokenizer_with_adapter(tokenizer, adapter, skip)
        if "nothink" not in skip:
            nothink_match = match_chat_template_to_adapter(tokenizer_config['chat_template'], autoguess_nothink)
            nothink_matched = nothink_match[0]['name'] if nothink_match else "MISSING"
            if nothink_matched != name:
                strict_errors.append(f"nothink: AutoGuess-NoThink.json picked {nothink_matched}")
            elif template_has_thinking_switch(tokenizer):
                strict_errors += [f"nothink {e}" for e in strict_test_tokenizer_with_adapter(tokenizer, nothink_match[0]['adapter'], skip, **NOTHINK_KWARGS)]
            else:
                # without a switch the NoThink adapter can only turn thinking off in its generation prompt, so check
                # everything else against the template, using the (already checked) regular generation prompt
                history_only = dict(nothink_match[0]['adapter'], assistant_gen=adapter.get("assistant_gen", adapter["assistant_start"]))
                strict_errors += [f"nothink {e}" for e in strict_test_tokenizer_with_adapter(tokenizer, history_only, skip)]
    sub_template = f"[{sub_template}]" if sub_template else ""
    print(namefmt.format(name=name) + " = " + namefmt.format(name=matched) + " : " + ("OK     " if adaptercheck and name == matched and not strict_errors else reason if not adaptercheck else "FAILURE" if name != matched else "STRICT ") + " " + hmifmt.format(huggingface_model_id=huggingface_model_id) + " " + sub_template + renderer_note)
    for e in strict_errors:
        print(f"    strict {e}")
    failures += name != matched or not adaptercheck or bool(strict_errors)

if filter is None:
    for entry in autoguess:
        if entry['name'] not in seen:
            print(namefmt.format(name=entry['name']) + "   MISSING MAPPING")
            failures += 1
    for entry in autoguess_nothink:
        if entry['name'] not in AUTOGUESS_MAPPING:
            print(namefmt.format(name=entry['name']) + "   MISSING MAPPING (AutoGuess-NoThink.json)")
            failures += 1

    autoguess_adapters = [entry['adapter'] for entry in autoguess + autoguess_nothink]
    for fname in sorted(os.listdir("../kcpp_adapters")):
        if not fname.endswith(".json") or fname.startswith("AutoGuess") or fname in STANDALONE_WITHOUT_TEMPLATE:
            continue
        with open(f"../kcpp_adapters/{fname}") as f:
            adapter = json.load(f)
        renderer_note = ""
        if fname in STANDALONE_ADAPTER_TESTS:
            name, template_kwargs = STANDALONE_ADAPTER_TESTS[fname]
            tokenizer_config = get_tokenizer_config_for_huggingface_model_id(AUTOGUESS_MAPPING[name])
            _, sub_template = match_chat_template_to_adapter(tokenizer_config['chat_template'])
            tokenizer, renderer_note = load_tokenizer(AUTOGUESS_MAPPING[name], tokenizer_config, sub_template)
            errors = strict_test_tokenizer_with_adapter(tokenizer, adapter, AUTOGUESS_SKIP_ADAPTER_TESTS.get(name, set()), **template_kwargs)
        elif adapter in autoguess_adapters:
            errors = []
        else:
            errors = ["not identical to any AutoGuess.json or AutoGuess-NoThink.json adapter"]
        print(namefmt.format(name=fname) + "   " + ("OK     " if not errors else "STRICT ") + renderer_note)
        for e in errors:
            print(f"    strict {e}")
        failures += bool(errors)

if failures > 0:
    print(f"There were {failures} failure(s)!")
    sys.exit(1)
