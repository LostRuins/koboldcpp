import sys
import os

parent_dir = os.path.abspath(os.path.join(__file__, "..", ".."))
sys.path.append(parent_dir)

import koboldcpp

def extract_loras_from_prompt(*args, **kwargs):
    """
    >>> prompt = "no <lora: tag, even though with a : and 0> it could look like it"
    >>> clean, data = extract_loras_from_prompt(prompt)
    >>> clean
    'no <lora: tag, even though with a : and 0> it could look like it'
    >>> data
    []

    >>> prompt = "even after a <lora:valid:1> tag, an unending <lora: tag should be ignored"
    >>> clean, data = extract_loras_from_prompt(prompt)
    >>> clean
    'even after a  tag, an unending <lora: tag should be ignored'
    >>> data
    [{'name': 'valid', 'multiplier': 1.0}]

    >>> prompt = "A portrait <lora:models/face:0.8> with soft lighting"
    >>> clean, data = extract_loras_from_prompt(prompt)
    >>> clean
    'A portrait  with soft lighting'
    >>> data
    [{'name': 'models/face', 'multiplier': 0.8}]

    >>> prompt = "<lora:foo:1.0> start <lora:|high_noise|bar:0.5> end"
    >>> clean, data = extract_loras_from_prompt(prompt)
    >>> clean
    ' start  end'
    >>> data
    [{'name': 'foo', 'multiplier': 1.0}, {'name': 'bar', 'multiplier': 0.5, 'is_high_noise': True}]

    >>> prompt = "bad <lora:bad:abc> good <lora:good:2>"
    >>> clean, data = extract_loras_from_prompt(prompt)
    >>> clean
    'bad <lora:bad:abc> good '
    >>> data
    [{'name': 'good', 'multiplier': 2.0}]

    >>> prompt = "x<lora:a:0.15>y<lora:b:0.2>z"
    >>> clean, data = extract_loras_from_prompt(prompt)
    >>> clean
    'xyz'
    >>> data
    [{'name': 'a', 'multiplier': 0.15}, {'name': 'b', 'multiplier': 0.2}]
    """

    return koboldcpp.extract_loras_from_prompt(*args, **kwargs)

def mk_lora_info(imgloras, multipliers):
    """
    >>> pre, path, name = mk_lora_info(['/x/lora1.safetensors', '/y/lora2.gguf'], [])
    fake filesystem access
    fake filesystem access
    >>> pre
    [{'fullpath': '/x/lora1.safetensors', 'name': 'lora1', 'path': 'lora1.safetensors', 'multiplier': 1.0, 'preloaded': True}, {'fullpath': '/y/lora2.gguf', 'name': 'lora2', 'path': 'lora2.gguf', 'multiplier': 1.0, 'preloaded': True}]
    >>> path
    {}
    >>> name
    {}

    >>> pre, path, name = mk_lora_info(['/x/lora1.safetensors', '/y/lora2.gguf'], [0.])
    fake filesystem access
    fake filesystem access
    >>> pre
    [{'fullpath': '/x/lora1.safetensors', 'name': 'lora1', 'path': 'lora1.safetensors', 'multiplier': 0.0, 'preloaded': True}, {'fullpath': '/y/lora2.gguf', 'name': 'lora2', 'path': 'lora2.gguf', 'multiplier': 0.0, 'preloaded': True}]
    >>> path
    {'lora1.safetensors': {'fullpath': '/x/lora1.safetensors', 'name': 'lora1', 'path': 'lora1.safetensors', 'multiplier': 0.0, 'preloaded': True}, 'lora2.gguf': {'fullpath': '/y/lora2.gguf', 'name': 'lora2', 'path': 'lora2.gguf', 'multiplier': 0.0, 'preloaded': True}}
    >>> name
    {'lora1': 'lora1.safetensors', 'lora2': 'lora2.gguf'}

    >>> pre, path, name = mk_lora_info(['/x/lora1.safetensors', '/y/lora1.safetensors'], [0.3])
    fake filesystem access
    fake filesystem access
    >>> pre
    [{'fullpath': '/x/lora1.safetensors', 'name': 'lora1', 'path': 'lora1.safetensors', 'multiplier': 0.3, 'preloaded': True}, {'fullpath': '/y/lora1.safetensors', 'name': 'lora1_2', 'path': 'lora1_2.safetensors', 'multiplier': 0.3, 'preloaded': True}]
    >>> path
    {}

    >>> pre, path, name = mk_lora_info(['/lora/dir/'], [0.3])
    fake filesystem access
    Scanning /lora/dir/ for LoRAs...
    fake directory scan
      found 2 files under /lora/dir/
    >>> pre
    []
    >>> expected = {
    ... 'lora1_makebelieve.gguf': {
    ...     'fullpath': '/lora/dir/lora1_makebelieve.gguf',
    ...     'name': 'lora1_makebelieve',
    ...     'path': 'lora1_makebelieve.gguf',
    ...     'multiplier': 0.0,
    ...     'preloaded': False},
    ... 'lora2/makebelieve.gguf': {
    ...     'fullpath': '/lora/dir/lora2/makebelieve.gguf',
    ...     'name': 'lora2/makebelieve',
    ...     'path': 'lora2/makebelieve.gguf',
    ...     'multiplier': 0.0,
    ...     'preloaded': False}}
    >>> path == expected
    True
    >>> name
    {'lora1_makebelieve': 'lora1_makebelieve.gguf', 'lora2/makebelieve': 'lora2/makebelieve.gguf'}

    """
    return koboldcpp.mk_lora_info(imgloras, multipliers, True)

def sanitize_lora_multipliers(*args, **kwargs):
    """
    >>> sanitize_lora_multipliers(None)
    [1.0]

    >>> sanitize_lora_multipliers(0.75)
    [0.75]
    >>> sanitize_lora_multipliers("2")
    [2.0]

    >>> sanitize_lora_multipliers([0.5, "1.2", 3])
    [0.5, 1.2, 3.0]

    >>> sanitize_lora_multipliers([])
    []

    >>> sanitize_lora_multipliers(["bad", None, ""])
    [0.0, 0.0, 0.0]
    """
    return koboldcpp.sanitize_lora_multipliers(*args, **kwargs)


def prepare_lora_multipliers(req_list, imglora_bypath):
    """
    >>> req = [
    ...     {"path": "a.gguf", "multiplier": "0.5"},
    ...     {"path": "a.gguf", "multiplier": 1.0},
    ... ]
    >>> imglora = {"a.gguf": {"fullpath": "/abs/a.gguf"}}
    >>> paths, mults = prepare_lora_multipliers(req, imglora)
    >>> paths == [b"/abs/a.gguf"], mults == [1.5]
    (True, True)

    >>> req = [
    ...     {"path": "b.gguf", "multiplier": "2"},
    ...     {"path": "c.gguf"},
    ...     "not a dict",
    ...     {"path": "", "multiplier": "3"},
    ...     {"path": "b.gguf", "multiplier": 0},
    ... ]
    >>> imglora = {"b.gguf": {"fullpath": "/abs/b.gguf"},
    ...            "c.gguf": {"fullpath": "/abs/c.gguf"}}
    >>> paths, mults = prepare_lora_multipliers(req, imglora)
    >>> paths == [b"/abs/b.gguf"], mults == [2.0]
    (True, True)

    >>> req = [{"path": "missing.gguf", "multiplier": "5"}]
    >>> imglora = {}
    >>> paths, mults = prepare_lora_multipliers(req, imglora)
    >>> paths == [], mults == []
    (True, True)

    >>> req = [
    ...     {"path": "x.gguf", "multiplier": 1},
    ...     {"path": "y.gguf", "multiplier": 2},
    ... ]
    >>> imglora = {
    ...     "x.gguf": {"fullpath": "/abs/x.gguf", "path": "x.gguf", "multiplier": 0.0},
    ...     "y.gguf": {"fullpath": "/abs/y.gguf", "path": "y.gguf", "multiplier": 0.0},
    ... }
    >>> paths, mults = prepare_lora_multipliers(req, imglora)
    >>> paths == [b'/abs/x.gguf', b'/abs/y.gguf']
    True
    >>> mults == [1.0, 2.0]
    True
    """
    return koboldcpp.prepare_lora_multipliers_backend(req_list, imglora_bypath)

def mk_sdapi_lora_list(imglora_bypath):
    '''
    >>> imglora_bypath = {
    ...     'lora_a.safetensors': {'name': 'lora_a', 'path': 'lora_a.safetensors', 'multiplier': 0.0},
    ...     'lora_b.gguf'       : {'name': 'lora_b', 'path': 'lora_b.gguf', 'multiplier': 0.0},
    ...     'lora_c.safetensors': {'name': 'lora_c', 'path': 'lora_c.safetensors', 'multiplier': 1.0},
    ...     'chars/waifu.gguf'  : {'name': 'chars/waifu', 'path': 'chars/waifu.gguf', 'multiplier': 0.0}
    ... }
    >>> mk_sdapi_lora_list(imglora_bypath)
    [{'name': 'lora_a', 'path': 'lora_a.safetensors'}, {'name': 'lora_b', 'path': 'lora_b.gguf'}, {'name': 'chars/waifu', 'path': 'chars/waifu.gguf'}]

    >>> empty_data = {}
    >>> mk_sdapi_lora_list(empty_data)
    []

    >>> mixed_data = {
    ...     'k1': {'name': 'X', 'path': 'p1', 'multiplier': 0.5},
    ...     'k2': {'name': 'Y', 'path': 'p2', 'multiplier': 0.0}
    ... }
    >>> mk_sdapi_lora_list(mixed_data)
    [{'name': 'Y', 'path': 'p2'}]
    '''
    return koboldcpp.mk_sdapi_lora_list(imglora_bypath)


def gendefaults_parse_meta_field(*args, **kwargs):
    '''

    >>> [gendefaults_parse_meta_field(x) for x in [{}, None, '', "invalid json", '  ', 4]]
    Warning: gendefaults field - not a JSON object.
    Warning: couldn't parse gendefaults field.
    Warning: gendefaults field - not a JSON object.
    [{}, {}, {}, {}, {}, {}]

    >>> [gendefaults_parse_meta_field(x) for x in ['["valid", "json"]', 'but', '1']]
    Warning: gendefaults field - not a JSON object.
    Warning: couldn't parse gendefaults field.
    Warning: gendefaults field - not a JSON object.
    [{}, {}, {}]

    >>> gendefaults_parse_meta_field({"key": "value"})
    {'key': 'value'}

    >>> gendefaults_parse_meta_field(' "scheduler": "default", "steps": 10 ')
    {'scheduler': 'default', 'steps': 10}

    >>> gendefaults_parse_meta_field('{"cfg-scale": 0.5, "cfg_scale": 0.7}')
    {'cfg-scale': 0.5, 'cfg_scale': 0.7}
 
    >>> gendefaults_parse_meta_field('{"guidance": 1.2, "sampler": "ddim"}')
    {'distilled_guidance': 1.2, 'sampler_name': 'ddim', 'guidance': 1.2, 'sampler': 'ddim'}
    '''
    return koboldcpp.gendefaults_parse_meta_field(*args, **kwargs)


if __name__ == '__main__':
    import doctest
    failures, _ = doctest.testmod()
    if failures:
        raise SystemExit(f"{failures} doctest{'s' if failures != 1 else ''} failed")

