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


if __name__ == '__main__':
    import doctest
    failures, _ = doctest.testmod()
    if failures:
        raise SystemExit(f"{failures} doctest{'s' if failures != 1 else ''} failed")

