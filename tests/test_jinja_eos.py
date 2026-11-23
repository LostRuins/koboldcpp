import os
import sys
import unittest
from unittest import mock

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import koboldcpp


class JinjaEosTests(unittest.TestCase):
    def test_chat_history_keeps_the_model_eos_token(self):
        template = (
            "{% for message in messages %}"
            "{{ message.content }}"
            "{% if message.role == 'assistant' %}{{ eos_token }}{% endif %}"
            "{% endfor %}"
        )
        messages = [
            {"role": "user", "content": "First question"},
            {"role": "assistant", "content": "First answer"},
            {"role": "user", "content": "Follow-up"},
        ]
        for tools in (None, [{"type": "function", "function": {"name": "weather"}}]):
            with self.subTest(tools=tools), mock.patch.object(
                koboldcpp, "cached_chat_template", template
            ), mock.patch.object(koboldcpp, "cached_eos_token", "</s>", create=True):
                prompt = koboldcpp.format_jinja(messages, tools)
                self.assertEqual(prompt, "First questionFirst answer</s>Follow-up")


if __name__ == "__main__":
    unittest.main()
