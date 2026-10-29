import argparse
import os
import sys
import unittest
from unittest import mock


sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import koboldcpp


TOOLS = [
    {"name": "get_weather", "input_schema": {"type": "object", "properties": {"city": {"type": "string"}}}},
    {"name": "get_time", "input_schema": {"type": "object", "properties": {}}},
]


class AnthropicToolChoiceTests(unittest.TestCase):
    def transform(self, tool_choice, tools, model_reply):
        genparams = {"messages": [{"role": "user", "content": "what time is it"}], "max_tokens": 64,
                     "tools": [dict(t) for t in tools], "tool_choice": tool_choice}
        polls = []

        def fake_generate(genparams):
            polls.append(genparams)
            return {"text": model_reply}

        args = argparse.Namespace(defaultgenamt=512, quiet=True, debugmode=0, nobostoken=False)
        with mock.patch.object(koboldcpp, "args", args, create=True), \
             mock.patch.object(koboldcpp, "generate", fake_generate), \
             mock.patch.object(koboldcpp, "chatcompl_adapter", None, create=True):
            koboldcpp.transform_genparams(genparams, 9, False)
        return genparams, polls

    def test_auto_lets_the_model_decide(self):
        genparams, polls = self.transform({"type": "auto"}, TOOLS[:1], '{"reasoning":"chat","final_decision":"no","tool_name":"null"}')
        self.assertEqual(len(polls), 1)
        self.assertFalse(genparams.get("_generic_tool_json_array_fix"))

        genparams, polls = self.transform({"type": "auto"}, TOOLS, '{"reasoning":"time","final_decision":"yes","tool_name":"get_time"}')
        self.assertTrue(genparams.get("_generic_tool_json_array_fix"))
        self.assertIn("get_time", genparams["grammar"])

    def test_any_requires_a_tool(self):
        genparams, polls = self.transform({"type": "any"}, TOOLS, '{"tool_name":"get_time"}')
        self.assertEqual(len(polls), 1)
        self.assertTrue(genparams.get("_generic_tool_json_array_fix"))
        self.assertIn("get_time", genparams["grammar"])

    def test_named_tool_is_used(self):
        genparams, polls = self.transform({"type": "tool", "name": "get_time"}, TOOLS, "")
        self.assertEqual(polls, [])
        self.assertIn("get_time", genparams["grammar"])
        self.assertNotIn("get_weather", genparams["grammar"])


if __name__ == "__main__":
    unittest.main()
