import argparse
import os
import sys
import unittest
from unittest import mock


sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import koboldcpp


class OllamaChatFormatTests(unittest.TestCase):
    def transform(self, **extra):
        genparams = {"model": "m", "messages": [{"role": "user", "content": "hi"}], **extra}
        args = argparse.Namespace(defaultgenamt=512, quiet=True, debugmode=0, nobostoken=False)
        with mock.patch.object(koboldcpp, "args", args, create=True), \
             mock.patch.object(koboldcpp, "chatcompl_adapter", None, create=True):
            koboldcpp.transform_genparams(genparams, 7, False)
        return genparams

    def test_format_json_constrains_to_an_object(self):
        genparams = self.transform(format="json")
        self.assertEqual(genparams["response_format"], {"type": "json_object"})
        self.assertIn("root   ::= object", genparams["grammar"])

    def test_format_schema_uses_the_schema(self):
        schema = {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}
        genparams = self.transform(format=schema)
        self.assertEqual(genparams["response_format"]["json_schema"]["schema"], schema)
        self.assertIn("city", genparams["grammar"])

    def test_no_format_means_no_grammar(self):
        genparams = self.transform()
        self.assertNotIn("response_format", genparams)
        self.assertFalse(genparams.get("grammar"))

    def test_explicit_response_format_wins(self):
        genparams = self.transform(format="json", response_format={"type": "text"})
        self.assertEqual(genparams["response_format"], {"type": "text"})


if __name__ == "__main__":
    unittest.main()
