import argparse
import os
import sys
import unittest
from unittest import mock


sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import koboldcpp


OPTIONS = {"num_predict": 42, "num_ctx": 2048, "temperature": 0.1, "top_k": 7, "top_p": 0.5, "seed": 1234, "stop": ["END"]}


class OllamaOptionsTests(unittest.TestCase):
    def transform(self, genparams, api_format):
        args = argparse.Namespace(defaultgenamt=512, quiet=True, debugmode=0, nobostoken=False)
        with mock.patch.object(koboldcpp, "args", args, create=True), \
             mock.patch.object(koboldcpp, "chatcompl_adapter", None, create=True):
            koboldcpp.transform_genparams(genparams, api_format, False)
        return genparams

    def assert_options_applied(self, genparams):
        self.assertEqual(genparams["max_length"], 42)
        self.assertEqual(genparams["max_context_length"], 2048)
        self.assertEqual(genparams["temperature"], 0.1)
        self.assertEqual(genparams["top_k"], 7)
        self.assertEqual(genparams["top_p"], 0.5)
        self.assertEqual(genparams["sampler_seed"], 1234)
        self.assertIn("END", genparams["stop_sequence"])

    def test_generate_applies_options(self):
        genparams = self.transform({"model": "m", "prompt": "hi", "options": dict(OPTIONS, stop=["END"])}, 6)
        self.assert_options_applied(genparams)

    def test_chat_applies_options(self):
        genparams = self.transform({"model": "m", "messages": [{"role": "user", "content": "hi"}],
                                    "options": dict(OPTIONS, stop=["END"])}, 7)
        self.assert_options_applied(genparams)

    def test_chat_without_options_keeps_defaults(self):
        genparams = self.transform({"model": "m", "messages": [{"role": "user", "content": "hi"}]}, 7)
        self.assertEqual(genparams["max_length"], 512)
        self.assertEqual(genparams["sampler_seed"], -1)
        self.assertNotIn("temperature", genparams)

    def test_chat_non_positive_num_predict_uses_default(self):
        genparams = self.transform({"model": "m", "messages": [{"role": "user", "content": "hi"}],
                                    "options": {"num_predict": -1}}, 7)
        self.assertEqual(genparams["max_length"], 512)


if __name__ == "__main__":
    unittest.main()
