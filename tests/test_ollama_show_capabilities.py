import os
import sys
import unittest
from unittest import mock


sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import koboldcpp


class OllamaShowCapabilitiesTests(unittest.TestCase):
    def capabilities(self, vision=False, autoswap=False, mmproj=None):
        with mock.patch.object(koboldcpp, "has_vision_support", vision), \
             mock.patch.object(koboldcpp, "autoswapmode", autoswap), \
             mock.patch.object(koboldcpp, "mmprojName", mmproj):
            return koboldcpp.get_ollama_capabilities()

    def test_text_model_reports_completion_and_tools(self):
        self.assertEqual(self.capabilities(), ["completion", "tools"])

    def test_vision_model_reports_vision(self):
        self.assertEqual(self.capabilities(vision=True), ["completion", "tools", "vision"])

    def test_router_mode_with_mmproj_reports_vision(self):
        self.assertIn("vision", self.capabilities(autoswap=True, mmproj="mmproj.gguf"))
        self.assertNotIn("vision", self.capabilities(autoswap=False, mmproj="mmproj.gguf"))


if __name__ == "__main__":
    unittest.main()
