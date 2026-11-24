import argparse
import os
import sys
import unittest


sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import koboldcpp


class MmapFlagTests(unittest.TestCase):
    def parse_cli(self, argv):
        parser = argparse.ArgumentParser()
        parser.add_argument("model_param", nargs="?")
        parser.add_argument("--usemmap", nargs="?", const="prefetch", default=False, type=koboldcpp.validate_mmap_flag_list)
        args = parser.parse_args(koboldcpp.normalize_mmap_flag_args(argv))
        enabled, no_prefetch = koboldcpp.parse_mmap_flag_list(args.usemmap)
        return args.model_param, enabled, no_prefetch

    def test_legacy_values_keep_existing_behavior(self):
        for value, expected in ((None, (False, False)), (False, (False, False)), ("", (False, False)), (True, (True, False))):
            with self.subTest(value=value):
                self.assertEqual(koboldcpp.parse_mmap_flag_list(value), expected)

    def test_prefetch_and_noprefetch_flags(self):
        self.assertEqual(koboldcpp.parse_mmap_flag_list("prefetch"), (True, False))
        self.assertEqual(koboldcpp.parse_mmap_flag_list("noprefetch"), (True, True))
        self.assertEqual(koboldcpp.parse_mmap_flag_list(" NOPREFETCH "), (True, True))

    def test_unknown_and_conflicting_flags_are_rejected(self):
        for value in ("bogus", "prefetch,noprefetch", "noprefetch,prefetch"):
            with self.subTest(value=value), self.assertRaises(argparse.ArgumentTypeError):
                koboldcpp.parse_mmap_flag_list(value)

    def test_bare_usemmap_does_not_swallow_positional_model(self):
        self.assertEqual(self.parse_cli(["--usemmap", "model.gguf"]), ("model.gguf", True, False))

    def test_noprefetch_space_and_equals_forms(self):
        for argv in (["--usemmap", "noprefetch", "model.gguf"], ["--usemmap=noprefetch", "model.gguf"]):
            with self.subTest(argv=argv):
                self.assertEqual(self.parse_cli(argv), ("model.gguf", True, True))

    def test_bare_usemmap_at_end(self):
        self.assertEqual(self.parse_cli(["--usemmap"]), (None, True, False))


if __name__ == "__main__":
    unittest.main()
