import argparse
import ast
from pathlib import Path
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import koboldcpp


class ModelMemoryOptionsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Exercise the launcher's real declarations without starting the server.
        tree = ast.parse(Path(koboldcpp.__file__).read_text(encoding="utf-8"))
        main = next(n for n in tree.body if isinstance(n, ast.If) and
                    ast.unparse(n.test) == "__name__ == '__main__'")
        nodes = [n for n in main.body if isinstance(n, ast.FunctionDef) and n.name == "check_range"]
        nodes += [n for n in main.body if isinstance(n, ast.Expr) and isinstance(n.value, ast.Call)
                  and n.value.args and isinstance(n.value.args[0], ast.Constant)
                  and n.value.args[0].value in ("--lazy-mode", "--moe-cache-mib")]
        cls.parser = argparse.ArgumentParser()
        exec(compile(ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[])),
                     "memory_options", "exec"), {"argparse": argparse, "advparser": cls.parser})

    def test_gui_config_round_trip_and_old_config_defaults(self):
        tree = ast.parse(Path(koboldcpp.__file__).read_text(encoding="utf-8"))
        functions = {n.name: n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)}
        names = {"lazy_mode_var", "moe_cache_mib_var"}
        imports = [n for n in functions["import_vars"].body
                   if isinstance(n, ast.Expr) and isinstance(n.value, ast.Call)
                   and isinstance(n.value.func, ast.Attribute)
                   and isinstance(n.value.func.value, ast.Name)
                   and n.value.func.value.id in names]
        exports = [n for n in functions["export_vars"].body
                   if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Attribute)
                   and n.targets[0].attr in ("lazy_mode", "moe_cache_mib")]
        self.assertEqual((len(imports), len(exports)), (2, 2))

        class Variable:
            def set(self, value):
                self.value = value

            def get(self):
                return self.value

        env = {name: Variable() for name in names}
        env["args"] = SimpleNamespace()
        for config in ({"lazy_mode": "off", "moe_cache_mib": 4096},
                       {"lazy_mode": "on", "moe_cache_mib": 0}, {}):
            env["mydict"] = config
            for nodes in (imports, exports):
                exec(compile(ast.Module(body=nodes, type_ignores=[]), "gui_config", "exec"), env)
            self.assertEqual(env["args"].lazy_mode, config.get("lazy_mode", "auto"))
            self.assertEqual(env["args"].moe_cache_mib, config.get("moe_cache_mib", 0))

    def test_defaults_and_aliases(self):
        defaults = self.parser.parse_args([])
        self.assertEqual((defaults.lazy_mode, defaults.moe_cache_mib), ("auto", 0))
        for options in (["--lazy-mode", "off", "--moe-cache-mib", "4096"],
                        ["--lazymode", "off", "--moecache", "4096"]):
            parsed = self.parser.parse_args(options)
            self.assertEqual((parsed.lazy_mode, parsed.moe_cache_mib), ("off", 4096))

    def test_invalid_cli_values(self):
        for options in (["--lazy-mode", "invalid"], ["--moe-cache-mib", "-1"],
                        ["--moe-cache-mib", "1.5"], ["--moe-cache-mib", "2147483648"]):
            with self.subTest(options=options), self.assertRaises(SystemExit):
                self.parser.parse_args(options)

    def test_config_values_rejected_before_native_load(self):
        for mode, size in (("invalid", 0), ("auto", -1), ("on", 1.5),
                           ("on", True), ("off", "1024"), ("auto", 2147483648)):
            with self.subTest(mode=mode, size=size), patch.object(koboldcpp, "args",
                    SimpleNamespace(lazy_mode=mode, moe_cache_mib=size)):
                with self.assertRaises(ValueError):
                    koboldcpp.load_model("unused.gguf")

    def test_modes_and_cache_reach_native_input(self):
        class StopBeforeLoading(Exception):
            pass

        for mode, expected in (("off", 0), ("auto", 1), ("on", 2)):
            captured = {}

            class CaptureInputs:
                def __setattr__(self, name, value):
                    if name == "model_filename":
                        raise StopBeforeLoading()
                    captured[name] = value

            with patch.object(koboldcpp, "args", SimpleNamespace(lazy_mode=mode, moe_cache_mib=6144)), \
                 patch.object(koboldcpp, "load_model_inputs", CaptureInputs):
                with self.assertRaises(StopBeforeLoading):
                    koboldcpp.load_model("unused.gguf")
            self.assertEqual(captured, {"lazy_mode": expected, "moe_cache_mib": 6144})


if __name__ == "__main__":
    unittest.main()
