"""
Doctest-based tests for the CPU ISA library selection in koboldcpp.py.
"""
import doctest
import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import koboldcpp


class IsaSelectionTests(unittest.TestCase):
    def test_select_isa_libraries(self):
        """
        >>> cpu = {"default": "koboldcpp_default", "noavx2": "koboldcpp_noavx2", "failsafe": "koboldcpp_failsafe"}
        >>> all_files = ["koboldcpp_default.so", "koboldcpp_noavx2.so", "koboldcpp_failsafe.so",
        ...              "koboldcpp_legacy_ggml.so", "koboldcpp_legacy_ggml_noavx2.so", "koboldcpp_legacy_ggml_failsafe.so"]

        A machine with AVX2 gets the strongest flavor, backend and legacy:

        >>> select_isa_libraries({"avx", "avx2"}, all_files, cpu)
        {'backend': 'koboldcpp_default.so', 'legacy': 'koboldcpp_legacy_ggml.so', 'backend_matched': True, 'legacy_matched': True}

        Without AVX2, the noavx2 flavor:

        >>> select_isa_libraries({"avx"}, all_files, cpu)
        {'backend': 'koboldcpp_noavx2.so', 'legacy': 'koboldcpp_legacy_ggml_noavx2.so', 'backend_matched': True, 'legacy_matched': True}

        A baseline machine gets the failsafe flavor:

        >>> select_isa_libraries(set(), all_files, cpu)
        {'backend': 'koboldcpp_failsafe.so', 'legacy': 'koboldcpp_legacy_ggml_failsafe.so', 'backend_matched': True, 'legacy_matched': True}

        When no available flavor matches, the most general available one is
        returned with matched=False (the caller warns and tries it anyway):

        >>> select_isa_libraries(set(), ["koboldcpp_default.so", "koboldcpp_legacy_ggml.so"], cpu)
        {'backend': 'koboldcpp_default.so', 'legacy': 'koboldcpp_legacy_ggml.so', 'backend_matched': False, 'legacy_matched': False}

        The backend and the legacy library are picked separately, so partial
        sets compose (a failsafe legacy with a noavx2 backend on an AVX1
        machine):

        >>> select_isa_libraries({"avx"}, ["koboldcpp_noavx2.so", "koboldcpp_legacy_ggml_failsafe.so"], cpu)
        {'backend': 'koboldcpp_noavx2.so', 'legacy': 'koboldcpp_legacy_ggml_failsafe.so', 'backend_matched': True, 'legacy_matched': True}

        The --noavx2 option rules out the default flavor in both picks:

        >>> select_isa_libraries({"avx", "avx2"}, all_files, cpu, {"noavx2": True})
        {'backend': 'koboldcpp_noavx2.so', 'legacy': 'koboldcpp_legacy_ggml_noavx2.so', 'backend_matched': True, 'legacy_matched': True}

        The --failsafe option only considers the failsafe flavor:

        >>> select_isa_libraries({"avx", "avx2"}, all_files, cpu, {"failsafe": True})
        {'backend': 'koboldcpp_failsafe.so', 'legacy': 'koboldcpp_legacy_ggml_failsafe.so', 'backend_matched': True, 'legacy_matched': True}

        Unknown features (None) fall back to the most general flavor:

        >>> select_isa_libraries(None, all_files, cpu)
        {'backend': 'koboldcpp_failsafe.so', 'legacy': 'koboldcpp_legacy_ggml_failsafe.so', 'backend_matched': False, 'legacy_matched': False}

        Nothing available:

        >>> select_isa_libraries({"avx2"}, [], cpu)
        {'backend': None, 'legacy': None, 'backend_matched': False, 'legacy_matched': False}

        The Vulkan family uses the same logic with its own names:

        >>> vulkan = {"default": "koboldcpp_vulkan", "noavx2": "koboldcpp_vulkan_noavx2", "failsafe": "koboldcpp_vulkan_failsafe"}
        >>> select_isa_libraries({"avx"}, ["koboldcpp_vulkan.so", "koboldcpp_vulkan_noavx2.so", "koboldcpp_vulkan_failsafe.so",
        ...                              "koboldcpp_legacy_ggml.so", "koboldcpp_legacy_ggml_noavx2.so", "koboldcpp_legacy_ggml_failsafe.so"], vulkan)
        {'backend': 'koboldcpp_vulkan_noavx2.so', 'legacy': 'koboldcpp_legacy_ggml_noavx2.so', 'backend_matched': True, 'legacy_matched': True}

        Windows .dll names:

        >>> select_isa_libraries({"avx2"}, ["koboldcpp_default.dll", "koboldcpp_legacy_ggml.dll"], cpu)
        {'backend': 'koboldcpp_default.dll', 'legacy': 'koboldcpp_legacy_ggml.dll', 'backend_matched': True, 'legacy_matched': True}
        """
        globs = {"select_isa_libraries": koboldcpp.select_isa_libraries}
        test = doctest.DocTestParser().get_doctest(
            IsaSelectionTests.test_select_isa_libraries.__doc__, globs,
            "select_isa_libraries", "<doctest>", 0)
        runner = doctest.DocTestRunner(verbose=False)
        runner.run(test)
        self.assertEqual(runner.failures, 0, "doctest failures in select_isa_libraries examples")


if __name__ == "__main__":
    unittest.main()
