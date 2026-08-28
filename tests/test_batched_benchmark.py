import ctypes
import os
import sys
import types
import unittest
from unittest import mock


parent_dir = os.path.abspath(os.path.join(__file__, "..", ".."))
sys.path.append(parent_dir)

import koboldcpp


EXPECTED_OUTPUT_FIELDS = (
    "status",
    "jobs",
    "slots",
    "successes",
    "failures",
    "engine_context",
    "prompt_tokens_min",
    "prompt_tokens_max",
    "first_failure_request",
    "first_failure_stopreason",
    "total_prompt_tokens",
    "total_generated_tokens",
    "wall_seconds",
    "cohort_e2e_generated_tps",
    "requests_per_second",
    "jobs_per_hour",
    "latency_p50_ms",
    "latency_p95_ms",
    "latency_max_ms",
)


class BatchedBenchmarkABITests(unittest.TestCase):
    def test_ctypes_output_layout_is_fixed(self):
        self.assertEqual(
            tuple(name for name, _ in koboldcpp.batch_benchmark_outputs._fields_),
            EXPECTED_OUTPUT_FIELDS,
        )
        self.assertEqual(
            tuple(field_type for _, field_type in koboldcpp.batch_benchmark_outputs._fields_),
            (ctypes.c_int,) * 10 + (ctypes.c_int64,) * 2 + (ctypes.c_double,) * 7,
        )
        self.assertEqual(ctypes.sizeof(koboldcpp.batch_benchmark_outputs), 112)
        self.assertEqual(
            tuple(getattr(koboldcpp.batch_benchmark_outputs, name).offset for name in EXPECTED_OUTPUT_FIELDS),
            tuple(range(0, 40, 4)) + (40, 48) + tuple(range(56, 112, 8)),
        )

    def test_private_generate_path_calls_native_runner_once(self):
        captured = {"calls": 0}

        class FakeHandle:
            def batch_benchmark(self, generation, jobs):
                captured.update(
                    calls=captured["calls"] + 1,
                    generation_type=type(generation),
                    prompt=generation.prompt,
                    max_context_length=generation.max_context_length,
                    max_length=generation.max_length,
                    temperature=generation.temperature,
                    allow_eos_token=generation.allow_eos_token,
                    sampler_len=generation.sampler_len,
                    jobs=jobs,
                )
                output = koboldcpp.batch_benchmark_outputs()
                output.status = 1
                output.jobs = jobs
                output.prompt_tokens_max = 7680
                return output

        params = {
            "prompt": "overlong prompt",
            "max_context_length": 8192,
            "max_length": 512,
            "temperature": 0.0,
            "top_k": 0,
            "top_p": 1.0,
            "sampler_order": [],
            "ban_eos_token": True,
            "_batch_benchmark": {
                "_sentinel": koboldcpp._BATCH_BENCHMARK_SENTINEL,
                "jobs": 8,
            },
        }
        fake_args = types.SimpleNamespace(defaultgenamt=100, genlimit=0)
        with mock.patch.object(koboldcpp, "args", fake_args), \
                mock.patch.object(koboldcpp, "maxctx", 40960), \
                mock.patch.object(koboldcpp, "handle", FakeHandle()):
            result = koboldcpp.generate(params)

        self.assertEqual(result["status"], 1)
        self.assertEqual(captured["calls"], 1)
        self.assertIs(captured["generation_type"], koboldcpp.generation_inputs)
        self.assertEqual(captured["prompt"], b"overlong prompt")
        self.assertEqual(captured["max_context_length"], 8192)
        self.assertEqual(captured["max_length"], 512)
        self.assertEqual(captured["temperature"], 0.0)
        self.assertFalse(captured["allow_eos_token"])
        self.assertEqual(captured["sampler_len"], 0)
        self.assertEqual(captured["jobs"], 8)


class BatchedBenchmarkWorkloadTests(unittest.TestCase):
    def test_bare_batched_flag_defaults_jobs_to_slots(self):
        workload = koboldcpp.resolve_batched_benchmark_workload(40960, 4, 0, 0, 512)
        self.assertEqual(workload["jobs"], 4)

    def test_default_context_uses_active_slots(self):
        workload = koboldcpp.resolve_batched_benchmark_workload(40960, 4, 8, 0, 512)
        self.assertEqual(workload["active_jobs"], 4)
        self.assertEqual(workload["context_per_job"], 10240)

    def test_jobs_below_slots_only_reserve_jobs(self):
        workload = koboldcpp.resolve_batched_benchmark_workload(40960, 20, 4, 0, 512)
        self.assertEqual(workload["active_jobs"], 4)
        self.assertEqual(workload["context_per_job"], 10240)

    def test_explicit_context_cannot_overcommit_active_slots(self):
        with self.assertRaisesRegex(ValueError, "shared context"):
            koboldcpp.resolve_batched_benchmark_workload(40960, 4, 8, 12000, 512)

    def test_output_must_fit_without_generate_clamping(self):
        with self.assertRaisesRegex(ValueError, "too large"):
            koboldcpp.resolve_batched_benchmark_workload(1024, 2, 2, 512, 450)

    def test_prompt_builder_grows_until_token_count_exceeds_context(self):
        with mock.patch.object(koboldcpp, "tokenize_ids", side_effect=lambda prompt, _special: range(prompt.count(" 1") // 4)):
            prompt = koboldcpp.build_batched_benchmark_prompt(512)
        self.assertGreater(prompt.count(" 1") // 4, 512)


class BatchedBenchmarkReportingTests(unittest.TestCase):
    def make_summary(self, status=1):
        summary = {
            name: (0.0 if field_type is ctypes.c_double else 0)
            for name, field_type in koboldcpp.batch_benchmark_outputs._fields_
        }
        summary.update(
            status=status,
            jobs=8,
            slots=4,
            successes=8 if status == 1 else 7,
            failures=0 if status == 1 else 1,
            engine_context=40960,
            prompt_tokens_min=9940,
            prompt_tokens_max=9940,
            total_prompt_tokens=79520,
            total_generated_tokens=2400,
            wall_seconds=10.0,
            cohort_e2e_generated_tps=240.0,
            requests_per_second=0.8,
            jobs_per_hour=2880.0,
            latency_p50_ms=5000.0,
            latency_p95_ms=10000.0,
            latency_max_ms=10000.0,
        )
        return summary

    def test_success_output_describes_native_scope_and_overflow(self):
        workload = koboldcpp.resolve_batched_benchmark_workload(40960, 4, 8, 10240, 300)
        lines = koboldcpp.format_batched_benchmark_lines(self.make_summary(), workload)
        output = "\n".join(lines)
        self.assertIn("status=PASS", output)
        self.assertIn("overflow_jobs=4", output)
        self.assertIn("minimum_waves=2", output)
        self.assertIn("pp_tg_source=BatchRequest_lines", output)
        self.assertIn("cohort_latency_queue_included=true", output)

    def test_failure_output_includes_native_request_and_stop_reason(self):
        workload = koboldcpp.resolve_batched_benchmark_workload(40960, 4, 8, 10240, 300)
        summary = self.make_summary(status=-5)
        summary["first_failure_request"] = 42
        summary["first_failure_stopreason"] = -2
        output = "\n".join(koboldcpp.format_batched_benchmark_lines(summary, workload))
        self.assertIn("status=FAIL code=-5 reason=request_failed", output)
        self.assertIn("first_failure_request=42", output)
        self.assertIn("first_failure_stopreason=-2", output)

    def test_config_snapshot_is_reproducible_and_omits_credentials(self):
        launch_args = types.SimpleNamespace(
            batchsize=512, blasthreads=8, device="CUDA0", noflashattention=False,
            autofit=False, gpulayers=99, maingpu=0, usemmap=True, usemlock=False,
            nommq=True, parallelrequests=4, nopipelineparallel=False, quantkv="q8_0",
            splitmode="layer", tensor_split=[1.0], threads=12, password="secret")
        snapshot = koboldcpp.batched_benchmark_config_snapshot(launch_args, 40960)
        self.assertEqual(snapshot["context_size"], 40960)
        self.assertEqual(snapshot["parallel_requests"], 4)
        self.assertFalse(snapshot["mmq"])
        self.assertNotIn("password", snapshot)


if __name__ == "__main__":
    unittest.main()
