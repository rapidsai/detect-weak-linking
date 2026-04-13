# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

import re
import subprocess
import tempfile
from pathlib import Path

import pytest

TESTS_DIR = Path(__file__).parent
SCRIPT = TESTS_DIR.parent / "kernel_sizes.py"


def nvcc_compile(cu_file: Path, output: Path):
    subprocess.run(
        [
            "nvcc",
            "--shared",
            "-Xcompiler",
            "-fPIC",
            str(cu_file),
            "-o",
            str(output),
        ],
        check=True,
        cwd=str(cu_file.parent),
    )


def run_kernel_sizes(*so_files, include=None):
    cmd = ["python", str(SCRIPT)]
    if include:
        cmd += ["-i"] + list(include) + ["--"]
    cmd += [str(f) for f in so_files]
    result = subprocess.run(cmd, capture_output=True, text=True)
    assert result.returncode == 0, f"kernel_sizes.py failed:\n{result.stderr}"
    return result.stdout


def parse_output(stdout):
    """Parse kernel_sizes.py output into {name: (bytes, instantiations)} and total."""
    entries = {}
    total = None
    for line in stdout.splitlines():
        m = re.match(r"^(.+):\s+(\d+)\s+bytes\s+\((\d+)\s+instantiations\)$", line)
        if m:
            entries[m.group(1)] = (int(m.group(2)), int(m.group(3)))
            continue
        m = re.match(
            r"^Total uncompressed size of CUDA kernels:\s+(\d+)\s+bytes$", line
        )
        if m:
            total = int(m.group(1))
    return entries, total


@pytest.fixture(scope="session")
def compiled_libs():
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        lib_a = tmpdir / "lib_a.so"
        lib_b = tmpdir / "lib_b.so"
        nvcc_compile(TESTS_DIR / "lib_a.cu", lib_a)
        nvcc_compile(TESTS_DIR / "lib_b.cu", lib_b)
        yield lib_a, lib_b


class TestSingleLibrary:
    def test_finds_kernels(self, compiled_libs):
        lib_a, _ = compiled_libs
        stdout = run_kernel_sizes(lib_a)
        entries, total = parse_output(stdout)
        assert "kernel_only_in_a" in " ".join(entries.keys())
        assert "shared_add" in " ".join(entries.keys())

    def test_nonzero_sizes(self, compiled_libs):
        lib_a, _ = compiled_libs
        stdout = run_kernel_sizes(lib_a)
        entries, total = parse_output(stdout)
        for name, (size, count) in entries.items():
            assert size > 0, f"{name} has zero size"

    def test_single_instantiation(self, compiled_libs):
        lib_a, _ = compiled_libs
        stdout = run_kernel_sizes(lib_a)
        entries, total = parse_output(stdout)
        for name, (size, count) in entries.items():
            assert count == 1, f"{name} expected 1 instantiation, got {count}"

    def test_total_is_sum_of_parts(self, compiled_libs):
        lib_a, _ = compiled_libs
        stdout = run_kernel_sizes(lib_a)
        entries, total = parse_output(stdout)
        assert total == sum(size for size, _ in entries.values())


class TestMultipleLibraries:
    def test_shared_kernel_counted_twice(self, compiled_libs):
        lib_a, lib_b = compiled_libs
        stdout = run_kernel_sizes(lib_a, lib_b)
        entries, total = parse_output(stdout)
        shared = {k: v for k, v in entries.items() if "shared_add" in k}
        assert len(shared) == 1, f"Expected one shared_add entry, got {shared.keys()}"
        _, count = list(shared.values())[0]
        assert count == 2, f"shared_add expected 2 instantiations, got {count}"

    def test_unique_kernels_counted_once(self, compiled_libs):
        lib_a, lib_b = compiled_libs
        stdout = run_kernel_sizes(lib_a, lib_b)
        entries, total = parse_output(stdout)
        for name, (size, count) in entries.items():
            if "only_in_a" in name or "only_in_b" in name:
                assert count == 1, f"{name} expected 1 instantiation, got {count}"

    def test_all_three_kernels_present(self, compiled_libs):
        lib_a, lib_b = compiled_libs
        stdout = run_kernel_sizes(lib_a, lib_b)
        entries, _ = parse_output(stdout)
        all_names = " ".join(entries.keys())
        assert "kernel_only_in_a" in all_names
        assert "kernel_only_in_b" in all_names
        assert "shared_add" in all_names

    def test_total_is_sum_of_parts(self, compiled_libs):
        lib_a, lib_b = compiled_libs
        stdout = run_kernel_sizes(lib_a, lib_b)
        entries, total = parse_output(stdout)
        assert total == sum(size for size, _ in entries.values())


class TestInclusionFilter:
    def test_include_filters_output(self, compiled_libs):
        lib_a, lib_b = compiled_libs
        stdout = run_kernel_sizes(lib_a, lib_b, include=["only_in_a"])
        entries, _ = parse_output(stdout)
        all_names = " ".join(entries.keys())
        assert "kernel_only_in_a" in all_names
        assert "kernel_only_in_b" not in all_names
        assert "shared_add" not in all_names

    def test_include_regex(self, compiled_libs):
        lib_a, lib_b = compiled_libs
        stdout = run_kernel_sizes(lib_a, lib_b, include=["only_in_[ab]"])
        entries, _ = parse_output(stdout)
        all_names = " ".join(entries.keys())
        assert "kernel_only_in_a" in all_names
        assert "kernel_only_in_b" in all_names
        assert "shared_add" not in all_names
