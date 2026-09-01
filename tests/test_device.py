"""Tests for GPU environment and device configuration."""

from portautomation.gpu_env import discover_nvidia_lib_dirs, setup_nvidia_library_path


def test_discover_nvidia_lib_dirs_finds_cuda_libs():
    lib_dirs = discover_nvidia_lib_dirs()
    assert lib_dirs
    assert any("cudnn" in path or "cuda_runtime" in path for path in lib_dirs)


def test_setup_nvidia_library_path_updates_env():
    lib_dirs = setup_nvidia_library_path()
    assert lib_dirs
    assert "LD_LIBRARY_PATH" in __import__("os").environ
