"""GPU/CPU device detection and TensorFlow configuration."""

from __future__ import annotations

import logging
import os
from dataclasses import asdict, dataclass

from portautomation.gpu_env import ensure_gpu_environment

ensure_gpu_environment()

import tensorflow as tf

logger = logging.getLogger(__name__)


@dataclass
class DeviceInfo:
    using_gpu: bool
    device_type: str
    device_name: str
    gpu_count: int
    cuda_built: bool
    message: str


def _logical_device_name(gpu) -> str:
    name = gpu.name
    if name.startswith("/physical_device:"):
        return name.replace("/physical_device:", "/")
    return name


def _physical_gpus() -> list:
    return list(tf.config.list_physical_devices("GPU"))


def configure_devices(
    *,
    memory_growth: bool = True,
    allow_soft_placement: bool = True,
) -> DeviceInfo:
    """Enable GPU when available; otherwise use CPU."""
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "1")
    if allow_soft_placement:
        tf.config.set_soft_device_placement(True)

    gpus = _physical_gpus()
    if gpus:
        try:
            for gpu in gpus:
                if memory_growth:
                    tf.config.experimental.set_memory_growth(gpu, True)
            logical_name = _logical_device_name(gpus[0])
            with tf.device(logical_name):
                _ = tf.nn.relu(tf.constant(1.0))
                # Verify conv ops work; some GPU/cuDNN combos fail here.
                sample = tf.zeros((1, 32, 32, 3))
                kernel = tf.zeros((3, 3, 3, 8))
                _ = tf.nn.conv2d(sample, kernel, strides=[1, 1, 1, 1], padding="SAME")
            message = f"Using GPU acceleration ({len(gpus)} device(s) detected)."
            logger.info(message)
            logger.info("Primary GPU: %s", logical_name)
            return DeviceInfo(
                using_gpu=True,
                device_type="GPU",
                device_name=logical_name,
                gpu_count=len(gpus),
                cuda_built=tf.test.is_built_with_cuda(),
                message=message,
            )
        except Exception as exc:
            logger.warning("GPU detected but unavailable, falling back to CPU: %s", exc)
            message = (
                f"GPU detected ({len(gpus)} device(s)) but not usable; using CPU. "
                f"Reason: {exc}"
            )
            logger.info(message)
            return DeviceInfo(
                using_gpu=False,
                device_type="CPU",
                device_name="/CPU:0",
                gpu_count=len(gpus),
                cuda_built=tf.test.is_built_with_cuda(),
                message=message,
            )

    message = "No GPU detected; using CPU."
    logger.info(message)
    return DeviceInfo(
        using_gpu=False,
        device_type="CPU",
        device_name="/CPU:0",
        gpu_count=0,
        cuda_built=tf.test.is_built_with_cuda(),
        message=message,
    )


def get_device_info() -> DeviceInfo:
    """Return current device availability without changing configuration."""
    gpus = _physical_gpus()
    if gpus:
        logical_name = _logical_device_name(gpus[0])
        return DeviceInfo(
            using_gpu=True,
            device_type="GPU",
            device_name=logical_name,
            gpu_count=len(gpus),
            cuda_built=tf.test.is_built_with_cuda(),
            message=f"GPU available ({len(gpus)} device(s)).",
        )
    return DeviceInfo(
        using_gpu=False,
        device_type="CPU",
        device_name="/CPU:0",
        gpu_count=0,
        cuda_built=tf.test.is_built_with_cuda(),
        message="Running on CPU.",
    )


def optimize_dataset(dataset: tf.data.Dataset, *, for_training: bool) -> tf.data.Dataset:
    """Apply performance options, including GPU-friendly prefetching."""
    if for_training:
        dataset = dataset.cache()
    options = tf.data.Options()
    options.experimental_distribute.auto_shard_policy = (
        tf.data.experimental.AutoShardPolicy.DATA
    )
    dataset = dataset.with_options(options)
    return dataset.prefetch(tf.data.AUTOTUNE)


def device_dict() -> dict:
    return asdict(get_device_info())
