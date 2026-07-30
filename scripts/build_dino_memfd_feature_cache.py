#!/usr/bin/env python3
"""Build and serve an in-RAM, frozen-DINO feature cache for ACT training.

This is intentionally a *server*, rather than an offline file converter.  A
``memfd`` is backed by host RAM (and, if enabled, swap), and has no durable
filesystem name.  While this process is alive, another process in the same
PID namespace can open the cache through the ``cache_path`` written to the
JSON manifest, for example::

    /proc/<server-pid>/fd/<fd>

The cache stores the output immediately before ACT's trainable 1x1 visual
projection, so it skips only the frozen DINO backbone.  It does *not* freeze
or precompute ACT's visual projection, transformer, VAE, or action head.

The Robot8 cache layout is camera-count agnostic::

    [num_frames, num_cameras, 768 channels, 30, 40]  float16

That is roughly 256.9 GiB for 49,876 frames.  This script checks available
memory before allocating it, starts one worker per requested CUDA device, and
keeps the descriptor open until SIGTERM/SIGINT.  A manifest is marked
``ready: true`` only after every worker finishes and sampled cache values have
been checked.

Run with the LeRobot environment and source tree on ``PYTHONPATH``.  A typical
remote launch is documented in the repository's training wrapper; this module
is deliberately self-contained so it can also be supervised separately.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import mmap
import os
import queue
import signal
import sys
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any


INPUT_HEIGHT = 480
INPUT_WIDTH = 640
INPUT_CHANNELS = 3
FEATURE_CHANNELS = 768
FEATURE_HEIGHT = 30
FEATURE_WIDTH = 40
DTYPE_NAME = "float16"
DTYPE_BYTES = 2
MIN_CAMERA_COUNT = 1
MAX_CAMERA_COUNT = 8
DEFAULT_CACHE_DEVICES = (0, 1)
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
_FILE_SHA256_CACHE: dict[tuple[Path, int, int], str] = {}


class CacheBuildError(RuntimeError):
    """A recoverable configuration, dataset, or cache-build failure."""


class CacheBuildCancelled(CacheBuildError):
    """The parent requested a clean stop before the cache became ready."""


@dataclass(frozen=True)
class DatasetValidation:
    root: Path
    repo_id: str
    total_frames: int
    camera_keys: tuple[str, ...]
    info_sha256: str
    source_fingerprint: str


@dataclass(frozen=True)
class CacheSelection:
    """A contiguous cache axis mapped to a contiguous absolute-index interval."""

    absolute_start: int
    absolute_end: int

    @property
    def frame_count(self) -> int:
        return self.absolute_end - self.absolute_start


def _utc_now() -> str:
    return dt.datetime.now(dt.UTC).isoformat().replace("+00:00", "Z")


def _scalar(value: Any) -> int:
    """Return an integer from Python, NumPy, Arrow, or zero-dimensional torch values."""
    if hasattr(value, "item"):
        value = value.item()
    return int(value)


def _parse_camera_keys(value: str) -> tuple[str, ...]:
    names = tuple(part.strip() for part in value.split(",") if part.strip())
    if not MIN_CAMERA_COUNT <= len(names) <= MAX_CAMERA_COUNT:
        raise argparse.ArgumentTypeError(
            f"cache supports {MIN_CAMERA_COUNT}..{MAX_CAMERA_COUNT} cameras, comma-separated"
        )

    keys: list[str] = []
    for name in names:
        key = name if name.startswith("observation.images.") else f"observation.images.{name}"
        keys.append(key)
    if len(set(keys)) != len(keys):
        raise argparse.ArgumentTypeError("camera names must be unique")
    return tuple(keys)


def _parse_devices(value: str) -> tuple[int, ...]:
    raw = tuple(part.strip() for part in value.split(",") if part.strip())
    if not 1 <= len(raw) <= len(DEFAULT_CACHE_DEVICES):
        raise argparse.ArgumentTypeError("one or two CUDA device indices are required, e.g. 0 or 0,1")
    try:
        devices = tuple(int(part) for part in raw)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("CUDA device indices must be integers") from exc
    if min(devices) < 0 or len(set(devices)) != len(devices):
        raise argparse.ArgumentTypeError("CUDA device indices must be distinct non-negative integers")
    return devices


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create a one- or two-GPU, memfd-backed fp16 DINO feature-map cache and keep it alive for ACT training."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--dataset-root", type=Path, required=True, help="Local LeRobot dataset directory.")
    parser.add_argument(
        "--repo-id",
        default=None,
        help="LeRobot repository identifier. Defaults to the dataset-root directory name.",
    )
    parser.add_argument(
        "--model",
        default="vit_base_patch16_dinov3.lvd1689m",
        help="timm DINO model name used by the ACT run.",
    )
    parser.add_argument(
        "--cameras",
        type=_parse_camera_keys,
        default=_parse_camera_keys("head_cam,left_arm_cam,right_arm_cam"),
        help="Camera names or full observation.images.* keys, in ACT camera order.",
    )
    parser.add_argument("--manifest", type=Path, required=True, help="JSON manifest consumed by the trainer.")
    parser.add_argument(
        "--devices",
        type=_parse_devices,
        default=DEFAULT_CACHE_DEVICES,
        help="One or two visible CUDA device indices, one per cache worker.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Samples per cache worker batch (each sample includes every requested camera).",
    )
    parser.add_argument(
        "--video-backend",
        choices=("pyav", "torchcodec"),
        default="pyav",
        help=(
            "Video decoder used only while constructing the frozen-DINO cache. "
            "torchcodec keeps CPU decoders open across samples and is faster for "
            "the contiguous robot8 cache pass; pyav remains available for parity/debugging."
        ),
    )
    parser.add_argument(
        "--pretrained-weights",
        type=Path,
        default=None,
        help="Optional local DINO checkpoint; otherwise timm's pretrained model cache is used.",
    )
    pretrained_group = parser.add_mutually_exclusive_group()
    pretrained_group.add_argument(
        "--pretrained", dest="pretrained", action="store_true", default=True, help="Use timm pretrained weights."
    )
    pretrained_group.add_argument(
        "--no-pretrained", dest="pretrained", action="store_false", help="Do not load timm pretrained weights."
    )
    parser.add_argument(
        "--min-free-gib",
        type=float,
        default=64.0,
        help="Required MemAvailable headroom after the cache allocation; set deliberately if overriding.",
    )
    parser.add_argument(
        "--progress-every-batches",
        type=int,
        default=10,
        help="Worker progress report interval.",
    )
    parser.add_argument(
        "--start-frame",
        type=int,
        default=0,
        help="First absolute LeRobot frame index to cache. Intended primarily for small-path tests.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help=(
            "Cache at most this many contiguous frames starting at --start-frame. "
            "Omit for the full dataset; useful for a small end-to-end validation."
        ),
    )
    parser.add_argument(
        "--replace-stale-manifest",
        action="store_true",
        help="Replace a manifest only if its recorded owner PID is no longer alive.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate dataset/index/memory requirements but do not allocate a memfd or start workers.",
    )
    args = parser.parse_args(argv)
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")
    if args.min_free_gib < 0:
        parser.error("--min-free-gib must be non-negative")
    if args.progress_every_batches < 1:
        parser.error("--progress-every-batches must be positive")
    if args.start_frame < 0:
        parser.error("--start-frame must be non-negative")
    if args.max_frames is not None and args.max_frames < 1:
        parser.error("--max-frames must be positive when supplied")
    return args


def _read_host_mem_available_bytes() -> int | None:
    """Read host ``MemAvailable`` without importing a heavyweight library."""
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) * 1024
    except (FileNotFoundError, ValueError, IndexError):
        return None
    return None


def _read_cgroup_mem_available_bytes() -> int | None:
    """Return this process's cgroup-v2 RAM headroom when it is finite.

    ``/proc/meminfo`` describes the host, which can be much larger than the
    container allocation.  A memfd is charged to the process's cgroup, so a
    host-only check could approve an allocation that later OOM-kills the cache
    workers.  In an unlimited cgroup, ``memory.max`` is the literal ``max``;
    in that case this function deliberately returns ``None`` and callers can
    fall back to the host value.
    """
    try:
        cgroup_root = Path("/sys/fs/cgroup")
        cgroup_rel: str | None = None
        for line in Path("/proc/self/cgroup").read_text().splitlines():
            hierarchy, controllers, relative_path = line.split(":", 2)
            if hierarchy == "0" and controllers == "":
                cgroup_rel = relative_path
                break

        # Most containers expose the process cgroup as the namespace root.
        # Retain the direct-root fallback for older/simple cgroup namespaces.
        cgroup_path = cgroup_root / cgroup_rel.lstrip("/") if cgroup_rel else cgroup_root
        max_path = cgroup_path / "memory.max"
        current_path = cgroup_path / "memory.current"
        if not max_path.is_file() or not current_path.is_file():
            max_path = cgroup_root / "memory.max"
            current_path = cgroup_root / "memory.current"
        limit_text = max_path.read_text().strip()
        if limit_text == "max":
            return None
        limit = int(limit_text)
        current = int(current_path.read_text().strip())
        if limit <= 0:
            return None
        return max(0, limit - current)
    except (FileNotFoundError, OSError, ValueError):
        # Non-Linux/cgroup-v1 hosts retain the existing host-memory behavior.
        return None


def _read_mem_available_bytes() -> tuple[int | None, int | None, int | None]:
    """Return effective, host, and finite-cgroup memory headroom in bytes."""
    host_available = _read_host_mem_available_bytes()
    cgroup_available = _read_cgroup_mem_available_bytes()
    candidates = [value for value in (host_available, cgroup_available) if value is not None]
    effective_available = min(candidates) if candidates else None
    return effective_available, host_available, cgroup_available


def _format_gib(byte_count: int | float) -> str:
    return f"{byte_count / (1024**3):.2f} GiB"


def _sha256_file(path: Path) -> str:
    stat = path.stat()
    key = (path.resolve(), stat.st_size, stat.st_mtime_ns)
    cached = _FILE_SHA256_CACHE.get(key)
    if cached is not None:
        return cached
    digest = hashlib.sha256()
    with path.open("rb") as file:
        while True:
            block = file.read(1024 * 1024)
            if not block:
                break
            digest.update(block)
    value = digest.hexdigest()
    _FILE_SHA256_CACHE[key] = value
    return value


def _source_fingerprint(root: Path, camera_keys: tuple[str, ...]) -> str:
    """Cheap source identity that catches metadata/video/data replacement.

    The feature cache is ephemeral, so an mtime/size fingerprint is enough to
    prevent accidental reuse across a re-conversion without spending minutes
    hashing the H.264 files themselves.
    """
    digest = hashlib.sha256()
    paths: list[Path] = [root / "meta" / "info.json", root / "meta" / "stats.json"]
    paths.extend(sorted((root / "data").glob("*/*.parquet")))
    for key in camera_keys:
        paths.extend(sorted((root / "videos" / key).glob("*/*.mp4")))
    for path in paths:
        if not path.is_file():
            raise CacheBuildError(f"dataset source file is missing: {path}")
        stat = path.stat()
        digest.update(path.relative_to(root).as_posix().encode())
        digest.update(b"\0")
        digest.update(str(stat.st_size).encode())
        digest.update(b"\0")
        digest.update(str(stat.st_mtime_ns).encode())
        digest.update(b"\n")
    return digest.hexdigest()


def validate_dataset(root: Path, repo_id: str, camera_keys: tuple[str, ...]) -> DatasetValidation:
    """Validate the exact raw inputs and dense absolute-index contract.

    The cache is addressed by absolute LeRobot ``index``.  Training assumes
    the same mapping, so accepting sparse, reordered, or filtered rows here
    would silently feed a different feature map to a sample.  Check every
    Parquet index before we reserve hundreds of GiB of RAM.
    """
    root = root.expanduser().resolve()
    info_path = root / "meta" / "info.json"
    if not info_path.is_file():
        raise CacheBuildError(f"missing LeRobot metadata: {info_path}")
    try:
        info_bytes = info_path.read_bytes()
        info = json.loads(info_bytes)
    except (OSError, json.JSONDecodeError) as exc:
        raise CacheBuildError(f"could not read {info_path}: {exc}") from exc

    try:
        total_frames = int(info["total_frames"])
        features = info["features"]
    except (KeyError, TypeError, ValueError) as exc:
        raise CacheBuildError("meta/info.json is missing valid total_frames/features") from exc
    if total_frames <= 0:
        raise CacheBuildError(f"dataset has no frames: {total_frames}")

    expected_image_shape = [INPUT_HEIGHT, INPUT_WIDTH, INPUT_CHANNELS]
    for key in camera_keys:
        feature = features.get(key)
        if not isinstance(feature, dict):
            raise CacheBuildError(f"requested camera is absent from metadata: {key}")
        if feature.get("dtype") != "video":
            raise CacheBuildError(f"requested camera is not a video feature: {key}")
        if feature.get("shape") != expected_image_shape:
            raise CacheBuildError(
                f"{key} must be raw 640x480 RGB (metadata shape {expected_image_shape}), "
                f"got {feature.get('shape')!r}"
            )

    data_paths = sorted((root / "data").glob("*/*.parquet"))
    if not data_paths:
        raise CacheBuildError(f"no Parquet data files found under {root / 'data'}")

    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise CacheBuildError("pyarrow is required; launch with the LeRobot virtual environment") from exc

    expected_index = 0
    for parquet_path in data_paths:
        try:
            parquet = pq.ParquetFile(parquet_path)
            for batch in parquet.iter_batches(columns=["index"], batch_size=65_536):
                values = batch.column(0).to_pylist()
                for value in values:
                    actual_index = _scalar(value)
                    if actual_index != expected_index:
                        raise CacheBuildError(
                            "LeRobot absolute index must be dense and ordered for this cache: "
                            f"expected {expected_index}, got {actual_index} in {parquet_path}"
                        )
                    expected_index += 1
        except CacheBuildError:
            raise
        except Exception as exc:
            raise CacheBuildError(f"could not validate index column in {parquet_path}: {exc}") from exc
    if expected_index != total_frames:
        raise CacheBuildError(
            f"metadata says total_frames={total_frames}, but Parquet contains {expected_index} indexed rows"
        )

    return DatasetValidation(
        root=root,
        repo_id=repo_id,
        total_frames=total_frames,
        camera_keys=camera_keys,
        info_sha256=hashlib.sha256(info_bytes).hexdigest(),
        source_fingerprint=_source_fingerprint(root, camera_keys),
    )


def _owner_is_alive(pid: int | None) -> bool:
    if pid is None or pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        # Treat another user's process as alive: overwriting its manifest would
        # be unsafe even if this invocation cannot inspect it further.
        return True
    return True


def _check_manifest_target(path: Path, replace_stale: bool) -> None:
    if not path.exists():
        return
    try:
        old = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        if not replace_stale:
            raise CacheBuildError(
                f"manifest already exists and is not valid JSON: {path}; choose a new path or pass --replace-stale-manifest"
            )
        return
    old_pid = old.get("parent_pid", old.get("pid")) if isinstance(old, dict) else None
    try:
        old_pid = int(old_pid) if old_pid is not None else None
    except (TypeError, ValueError):
        old_pid = None
    if _owner_is_alive(old_pid):
        raise CacheBuildError(
            f"refusing to overwrite manifest owned by live PID {old_pid}: {path}. "
            "Stop that cache server first or use a different manifest path."
        )
    if not replace_stale:
        raise CacheBuildError(
            f"stale manifest already exists: {path}; pass --replace-stale-manifest after verifying its owner is gone"
        )


def _write_manifest(path: Path, payload: dict[str, Any]) -> None:
    """Atomically replace the small JSON control plane with owner-only permissions."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with os.fdopen(os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600), "w") as file:
            json.dump(payload, file, indent=2, sort_keys=True)
            file.write("\n")
            file.flush()
            os.fsync(file.fileno())
        os.replace(tmp, path)
    finally:
        try:
            tmp.unlink()
        except FileNotFoundError:
            pass


def _create_memfd(size_bytes: int) -> int:
    if sys.platform != "linux":
        raise CacheBuildError("memfd feature caching requires Linux")
    try:
        # Some otherwise fully capable conda-forge Python builds omit the
        # optional ``os.memfd_create`` wrapper (notably the Python 3.12 image
        # used by the 2026 Vast instance).  Linux/glibc still exposes the
        # syscall, so use it directly as a narrow, equivalent fallback rather
        # than falling back to a durable on-disk cache or disabling caching.
        if hasattr(os, "memfd_create"):
            fd = os.memfd_create("robot8_dinov3_feature_cache", flags=0)
        else:
            import ctypes
            import errno

            libc = ctypes.CDLL(None, use_errno=True)
            memfd_create = libc.memfd_create
            memfd_create.argtypes = [ctypes.c_char_p, ctypes.c_uint]
            memfd_create.restype = ctypes.c_int
            fd = int(memfd_create(b"robot8_dinov3_feature_cache", 0))
            if fd < 0:
                error_number = ctypes.get_errno()
                raise OSError(error_number, os.strerror(error_number or errno.ENOSYS))
        # Keep it inheritable even if a Python/platform default changes.  The
        # cache workers use fork, and the trainer later opens /proc/<pid>/fd/N.
        os.set_inheritable(fd, True)
        os.ftruncate(fd, size_bytes)
        actual = os.fstat(fd).st_size
        if actual != size_bytes:
            raise CacheBuildError(f"memfd truncated to {actual} bytes, expected {size_bytes}")
        return fd
    except Exception:
        try:
            os.close(fd)  # type: ignore[name-defined]
        except (NameError, OSError):
            pass
        raise


def _cache_shape(total_frames: int, camera_count: int) -> tuple[int, int, int, int, int]:
    if not MIN_CAMERA_COUNT <= camera_count <= MAX_CAMERA_COUNT:
        raise CacheBuildError(
            f"cache supports {MIN_CAMERA_COUNT}..{MAX_CAMERA_COUNT} cameras, got {camera_count}"
        )
    return (total_frames, camera_count, FEATURE_CHANNELS, FEATURE_HEIGHT, FEATURE_WIDTH)


def _cache_byte_count(shape: tuple[int, ...]) -> int:
    return math.prod(shape) * DTYPE_BYTES


def _select_cache_interval(validation: DatasetValidation, args: argparse.Namespace) -> CacheSelection:
    start = int(args.start_frame)
    if start >= validation.total_frames:
        raise CacheBuildError(
            f"--start-frame={start} is outside the dataset's 0..{validation.total_frames - 1} absolute index range"
        )
    if args.max_frames is None:
        end = validation.total_frames
    else:
        end = start + int(args.max_frames)
        if end > validation.total_frames:
            raise CacheBuildError(
                f"requested [{start}, {end}) exceeds the dataset's {validation.total_frames} frames"
            )
    return CacheSelection(absolute_start=start, absolute_end=end)


def _make_manifest(
    *,
    validation: DatasetValidation,
    args: argparse.Namespace,
    shape: tuple[int, int, int, int, int],
    selection: CacheSelection,
    cache_path: str,
    status: str,
    ready: bool,
    byte_count: int,
    workers: dict[str, Any],
    error: str | None = None,
) -> dict[str, Any]:
    model_weights: dict[str, Any] | None = None
    if args.pretrained_weights is not None:
        # Preserve a snapshot's ``model.safetensors`` symlink name. Resolving
        # it produces an extension-less HF blob and loses the format signal.
        weights = args.pretrained_weights.expanduser().absolute()
        model_weights = {
            "path": str(weights),
            "sha256": _sha256_file(weights),
        }

    payload: dict[str, Any] = {
        "version": 1,
        "schema_version": 1,
        "status": status,
        "ready": ready,
        "created_at": _utc_now(),
        "parent_pid": os.getpid(),
        "pid": os.getpid(),  # Friendly alias for simple operational tooling.
        # The cache loader consumes these top-level fields.  fd_path is an
        # explicit alias retained for debugging; cache_path is the contract.
        "cache_path": cache_path,
        "fd_path": cache_path,
        "shape": list(shape),
        "dtype": DTYPE_NAME,
        "byte_size": byte_count,
        "camera_keys": list(validation.camera_keys),
        "data_index": {
            "validated": True,
            "first": selection.absolute_start,
            "last": selection.absolute_end - 1,
            "count": selection.frame_count,
            # Axis 0 is local to this memfd; it maps densely to this exact
            # absolute dataset interval. Full cache mode is simply [0, N).
            "cache_axis0_absolute_start": selection.absolute_start,
            "cache_axis0_absolute_end": selection.absolute_end,
            "dataset_total_frames": validation.total_frames,
        },
        "dataset": {
            "repo_id": validation.repo_id,
            "root": str(validation.root),
            "total_frames": validation.total_frames,
            "info_sha256": validation.info_sha256,
            "source_fingerprint": validation.source_fingerprint,
        },
        "model": {
            "name": args.model,
            "pretrained": bool(args.pretrained),
            "pretrained_weights": model_weights,
            "feature_shape_per_camera": [FEATURE_CHANNELS, FEATURE_HEIGHT, FEATURE_WIDTH],
            "input_shape_per_camera": [INPUT_CHANNELS, INPUT_HEIGHT, INPUT_WIDTH],
            "input_normalization": {
                "source": "uint8/255 then ImageNet mean/std",
                "mean": list(IMAGENET_MEAN),
                "std": list(IMAGENET_STD),
                "epsilon": 1e-8,
            },
            "autocast_dtype": DTYPE_NAME,
        },
        "workers": workers,
    }
    if error is not None:
        payload["error"] = error
    return payload


def _load_batch(dataset: Any, indices: range, camera_keys: tuple[str, ...]) -> tuple[list[int], Any]:
    """Decode raw videos and stack B×camera uint8 tensors in deterministic order."""
    import torch

    samples = [dataset[idx] for idx in indices]
    absolute_indices = [_scalar(sample["index"]) for sample in samples]
    expected_indices = list(indices)
    if absolute_indices != expected_indices:
        raise CacheBuildError(
            "LeRobot dataset returned a row whose absolute index differs from the requested cache index: "
            f"expected {expected_indices[:3]}..., got {absolute_indices[:3]}..."
        )

    per_camera = []
    for key in camera_keys:
        images = torch.stack([sample[key] for sample in samples], dim=0)
        expected_shape = (len(samples), INPUT_CHANNELS, INPUT_HEIGHT, INPUT_WIDTH)
        if tuple(images.shape) != expected_shape:
            raise CacheBuildError(
                f"decoded {key} has shape {tuple(images.shape)}, expected {expected_shape}; "
                "the cache only supports un-cropped 640x480 RGB input"
            )
        if images.dtype != torch.uint8:
            raise CacheBuildError(
                f"decoded {key} has dtype {images.dtype}; expected uint8 before /255 normalization"
            )
        per_camera.append(images)
    # (B, cameras, C, H, W), with camera order exactly matching the manifest.
    return absolute_indices, torch.stack(per_camera, dim=1)


def _worker_main(
    rank: int,
    device_index: int,
    fd: int,
    shape: tuple[int, int, int, int, int],
    cache_absolute_start: int,
    index_start: int,
    index_end: int,
    *,
    dataset_root: str,
    repo_id: str,
    camera_keys: tuple[str, ...],
    model_name: str,
    pretrained: bool,
    pretrained_weights: str | None,
    video_backend: str,
    batch_size: int,
    progress_every_batches: int,
    event_queue: Any,
    stop_event: Any,
    flush_every_batches: int = 0,
    evict_after_flush: bool = False,
) -> None:
    """Populate a disjoint contiguous range of absolute frame indices on one GPU."""
    # The parent installs a graceful SIGTERM handler for its lifetime. Workers
    # inherit it through fork, but must restore the default so `terminate()`
    # cannot leave a CUDA worker stuck halfway through a long kernel.
    signal.signal(signal.SIGTERM, signal.SIG_DFL)
    signal.signal(signal.SIGINT, signal.SIG_IGN)

    # Do not move these imports to module scope: the parent must remain CUDA
    # untouched before it forks its cache workers.
    import gc

    import numpy as np
    import torch

    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.policies.act.vision_backbone import TimmDinoV2FeatureMap

    cache_map: mmap.mmap | None = None
    cache_array: Any = None
    try:
        if not torch.cuda.is_available():
            raise CacheBuildError("CUDA is unavailable in cache worker")
        if device_index >= torch.cuda.device_count():
            raise CacheBuildError(
                f"requested CUDA device {device_index}, but only {torch.cuda.device_count()} visible device(s) exist"
            )
        torch.cuda.set_device(device_index)
        device = torch.device(f"cuda:{device_index}")

        expected_bytes = _cache_byte_count(shape)
        if os.fstat(fd).st_size != expected_bytes:
            raise CacheBuildError(
                f"worker sees memfd size {os.fstat(fd).st_size}, expected {expected_bytes}"
            )
        cache_map = mmap.mmap(
            fd,
            expected_bytes,
            flags=mmap.MAP_SHARED,
            prot=mmap.PROT_READ | mmap.PROT_WRITE,
        )
        cache_array = np.ndarray(shape, dtype=np.float16, buffer=cache_map)
        if cache_array.shape != shape or cache_array.dtype != np.float16:
            raise CacheBuildError("mapped cache shape or dtype does not match its manifest contract")

        dataset = LeRobotDataset(
            repo_id,
            root=dataset_root,
            video_backend=video_backend,
            return_uint8=True,
        )
        if len(dataset) < cache_absolute_start + shape[0]:
            raise CacheBuildError(
                f"LeRobotDataset length changed after validation: {len(dataset)} cannot cover "
                f"[{cache_absolute_start}, {cache_absolute_start + shape[0]})"
            )

        backbone = TimmDinoV2FeatureMap(
            model_name,
            pretrained=pretrained,
            pretrained_weights=pretrained_weights,
            train_backbone=False,
            image_size=(INPUT_HEIGHT, INPUT_WIDTH),
        ).to(device)
        backbone.eval()
        if any(parameter.requires_grad for parameter in backbone.parameters()):
            raise CacheBuildError("DINO cache worker requires a frozen backbone")

        mean = torch.tensor(IMAGENET_MEAN, device=device, dtype=torch.float32).view(1, 1, 3, 1, 1)
        std = torch.tensor(IMAGENET_STD, device=device, dtype=torch.float32).view(1, 1, 3, 1, 1)
        event_queue.put(
            {
                "type": "started",
                "rank": rank,
                "device": device_index,
                "range": [index_start, index_end],
            }
        )

        written = 0
        batches = 0
        started_at = time.monotonic()
        for batch_start in range(index_start, index_end, batch_size):
            if stop_event.is_set():
                raise CacheBuildCancelled("parent requested cache shutdown")
            batch_end = min(batch_start + batch_size, index_end)
            absolute_indices, raw_images = _load_batch(
                dataset,
                range(batch_start, batch_end),
                camera_keys,
            )
            # This exactly follows lerobot_train.py: uint8 -> float32 / 255,
            # then visual MEAN_STD normalization using ImageNet stats.
            images = raw_images.to(device=device, dtype=torch.float32, non_blocking=False).div_(255.0)
            images.sub_(mean).div_(std + 1e-8)
            flattened_images = images.flatten(0, 1)
            with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.float16):
                feature_maps = backbone(flattened_images)["feature_map"]
            expected_feature_shape = (
                len(absolute_indices) * len(camera_keys),
                FEATURE_CHANNELS,
                FEATURE_HEIGHT,
                FEATURE_WIDTH,
            )
            if tuple(feature_maps.shape) != expected_feature_shape:
                raise CacheBuildError(
                    "DINO feature-map shape does not match the cache contract: "
                    f"got {tuple(feature_maps.shape)}, expected {expected_feature_shape}"
                )
            if not torch.isfinite(feature_maps).all():
                raise CacheBuildError(f"DINO produced non-finite feature values for frame {batch_start}")

            feature_maps = feature_maps.reshape(
                len(absolute_indices), len(camera_keys), FEATURE_CHANNELS, FEATURE_HEIGHT, FEATURE_WIDTH
            ).to(dtype=torch.float16)
            # The destination is a zero-copy torch view over a MAP_SHARED
            # backing file. ``copy_`` is synchronous here, so an acknowledged
            # range is visible before progress is reported.
            cache_start = batch_start - cache_absolute_start
            cache_end = batch_end - cache_absolute_start
            if cache_start < 0 or cache_end > shape[0]:
                raise CacheBuildError(
                    f"absolute cache range [{batch_start}, {batch_end}) is outside "
                    f"the mapped interval [{cache_absolute_start}, {cache_absolute_start + shape[0]})"
                )
            destination = torch.from_numpy(cache_array[cache_start:cache_end])
            destination.copy_(feature_maps, non_blocking=False)
            del destination, raw_images, images, flattened_images, feature_maps

            written += batch_end - batch_start
            batches += 1
            if flush_every_batches and (
                batches % flush_every_batches == 0 or batch_end == index_end
            ):
                # A full feature row is 7,372,800 bytes with four cameras,
                # hence every cache-row offset is page aligned. Incremental
                # flushing keeps a durable disk cache below a tight cgroup
                # memory limit instead of retaining its whole dirty working
                # set until the end of the pass.
                bytes_per_row = _cache_byte_count(shape[1:])
                offset = cache_start * bytes_per_row
                length = (cache_end - cache_start) * bytes_per_row
                cache_map.flush(offset, length)
                if evict_after_flush and hasattr(cache_map, "madvise"):
                    madv_dontneed = getattr(mmap, "MADV_DONTNEED", None)
                    if madv_dontneed is not None:
                        cache_map.madvise(madv_dontneed, offset, length)
            if batches % progress_every_batches == 0 or batch_end == index_end:
                event_queue.put(
                    {
                        "type": "progress",
                        "rank": rank,
                        "device": device_index,
                        "frames_written": written,
                        "frame_count": index_end - index_start,
                        "range": [index_start, index_end],
                        "elapsed_s": round(time.monotonic() - started_at, 3),
                    }
                )

        torch.cuda.synchronize(device)
        # Flush is not required for MAP_SHARED visibility, but makes the
        # lifecycle boundary explicit and catches mapping failures here.
        cache_map.flush()
        event_queue.put(
            {
                "type": "complete",
                "rank": rank,
                "device": device_index,
                "frames_written": written,
                "frame_count": index_end - index_start,
                "range": [index_start, index_end],
                "elapsed_s": round(time.monotonic() - started_at, 3),
            }
        )
    except BaseException as exc:
        event_queue.put(
            {
                "type": "error",
                "rank": rank,
                "device": device_index,
                "range": [index_start, index_end],
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(),
            }
        )
        raise
    finally:
        # Explicitly release NumPy's exported buffer before closing the mmap;
        # otherwise Python can raise BufferError and leave diagnostics unclear.
        del cache_array
        gc.collect()
        if cache_map is not None:
            cache_map.close()


def _drain_events(event_queue: Any, workers: dict[str, Any]) -> list[dict[str, Any]]:
    errors: list[dict[str, Any]] = []
    while True:
        try:
            event = event_queue.get_nowait()
        except queue.Empty:
            break
        rank = str(event.get("rank", "unknown"))
        workers[rank] = event
        if event.get("type") == "error":
            errors.append(event)
    return errors


def _verify_ready_cache(fd: int, shape: tuple[int, ...]) -> None:
    """Sample the completed cache without faulting all 250+ GiB into the parent."""
    import gc

    import numpy as np

    expected_bytes = _cache_byte_count(shape)
    if os.fstat(fd).st_size != expected_bytes:
        raise CacheBuildError("memfd size changed while workers were writing")
    cache_map = mmap.mmap(fd, expected_bytes, flags=mmap.MAP_SHARED, prot=mmap.PROT_READ)
    cache_array: Any = None
    try:
        cache_array = np.ndarray(shape, dtype=np.float16, buffer=cache_map)
        if cache_array.shape != shape or cache_array.dtype != np.float16:
            raise CacheBuildError("completed cache does not have the expected shape/dtype")
        positions = sorted({0, shape[0] // 2, shape[0] - 1})
        samples = cache_array[positions]
        if not np.isfinite(samples).all():
            raise CacheBuildError("completed cache contains non-finite sampled feature values")
        if float(np.max(np.abs(samples))) == 0.0:
            raise CacheBuildError("completed cache sampled as all zero; refusing to mark it ready")
    finally:
        del cache_array
        gc.collect()
        cache_map.close()


def _terminate_workers(workers: list[Any], stop_event: Any, grace_s: float = 30.0) -> None:
    stop_event.set()
    deadline = time.monotonic() + grace_s
    for worker in workers:
        remaining = max(0.0, deadline - time.monotonic())
        worker.join(timeout=remaining)
    for worker in workers:
        if worker.is_alive():
            worker.terminate()
    for worker in workers:
        worker.join(timeout=5)


def _print_plan(
    validation: DatasetValidation,
    selection: CacheSelection,
    shape: tuple[int, ...],
    byte_count: int,
    args: argparse.Namespace,
) -> None:
    available, host_available, cgroup_available = _read_mem_available_bytes()
    print("DINO memfd cache plan", flush=True)
    print(f"  dataset: {validation.root}", flush=True)
    print(f"  repo_id: {validation.repo_id}", flush=True)
    print(
        f"  frames: {selection.frame_count} cached from absolute interval "
        f"[{selection.absolute_start}, {selection.absolute_end}) / {validation.total_frames}",
        flush=True,
    )
    print(f"  cameras: {', '.join(validation.camera_keys)}", flush=True)
    print(f"  shape: {list(shape)} {DTYPE_NAME}", flush=True)
    print(f"  cache bytes: {byte_count:,} ({_format_gib(byte_count)})", flush=True)
    if available is not None:
        print(f"  effective RAM headroom: {_format_gib(available)}", flush=True)
    if host_available is not None:
        print(f"  host MemAvailable: {_format_gib(host_available)}", flush=True)
    if cgroup_available is not None:
        print(f"  cgroup RAM headroom: {_format_gib(cgroup_available)}", flush=True)
    print(f"  CUDA workers: {', '.join(str(device) for device in args.devices)}", flush=True)
    print(f"  manifest: {args.manifest.expanduser().resolve()}", flush=True)


def run(args: argparse.Namespace) -> int:
    if sys.platform != "linux":
        raise CacheBuildError("this memfd cache server is Linux-only")
    repo_id = args.repo_id or args.dataset_root.expanduser().resolve().name
    if args.pretrained_weights is not None:
        # Do not resolve HF's ``model.safetensors`` symlink: the target blob
        # has no suffix, whereas the loader selects safetensors by suffix.
        args.pretrained_weights = args.pretrained_weights.expanduser().absolute()
        if not args.pretrained_weights.is_file():
            raise CacheBuildError(f"DINO pretrained weights do not exist: {args.pretrained_weights}")
    validation = validate_dataset(args.dataset_root, repo_id, args.cameras)
    selection = _select_cache_interval(validation, args)
    shape = _cache_shape(selection.frame_count, len(validation.camera_keys))
    byte_count = _cache_byte_count(shape)
    _print_plan(validation, selection, shape, byte_count, args)

    available, host_available, cgroup_available = _read_mem_available_bytes()
    min_free = int(args.min_free_gib * (1024**3))
    if available is not None and available < byte_count + min_free:
        limiting_source = "cgroup RAM headroom" if cgroup_available == available else "host MemAvailable"
        raise CacheBuildError(
            f"{limiting_source} is {_format_gib(available)}, but cache {_format_gib(byte_count)} plus "
            f"requested headroom {_format_gib(min_free)} requires {_format_gib(byte_count + min_free)}. "
            "Free RAM, choose a smaller feature-cache design, or explicitly lower --min-free-gib."
        )
    if args.dry_run:
        print("dry-run succeeded; no memfd was allocated and no manifest was written", flush=True)
        return 0

    manifest_path = args.manifest.expanduser().resolve()
    _check_manifest_target(manifest_path, args.replace_stale_manifest)
    fd = _create_memfd(byte_count)
    cache_path = f"/proc/{os.getpid()}/fd/{fd}"
    worker_states: dict[str, Any] = {}
    stop_event = None
    workers: list[Any] = []
    termination_requested = False
    try:
        _write_manifest(
            manifest_path,
            _make_manifest(
                validation=validation,
                args=args,
                shape=shape,
                selection=selection,
                cache_path=cache_path,
                status="building",
                ready=False,
                byte_count=byte_count,
                workers=worker_states,
            ),
        )

        # ``fork`` is deliberate: it shares the anonymous fd without staging
        # 250+ GiB through /dev/shm.  The parent has not imported torch/CUDA.
        import multiprocessing as mp

        context = mp.get_context("fork")
        stop_event = context.Event()
        event_queue = context.Queue()
        worker_count = len(args.devices)
        split = (selection.frame_count + worker_count - 1) // worker_count
        ranges = [
            (
                selection.absolute_start + rank * split,
                min(selection.absolute_start + (rank + 1) * split, selection.absolute_end),
            )
            for rank in range(worker_count)
        ]
        if any(start >= end for start, end in ranges):
            raise CacheBuildError(
                f"cannot split {selection.frame_count} frames across {worker_count} cache worker(s)"
            )

        def request_stop(signum: int, _frame: Any) -> None:
            nonlocal termination_requested
            termination_requested = True
            if stop_event is not None:
                stop_event.set()
            print(f"received signal {signum}; stopping cache server", flush=True)

        previous_handlers = {
            signal.SIGTERM: signal.signal(signal.SIGTERM, request_stop),
            signal.SIGINT: signal.signal(signal.SIGINT, request_stop),
        }
        try:
            for rank, ((index_start, index_end), device_index) in enumerate(zip(ranges, args.devices, strict=True)):
                worker = context.Process(
                    target=_worker_main,
                    args=(
                        rank,
                        device_index,
                        fd,
                        shape,
                        selection.absolute_start,
                        index_start,
                        index_end,
                    ),
                    kwargs={
                        "dataset_root": str(validation.root),
                        "repo_id": validation.repo_id,
                        "camera_keys": validation.camera_keys,
                        "model_name": args.model,
                        "pretrained": args.pretrained,
                        "pretrained_weights": (
                            str(args.pretrained_weights) if args.pretrained_weights is not None else None
                        ),
                        "video_backend": args.video_backend,
                        "batch_size": args.batch_size,
                        "progress_every_batches": args.progress_every_batches,
                        "event_queue": event_queue,
                        "stop_event": stop_event,
                    },
                    name=f"dino-cache-gpu{device_index}",
                )
                worker.start()
                workers.append(worker)

            last_manifest_write = 0.0
            errors: list[dict[str, Any]] = []
            while any(worker.is_alive() for worker in workers):
                errors.extend(_drain_events(event_queue, worker_states))
                now = time.monotonic()
                if now - last_manifest_write >= 5.0:
                    _write_manifest(
                        manifest_path,
                        _make_manifest(
                            validation=validation,
                            args=args,
                            shape=shape,
                            selection=selection,
                            cache_path=cache_path,
                            status="stopping" if termination_requested else "building",
                            ready=False,
                            byte_count=byte_count,
                            workers=worker_states,
                        ),
                    )
                    last_manifest_write = now
                if termination_requested or errors:
                    _terminate_workers(workers, stop_event)
                    break
                for worker in workers:
                    worker.join(timeout=0.2)

            errors.extend(_drain_events(event_queue, worker_states))
            exit_codes = {str(rank): worker.exitcode for rank, worker in enumerate(workers)}
            if termination_requested:
                raise CacheBuildCancelled("cache server stopped before all features were ready")
            if errors or any(code != 0 for code in exit_codes.values()):
                details = errors[-1]["error"] if errors else f"worker exit codes: {exit_codes}"
                raise CacheBuildError(f"DINO cache worker failed: {details}")

            completed = [state for state in worker_states.values() if state.get("type") == "complete"]
            completed_frames = sum(int(state.get("frames_written", 0)) for state in completed)
            if len(completed) != worker_count or completed_frames != selection.frame_count:
                raise CacheBuildError(
                    f"cache completion coverage is invalid: {len(completed)} workers, "
                    f"{completed_frames}/{selection.frame_count} frames"
                )
            _verify_ready_cache(fd, shape)
            _write_manifest(
                manifest_path,
                _make_manifest(
                    validation=validation,
                    args=args,
                    shape=shape,
                    selection=selection,
                    cache_path=cache_path,
                    status="ready",
                    ready=True,
                    byte_count=byte_count,
                    workers=worker_states,
                ),
            )
            print(
                f"cache ready: {cache_path} ({_format_gib(byte_count)}); "
                f"manifest={manifest_path}. Keep this process running while training.",
                flush=True,
            )

            # A ready cache must remain owned by this parent.  ``Event.wait``
            # avoids a busy loop and lets SIGTERM/SIGINT close the fd cleanly.
            while not termination_requested:
                time.sleep(1.0)
        finally:
            for sig, handler in previous_handlers.items():
                signal.signal(sig, handler)
    except BaseException as exc:
        error_text = f"{type(exc).__name__}: {exc}"
        try:
            if stop_event is not None and workers:
                _terminate_workers(workers, stop_event)
            _write_manifest(
                manifest_path,
                _make_manifest(
                    validation=validation,
                    args=args,
                    shape=shape,
                    selection=selection,
                    cache_path=cache_path,
                    status="stopped" if isinstance(exc, CacheBuildCancelled) else "failed",
                    ready=False,
                    byte_count=byte_count,
                    workers=worker_states,
                    error=error_text,
                ),
            )
        except Exception as manifest_exc:
            print(f"also failed to update cache manifest: {manifest_exc}", file=sys.stderr, flush=True)
        raise
    finally:
        # Once this closes, /proc/<parent>/fd/<fd> stops being a valid backing
        # file.  The terminal manifest is deliberately left ready=false so a
        # cache-aware trainer fails safely instead of consuming stale memory.
        os.close(fd)

    # The normal ready path only exits after SIGTERM/SIGINT.  Mark the retained
    # manifest stopped after that final lifecycle transition.
    _write_manifest(
        manifest_path,
        _make_manifest(
            validation=validation,
            args=args,
            shape=shape,
            selection=selection,
            cache_path=cache_path,
            status="stopped",
            ready=False,
            byte_count=byte_count,
            workers=worker_states,
            error="cache server shut down; backing memfd is closed",
        ),
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    try:
        return run(parse_args(argv))
    except CacheBuildError as exc:
        print(f"cache build error: {exc}", file=sys.stderr, flush=True)
        return 2
    except KeyboardInterrupt:
        print("cache build interrupted", file=sys.stderr, flush=True)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
