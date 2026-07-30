#!/usr/bin/env python3
"""Build a durable, disk-backed frozen-DINO cache for ACT training.

``CachedDinoFeatureDataset`` already consumes an ordinary raw cache file via
``numpy.memmap``.  This companion to ``build_dino_memfd_feature_cache.py``
uses that existing contract while avoiding the RAM requirement of a memfd.
It is intended for two-GPU hosts whose cgroup limit is smaller than the full
four-camera DINO cache.  The completed cache and manifest survive the builder
process, so the two independent training jobs can safely share them.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import queue
import shutil
import signal
import sys
import time
from pathlib import Path
from typing import Any

import build_dino_memfd_feature_cache as shared


def _topcam_provenance(dataset_root: Path) -> dict[str, Any] | None:
    """Bind a visual cache to the exact topcam geometry when one is present."""
    path = dataset_root / "meta" / "topcam_rectification.json"
    if not path.is_file():
        return None
    raw = path.read_bytes()
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise shared.CacheBuildError(f"Invalid topcam provenance JSON: {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise shared.CacheBuildError(f"Topcam provenance must be a JSON object: {path}")
    return {
        "path": str(path),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "profile": payload.get("profile"),
        "pipeline": payload.get("pipeline"),
        "head_camera_feature_to_eye": payload.get("head_camera_feature_to_eye"),
        "selected_model_topcam_eye": payload.get("selected_model_topcam_eye"),
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--cache-file",
        type=Path,
        required=True,
        help="New raw fp16 cache file. It is never overwritten by this tool.",
    )
    parser.add_argument(
        "--min-free-disk-gib",
        type=float,
        default=64.0,
        help="Required free filesystem capacity after allocating the cache.",
    )
    parser.add_argument(
        "--flush-every-batches",
        type=int,
        default=1,
        help="Durably flush each worker's mapped cache at this cadence.",
    )
    disk_args, remaining = parser.parse_known_args(argv)
    args = shared.parse_args(remaining)
    if disk_args.min_free_disk_gib < 0:
        parser.error("--min-free-disk-gib must be non-negative")
    if disk_args.flush_every_batches < 1:
        parser.error("--flush-every-batches must be positive")
    args.cache_file = disk_args.cache_file.expanduser().resolve()
    args.min_free_disk_gib = disk_args.min_free_disk_gib
    args.flush_every_batches = disk_args.flush_every_batches
    return args


def _make_manifest(
    *,
    validation: shared.DatasetValidation,
    args: argparse.Namespace,
    shape: tuple[int, int, int, int, int],
    selection: shared.CacheSelection,
    cache_path: str,
    status: str,
    ready: bool,
    byte_count: int,
    workers: dict[str, Any],
    error: str | None = None,
) -> dict[str, Any]:
    payload = shared._make_manifest(
        validation=validation,
        args=args,
        shape=shape,
        selection=selection,
        cache_path=cache_path,
        status=status,
        ready=ready,
        byte_count=byte_count,
        workers=workers,
        error=error,
    )
    # There is no live cache server to retain after a durable file is ready.
    # Keep the manifest explicitly self-describing so consumer launchers do
    # not accidentally require a dead builder PID.
    payload["storage"] = "disk"
    payload["parent_pid"] = None
    payload["pid"] = None
    payload["fd_path"] = cache_path
    topcam = _topcam_provenance(validation.root)
    if topcam is not None:
        payload["topcam_rectification"] = topcam
    return payload


def _create_cache_file(path: Path, size_bytes: int, min_free_bytes: int) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise shared.CacheBuildError(
            f"Refusing to overwrite an existing DINO disk cache: {path}. Choose a new path."
        )
    usage = shutil.disk_usage(path.parent)
    if usage.free < size_bytes + min_free_bytes:
        raise shared.CacheBuildError(
            f"Filesystem free space at {path.parent} is {shared._format_gib(usage.free)}, but cache "
            f"{shared._format_gib(size_bytes)} plus requested headroom "
            f"{shared._format_gib(min_free_bytes)} requires "
            f"{shared._format_gib(size_bytes + min_free_bytes)}"
        )
    fd = os.open(path, os.O_RDWR | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        os.set_inheritable(fd, True)
        os.ftruncate(fd, size_bytes)
        if os.fstat(fd).st_size != size_bytes:
            raise shared.CacheBuildError(
                f"Disk cache truncated to {os.fstat(fd).st_size}, expected {size_bytes}"
            )
        return fd
    except BaseException:
        os.close(fd)
        raise


def run(args: argparse.Namespace) -> int:
    repo_id = args.repo_id or args.dataset_root.expanduser().resolve().name
    if args.pretrained_weights is not None:
        # Keep a Hugging Face snapshot's ``.safetensors`` filename intact.
        # ``Path.resolve()`` follows its symlink into ``blobs/<sha>`` (which
        # has no suffix), making the downstream backbone incorrectly choose
        # torch.load instead of safetensors.load_file.
        args.pretrained_weights = args.pretrained_weights.expanduser().absolute()
        if not args.pretrained_weights.is_file():
            raise shared.CacheBuildError(f"DINO pretrained weights do not exist: {args.pretrained_weights}")

    validation = shared.validate_dataset(args.dataset_root, repo_id, args.cameras)
    selection = shared._select_cache_interval(validation, args)
    shape = shared._cache_shape(selection.frame_count, len(validation.camera_keys))
    byte_count = shared._cache_byte_count(shape)
    shared._print_plan(validation, selection, shape, byte_count, args)
    print(f"  storage: disk ({args.cache_file})", flush=True)
    print(f"  required disk headroom: {args.min_free_disk_gib:.1f} GiB", flush=True)

    if args.dry_run:
        usage = shutil.disk_usage(args.cache_file.parent)
        required = byte_count + int(args.min_free_disk_gib * (1024**3))
        if usage.free < required:
            raise shared.CacheBuildError(
                f"dry-run disk check failed: free={shared._format_gib(usage.free)}, "
                f"required={shared._format_gib(required)}"
            )
        print("dry-run succeeded; no disk cache or manifest was written", flush=True)
        return 0

    manifest_path = args.manifest.expanduser().resolve()
    shared._check_manifest_target(manifest_path, args.replace_stale_manifest)
    fd = _create_cache_file(
        args.cache_file,
        byte_count,
        int(args.min_free_disk_gib * (1024**3)),
    )
    cache_path = str(args.cache_file)
    worker_states: dict[str, Any] = {}
    stop_event: Any = None
    workers: list[Any] = []
    termination_requested = False
    try:
        shared._write_manifest(
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

        # Fork shares the pre-sized file descriptor without staging feature
        # maps through RAM. The parent remains CUDA untouched.
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
            raise shared.CacheBuildError(
                f"cannot split {selection.frame_count} frames across {worker_count} cache worker(s)"
            )

        def request_stop(signum: int, _frame: Any) -> None:
            nonlocal termination_requested
            termination_requested = True
            if stop_event is not None:
                stop_event.set()
            print(f"received signal {signum}; stopping disk-cache build", flush=True)

        previous_handlers = {
            signal.SIGTERM: signal.signal(signal.SIGTERM, request_stop),
            signal.SIGINT: signal.signal(signal.SIGINT, request_stop),
        }
        try:
            for rank, ((index_start, index_end), device_index) in enumerate(
                zip(ranges, args.devices, strict=True)
            ):
                worker = context.Process(
                    target=shared._worker_main,
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
                        "flush_every_batches": args.flush_every_batches,
                        "evict_after_flush": True,
                    },
                    name=f"dino-disk-cache-gpu{device_index}",
                )
                worker.start()
                workers.append(worker)

            last_manifest_write = 0.0
            errors: list[dict[str, Any]] = []
            while any(worker.is_alive() for worker in workers):
                errors.extend(shared._drain_events(event_queue, worker_states))
                now = time.monotonic()
                if now - last_manifest_write >= 5.0:
                    shared._write_manifest(
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
                    shared._terminate_workers(workers, stop_event)
                    break
                for worker in workers:
                    worker.join(timeout=0.2)

            errors.extend(shared._drain_events(event_queue, worker_states))
            exit_codes = {str(rank): worker.exitcode for rank, worker in enumerate(workers)}
            if termination_requested:
                raise shared.CacheBuildCancelled("disk-cache build stopped before all features were ready")
            if errors or any(code != 0 for code in exit_codes.values()):
                details = errors[-1]["error"] if errors else f"worker exit codes: {exit_codes}"
                raise shared.CacheBuildError(f"DINO disk-cache worker failed: {details}")

            completed = [state for state in worker_states.values() if state.get("type") == "complete"]
            completed_frames = sum(int(state.get("frames_written", 0)) for state in completed)
            if len(completed) != worker_count or completed_frames != selection.frame_count:
                raise shared.CacheBuildError(
                    f"disk-cache completion coverage invalid: {len(completed)} workers, "
                    f"{completed_frames}/{selection.frame_count} frames"
                )
            os.fsync(fd)
            shared._verify_ready_cache(fd, shape)
            shared._write_manifest(
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
                f"disk cache ready: {cache_path} ({shared._format_gib(byte_count)}); "
                f"manifest={manifest_path}",
                flush=True,
            )
            return 0
        finally:
            for sig, handler in previous_handlers.items():
                signal.signal(sig, handler)
    except BaseException as exc:
        error_text = f"{type(exc).__name__}: {exc}"
        try:
            if stop_event is not None and workers:
                shared._terminate_workers(workers, stop_event)
            shared._write_manifest(
                manifest_path,
                _make_manifest(
                    validation=validation,
                    args=args,
                    shape=shape,
                    selection=selection,
                    cache_path=cache_path,
                    status="stopped" if isinstance(exc, shared.CacheBuildCancelled) else "failed",
                    ready=False,
                    byte_count=byte_count,
                    workers=worker_states,
                    error=error_text,
                ),
            )
        except Exception as manifest_exc:
            print(f"also failed to update disk-cache manifest: {manifest_exc}", file=sys.stderr, flush=True)
        raise
    finally:
        os.close(fd)


def main(argv: list[str] | None = None) -> int:
    try:
        return run(parse_args(argv))
    except shared.CacheBuildError as exc:
        print(f"disk-cache build error: {exc}", file=sys.stderr, flush=True)
        return 2
    except KeyboardInterrupt:
        print("disk-cache build interrupted", file=sys.stderr, flush=True)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
