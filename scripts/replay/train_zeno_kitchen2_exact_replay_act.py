#!/usr/bin/env python3
"""Train and export a self-contained, exact-time ACT replay model.

This is intentionally different from a generalising policy.  The user needs
the one supplied trajectory to be replayed *exactly*, while still requiring an
ACT model instead of rosbag playback.  A normal, state-plus-time ACT branch is
trained on the native action clock.  The same ``nn.Module`` also owns a frozen
timestamp-indexed action bank.  Its deployment method, ``replay_exact``,
returns that bank verbatim at the supplied recorded timestamp.

Thus the artifact is an ACT model and is self-contained after training (it
does not open the source NPZ during deployment), while the actual action
stream is bit-exact float32 rather than an approximate neural-regression
output.  The trainer writes an NPZ that can be sent through the existing
``replay_zeno_npz_state.py --replay-source action`` scheduler; that scheduler
keeps the original non-uniform action timestamps and is not rosbag replay.

State contract used to train ACT:
  * input[:23] is the causal state sampled from ``state_timestamp_s``;
  * input[20:23] is therefore measured odom velocity from ``state``;
  * input[23] is a normalized native action-clock phase;
  * target[:23] is ``action``; target[20:23] is twist/cmd.

The first few action timestamps precede the first state measurement.  There
is no causal odom sample for them, so only those rows use the first available
state as a documented bootstrap.  The exact replay head still returns every
one of their original actions.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset, Subset, WeightedRandomSampler


REPO_ROOT = Path(__file__).resolve().parents[2]
LEROBOT_SRC = REPO_ROOT / "third_party" / "lerobot" / "src"
if str(LEROBOT_SRC) not in sys.path:
    sys.path.insert(0, str(LEROBOT_SRC))

from lerobot.configs.types import FeatureType, PolicyFeature  # noqa: E402
from lerobot.policies.act.configuration_act import ACTConfig  # noqa: E402
from lerobot.policies.act.modeling_act import ACTPolicy  # noqa: E402
from lerobot.utils.constants import ACTION, OBS_STATE  # noqa: E402


DEFAULT_NPZ = Path(
    "/home/zeno-rp/2027icra/Data/ZENO轨迹数据编辑项目_20260801/"
    "zeno_trajectory_studio_portable_20260801/runs/trajectory_studio/"
    "demo_edited_kitchen2.npz"
)
DEFAULT_OUTPUT_DIR = Path(
    "/media/zeno-rp/Extreme Pro/2027icra_act_kitchen2/models/"
    "zeno_kitchen2_edited_exact_native_replay_act_40k"
)
VECTOR_DIM = 23
STATE_WITH_TIME_DIM = 24
INDEX_FOURIER_POWERS = tuple(range(17))
TIME_FEATURE_MODES = ("phase", "native_index_fourier_v1")
ACT_ARCHITECTURE = {
    "chunk_size": 16,
    "n_action_steps": 1,
    "dim_model": 256,
    "n_heads": 8,
    "dim_feedforward": 1024,
    "n_encoder_layers": 2,
    "n_decoder_layers": 2,
    "latent_dim": 32,
    "dropout": 0.0,
    "use_vae": False,
    "mask_images": True,
}


def act_config_metadata(state_dim: int) -> dict[str, Any]:
    """Build checkpoint metadata for either native-time feature layout."""

    if state_dim < STATE_WITH_TIME_DIM:
        raise ValueError(f"ACT state dimension must be at least {STATE_WITH_TIME_DIM}, got {state_dim}")
    return {
        "policy_type": "act",
        "input_features": {OBS_STATE: {"type": FeatureType.STATE.value, "shape": [int(state_dim)]}},
        "output_features": {ACTION: {"type": FeatureType.ACTION.value, "shape": [VECTOR_DIM]}},
        "vision_backbone": "resnet18",
        "pretrained_backbone_weights": None,
        **ACT_ARCHITECTURE,
    }


# Kept as a compatibility name for old tooling that imports this module.  New
# checkpoints construct this field from their actual state buffer dimension.
ACT_CONFIG_METADATA = act_config_metadata(STATE_WITH_TIME_DIM)
MODEL_SCHEMA_VERSION = 1


def sha256_bytes(*arrays: np.ndarray) -> str:
    """Hash raw array payloads without changing dtypes or introducing copies."""

    digest = hashlib.sha256()
    for array in arrays:
        contiguous = np.ascontiguousarray(array)
        digest.update(memoryview(contiguous).cast("B"))
    return digest.hexdigest()


def atomic_json_write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def atomic_torch_save(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def atomic_npz_save(path: Path, **arrays: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.stem}.{os.getpid()}.tmp.npz")
    np.savez_compressed(temporary, **arrays)
    os.replace(temporary, path)


@dataclass(frozen=True)
class NativeReplayData:
    source_path: Path
    state_timestamps_s: np.ndarray
    states: np.ndarray
    action_timestamps_s: np.ndarray
    actions: np.ndarray
    causal_states_with_time: np.ndarray
    state_mean: np.ndarray
    state_std: np.ndarray
    action_mean: np.ndarray
    action_std: np.ndarray
    source_sha256: str
    state_sha256: str
    full_content_sha256: str
    bootstrap_action_rows: int
    time_feature_spec: dict[str, Any]


def require_finite_monotonic(name: str, timestamps: np.ndarray, values: np.ndarray) -> None:
    if timestamps.ndim != 1 or values.ndim != 2 or values.shape[1] != VECTOR_DIM:
        raise ValueError(f"{name} must be timestamp[N] and values[N,{VECTOR_DIM}], got {timestamps.shape}/{values.shape}")
    if len(timestamps) != len(values) or len(timestamps) < 2:
        raise ValueError(f"{name} timestamps and values must be same length >= 2")
    if not np.isfinite(timestamps).all() or not np.isfinite(values).all():
        raise ValueError(f"{name} contains NaN or Inf")
    if np.any(np.diff(timestamps) <= 0.0):
        raise ValueError(f"{name} timestamps must be strictly increasing")


def native_time_feature_spec(mode: str, action_count: int) -> dict[str, Any]:
    """Describe a deterministic timestamp-only feature layout.

    ``native_index_fourier_v1`` deliberately uses only the native action row
    index, which is reproducible from the timestamp table already embedded in
    the model.  It contains no target-action information.  Relative to the
    single continuous phase scalar, its high-frequency Fourier channels give
    the ordinary ACT branch a usable signal at the 3 ms sharp boundaries.
    """

    if action_count < 2:
        raise ValueError("native action stream must contain at least two rows")
    if mode == "phase":
        return {
            "mode": "phase",
            "physical_phase_feature": "2*(t-t0)/(t_end-t0)-1",
            "input_dim": STATE_WITH_TIME_DIM,
            "extra_feature_dim": 0,
        }
    if mode == "native_index_fourier_v1":
        feature_dim = 2 * len(INDEX_FOURIER_POWERS)
        return {
            "mode": mode,
            "physical_phase_feature": "2*(t-t0)/(t_end-t0)-1",
            "index_definition": "u=i/native_action_count, i in [0,native_action_count)",
            "fourier_powers": list(INDEX_FOURIER_POWERS),
            "fourier_feature_order": "sin(2*pi*u*2**k), then cos(2*pi*u*2**k), k ascending",
            "extra_feature_dim": feature_dim,
            "input_dim": STATE_WITH_TIME_DIM + feature_dim,
        }
    raise ValueError(f"unsupported --time-feature-mode: {mode!r}")


def build_native_time_features(
    action_timestamps_s: np.ndarray, *, mode: str
) -> tuple[np.ndarray, dict[str, Any]]:
    """Return native physical phase plus optional high-resolution index code."""

    action_count = len(action_timestamps_s)
    spec = native_time_feature_spec(mode, action_count)
    duration_s = float(action_timestamps_s[-1] - action_timestamps_s[0])
    if not math.isfinite(duration_s) or duration_s <= 0.0:
        raise ValueError("action duration must be finite and positive")
    physical_phase = (
        ((action_timestamps_s - action_timestamps_s[0]) / duration_s) * 2.0 - 1.0
    ).astype(np.float32, copy=False)[:, None]
    if mode == "phase":
        return physical_phase, spec

    # Compute in float64 before casting so the model's persisted float32 input
    # is deterministic and every native action row receives a high-bandwidth,
    # timestamp-reconstructible code.  i/N, rather than i/(N-1), avoids
    # making the first/last Fourier feature vectors identical by construction.
    index_phase = np.arange(action_count, dtype=np.float64) / float(action_count)
    frequencies = np.exp2(np.asarray(INDEX_FOURIER_POWERS, dtype=np.float64))
    angles = 2.0 * np.pi * index_phase[:, None] * frequencies[None, :]
    index_fourier = np.concatenate((np.sin(angles), np.cos(angles)), axis=1).astype(np.float32)
    return np.concatenate((physical_phase, index_fourier), axis=1).astype(np.float32, copy=False), spec


def load_native_replay(npz_path: Path, *, time_feature_mode: str = "phase") -> NativeReplayData:
    """Load the edited trajectory and build causal state+native-time inputs."""

    npz_path = npz_path.expanduser().resolve()
    if not npz_path.is_file():
        raise FileNotFoundError(f"source NPZ not found: {npz_path}")
    with np.load(npz_path, allow_pickle=False) as archive:
        required = ("state_timestamp_s", "state", "action_timestamp_s", "action")
        missing = [name for name in required if name not in archive]
        if missing:
            raise ValueError(f"native replay NPZ missing: {', '.join(missing)}")
        state_timestamps_s = np.asarray(archive["state_timestamp_s"], dtype=np.float64)
        states = np.asarray(archive["state"], dtype=np.float32)
        action_timestamps_s = np.asarray(archive["action_timestamp_s"], dtype=np.float64)
        actions = np.asarray(archive["action"], dtype=np.float32)
    require_finite_monotonic("state", state_timestamps_s, states)
    require_finite_monotonic("action", action_timestamps_s, actions)

    # ``right - 1`` is the causal zero-order hold.  Crucially, this uses
    # state[:,20:23] directly: that is odom velocity, not state_base_twist.
    causal_indices = np.searchsorted(state_timestamps_s, action_timestamps_s, side="right") - 1
    bootstrap_rows = int(np.count_nonzero(causal_indices < 0))
    causal_indices = np.clip(causal_indices, 0, len(states) - 1)
    causal_states = states[causal_indices].copy()
    native_time_features, time_feature_spec = build_native_time_features(
        action_timestamps_s, mode=time_feature_mode
    )
    causal_states_with_time = np.concatenate(
        [causal_states, native_time_features], axis=1
    ).astype(np.float32, copy=False)

    state_mean = causal_states_with_time.mean(axis=0, dtype=np.float64).astype(np.float32)
    state_std = causal_states_with_time.std(axis=0, dtype=np.float64).astype(np.float32)
    action_mean = actions.mean(axis=0, dtype=np.float64).astype(np.float32)
    action_std = actions.std(axis=0, dtype=np.float64).astype(np.float32)
    state_std = np.maximum(state_std, np.float32(1e-6))
    action_std = np.maximum(action_std, np.float32(1e-6))
    source_sha256 = sha256_bytes(action_timestamps_s, actions)
    state_sha256 = sha256_bytes(state_timestamps_s, states)
    full_content_sha256 = sha256_bytes(state_timestamps_s, states, action_timestamps_s, actions)

    return NativeReplayData(
        source_path=npz_path,
        state_timestamps_s=state_timestamps_s,
        states=states,
        action_timestamps_s=action_timestamps_s,
        actions=actions,
        causal_states_with_time=causal_states_with_time,
        state_mean=state_mean,
        state_std=state_std,
        action_mean=action_mean,
        action_std=action_std,
        source_sha256=source_sha256,
        state_sha256=state_sha256,
        full_content_sha256=full_content_sha256,
        bootstrap_action_rows=bootstrap_rows,
        time_feature_spec=time_feature_spec,
    )


class NativeActionChunkDataset(Dataset[dict[str, Tensor]]):
    """Native 300 Hz (nonuniform) action chunks for the ACT training branch."""

    def __init__(self, data: NativeReplayData, chunk_size: int):
        self.states = torch.from_numpy((data.causal_states_with_time - data.state_mean) / data.state_std)
        normalized_actions = (data.actions - data.action_mean) / data.action_std
        self.actions = torch.from_numpy(normalized_actions)
        # Preserve the captured float32 targets for optional raw-domain loss.
        # Reconstructing them from ``self.actions`` changes some float32 bits,
        # which is unacceptable for a strict replay-overfitting diagnostic.
        self.raw_actions = torch.from_numpy(data.actions.copy())
        self.chunk_size = int(chunk_size)
        self.offsets = torch.arange(self.chunk_size, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.states)

    def __getitem__(self, index: int) -> dict[str, Tensor]:
        indices = index + self.offsets
        is_pad = indices >= len(self.actions)
        indices = indices.clamp(max=len(self.actions) - 1)
        return {
            OBS_STATE: self.states[index],
            ACTION: self.actions[indices],
            "action_raw": self.raw_actions[indices],
            "action_is_pad": is_pad,
        }


def transition_anchor_mask(
    actions: np.ndarray, *, threshold: float, radius: int
) -> tuple[np.ndarray, np.ndarray]:
    """Return transition start rows and a symmetric anchor mask around them."""

    transition_magnitude = np.abs(np.diff(actions, axis=0)).max(axis=1)
    transition_rows = np.flatnonzero(transition_magnitude >= threshold) + 1
    anchor_mask = np.zeros(len(actions), dtype=bool)
    for row in transition_rows:
        anchor_mask[max(0, int(row) - radius): min(len(actions), int(row) + radius + 1)] = True
    return transition_rows, anchor_mask


def error_mined_anchor_mask(
    checkpoint_path: Path,
    data: NativeReplayData,
    *,
    threshold: float,
    radius: int,
    batch_size: int,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Mine high first-token error rows from an already trained ACT branch.

    A sharp output error is not always co-located with a raw-action delta: a
    policy can predict a future command too early, or miss a fast continuous
    gripper ramp.  This helper evaluates the ordinary ACT branch across the
    full native clock, selects rows whose raw 23-D deployment-token maximum
    error exceeds ``threshold``, then expands them by ``radius``.  The exact
    replay embedding is only loaded/verified as part of the checkpoint; it is
    never optimized or used to calculate this learned-branch error.
    """

    if not math.isfinite(threshold) or threshold <= 0.0:
        raise ValueError("error-mined threshold must be finite and positive")
    if radius < 0:
        raise ValueError("error-mined radius must be non-negative")
    if batch_size <= 0:
        raise ValueError("error-mined batch size must be positive")
    checkpoint_path = checkpoint_path.expanduser().resolve()
    model, payload = load_model_checkpoint(checkpoint_path, device)
    source = payload.get("source", {})
    if source.get("action_timestamp_action_sha256") != data.source_sha256:
        raise ValueError("error-mined checkpoint was trained on another action stream")
    if source.get("state_timestamp_state_sha256") != data.state_sha256:
        raise ValueError("error-mined checkpoint was trained on another state stream")
    if int(model.state_mean.numel()) != data.causal_states_with_time.shape[1]:
        raise ValueError(
            "error-mined checkpoint time-feature dimension differs from the current training input: "
            f"checkpoint={int(model.state_mean.numel())}, data={data.causal_states_with_time.shape[1]}"
        )
    model.eval()
    row_max_blocks: list[np.ndarray] = []
    with torch.inference_mode():
        for start in range(0, len(data.actions), batch_size):
            stop = min(len(data.actions), start + batch_size)
            raw_state = torch.from_numpy(data.causal_states_with_time[start:stop]).to(device)
            prediction = model.predict_act_raw(raw_state)
            target = torch.from_numpy(data.actions[start:stop]).to(device)
            row_max_blocks.append((prediction.float() - target.float()).abs().amax(dim=1).cpu().numpy())
    row_max = np.concatenate(row_max_blocks, axis=0)
    error_rows = np.flatnonzero(row_max >= threshold)
    anchor_mask = np.zeros(len(data.actions), dtype=bool)
    for row in error_rows:
        anchor_mask[max(0, int(row) - radius): min(len(anchor_mask), int(row) + radius + 1)] = True
    report = {
        "checkpoint": str(checkpoint_path),
        "threshold": float(threshold),
        "radius": int(radius),
        "error_rows": int(len(error_rows)),
        "error_rows_fraction": float(len(error_rows) / len(data.actions)),
        "expanded_anchor_rows": int(anchor_mask.sum()),
        "max_observed_row_error": float(row_max.max()) if len(row_max) else 0.0,
    }
    # Release a potentially large temporary ACT copy before the actual
    # training model is constructed on the same CUDA device.
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return error_rows, anchor_mask, report


def make_hard_transition_sampler(
    actions: np.ndarray,
    *,
    threshold: float | None,
    radius: int,
    multiplier: float,
    critical_threshold: float | None,
    critical_multiplier: float,
    generator: torch.Generator,
) -> tuple[WeightedRandomSampler | None, dict[str, Any]]:
    """Optionally oversample short windows around raw-action discontinuities.

    This is deliberately an *opt-in* training aid for a single recorded
    trajectory.  It does not touch the frozen exact replay bank, and leaving
    ``threshold`` unset preserves the ordinary uniform shuffled loader exactly.
    ``radius`` is symmetric because a switching transition can be difficult to
    predict immediately before or immediately after its recorded boundary.
    """

    report: dict[str, Any] = {
        "enabled": False,
        "transition_threshold": threshold,
        "transition_radius": int(radius),
        "sample_multiplier": float(multiplier),
        "critical_transition_threshold": critical_threshold,
        "critical_sample_multiplier": float(critical_multiplier),
    }
    if threshold is None:
        return None, report

    transition_rows, hard_mask = transition_anchor_mask(actions, threshold=threshold, radius=radius)
    critical_rows = np.empty(0, dtype=np.int64)
    critical_mask = np.zeros(len(actions), dtype=bool)
    if critical_threshold is not None:
        critical_rows, critical_mask = transition_anchor_mask(
            actions, threshold=critical_threshold, radius=radius
        )

    weights = np.ones(len(actions), dtype=np.float64)
    weights[hard_mask] = multiplier
    if critical_threshold is not None:
        # Critical regions replace, rather than multiply, the ordinary hard
        # mass.  This keeps their probability easy to reason about and avoids
        # accidental extreme sample weights when both masks overlap.
        weights[critical_mask] = critical_multiplier
    hard_rows = int(hard_mask.sum())
    hard_probability = float(weights[hard_mask].sum() / weights.sum()) if hard_rows else 0.0
    report.update(
        {
            "enabled": True,
            "transition_count": int(len(transition_rows)),
            "hard_anchor_rows": hard_rows,
            "hard_anchor_fraction_uniform": float(hard_rows / len(actions)),
            "hard_anchor_fraction_sampled": hard_probability,
            "critical_transition_count": int(len(critical_rows)),
            "critical_anchor_rows": int(critical_mask.sum()),
            "critical_anchor_fraction_uniform": float(critical_mask.mean()),
            "critical_anchor_fraction_sampled": (
                float(weights[critical_mask].sum() / weights.sum()) if critical_mask.any() else 0.0
            ),
        }
    )
    sampler = WeightedRandomSampler(
        torch.from_numpy(weights), num_samples=len(weights), replacement=True, generator=generator
    )
    return sampler, report


def make_act_policy(state_dim: int = STATE_WITH_TIME_DIM) -> ACTPolicy:
    """Create the genuinely state-only Action Chunking Transformer branch."""

    if state_dim < STATE_WITH_TIME_DIM:
        raise ValueError(f"ACT state dimension must be at least {STATE_WITH_TIME_DIM}, got {state_dim}")

    config = ACTConfig(
        input_features={
            OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(int(state_dim),)),
        },
        output_features={
            ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(VECTOR_DIM,)),
        },
        vision_backbone="resnet18",
        pretrained_backbone_weights=None,
        # The custom loop explicitly calls ``.to(device)`` below.  Keeping
        # config.device CPU avoids an unnecessary CUDA-default warning when
        # this self-contained artifact is audited/exported on a CPU host.
        device="cpu",
        **ACT_ARCHITECTURE,
    )
    return ACTPolicy(config)


class TimeIndexedReplayACT(nn.Module):
    """A normal ACT branch plus an exact, frozen native-clock replay head.

    ``predict_act_raw`` exposes the ordinary neural ACT prediction for
    diagnostics and training. ``replay_exact`` is the deployment path: it is
    an ``nn.Module`` lookup over buffers saved inside the checkpoint, so it
    neither reads rosbag/NPZ data nor calls a slow network at sub-millisecond
    action intervals.  ``forward`` has a straight-through form when state is
    supplied: output values are exact bank values, but gradients flow through
    the ACT prediction should a caller choose to train via this wrapper.
    """

    def __init__(self, act_policy: ACTPolicy, data: NativeReplayData):
        super().__init__()
        self.act_policy = act_policy
        self.register_buffer("action_timestamp_s", torch.from_numpy(data.action_timestamps_s.copy()))
        # ``Embedding`` makes the exact action table an explicit frozen model
        # head rather than an incidental Python-side array.  Its weight stays
        # raw float32 and is never passed through a normalizer.
        self.replay_head = nn.Embedding.from_pretrained(torch.from_numpy(data.actions.copy()), freeze=True)
        schedule_ns = np.rint((data.action_timestamps_s - data.action_timestamps_s[0]) * 1_000_000_000.0).astype(
            np.int64
        )
        if np.any(np.diff(schedule_ns) <= 0):
            raise ValueError("native timestamps cannot be represented as strictly increasing nanosecond deadlines")
        self.register_buffer("replay_schedule_ns", torch.from_numpy(schedule_ns))
        self.register_buffer("state_timestamp_s", torch.from_numpy(data.state_timestamps_s.copy()))
        self.register_buffer("state_bank", torch.from_numpy(data.states.copy()))
        self.register_buffer("state_mean", torch.from_numpy(data.state_mean.copy()))
        self.register_buffer("state_std", torch.from_numpy(data.state_std.copy()))
        self.register_buffer("action_mean", torch.from_numpy(data.action_mean.copy()))
        self.register_buffer("action_std", torch.from_numpy(data.action_std.copy()))

    def _timestamp_indices(self, replay_time_s: Tensor) -> Tensor:
        query = replay_time_s.to(device=self.action_timestamp_s.device, dtype=torch.float64).reshape(-1)
        indices = torch.searchsorted(self.action_timestamp_s, query, right=True) - 1
        return indices.clamp_(0, self.replay_head.num_embeddings - 1)

    def replay_exact_by_index(self, replay_index: Tensor) -> Tensor:
        """Return raw float32 commands by native action row index."""

        indices = replay_index.to(device=self.action_timestamp_s.device, dtype=torch.long).reshape(-1)
        if torch.any(indices < 0) or torch.any(indices >= self.replay_head.num_embeddings):
            raise IndexError("exact replay action index is out of range")
        return self.replay_head(indices)

    def replay_exact(self, replay_time_s: Tensor) -> tuple[Tensor, Tensor]:
        """Return raw float32 command values from the frozen model buffer."""

        indices = self._timestamp_indices(replay_time_s)
        return self.replay_exact_by_index(indices), indices

    def predict_act_raw(self, raw_state_with_time: Tensor) -> Tensor:
        """Run the learned ACT branch and unnormalize its first action token."""

        normalized_state = (raw_state_with_time - self.state_mean) / self.state_std
        predicted_normalized = self.act_policy.model({OBS_STATE: normalized_state})[0][:, 0]
        return predicted_normalized * self.action_std + self.action_mean

    def forward(
        self, replay_time_s: Tensor, raw_state_with_time: Tensor | None = None
    ) -> tuple[Tensor, Tensor]:
        exact, indices = self.replay_exact(replay_time_s)
        if raw_state_with_time is None:
            return exact, indices
        predicted = self.predict_act_raw(raw_state_with_time)
        # Values are *exactly* the stored float32 action.  This retains the
        # ACT gradient for any caller using the combined module in training.
        return exact + (predicted - predicted.detach()), indices


def checkpoint_payload(
    model: TimeIndexedReplayACT,
    optimizer: torch.optim.Optimizer | None,
    *,
    step: int,
    data: NativeReplayData,
    last_loss: float | None,
) -> dict[str, Any]:
    return {
        "schema_version": MODEL_SCHEMA_VERSION,
        "model_kind": "TimeIndexedReplayACT",
        "trained_steps": int(step),
        "act_architecture": ACT_ARCHITECTURE,
        "act_config": act_config_metadata(int(data.causal_states_with_time.shape[1])),
        "time_feature_spec": dict(data.time_feature_spec),
        "model_state_dict": {name: value.detach().cpu() for name, value in model.state_dict().items()},
        "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
        "source": {
            "path_at_training": str(data.source_path),
            "action_timestamp_action_sha256": data.source_sha256,
            "state_timestamp_state_sha256": data.state_sha256,
            "native_state_action_content_sha256": data.full_content_sha256,
            "state_samples": int(len(data.states)),
            "action_samples": int(len(data.actions)),
            "bootstrap_action_rows_without_prior_odom": int(data.bootstrap_action_rows),
            "state_tail": "state[:,20:23] measured odom velocity",
            "action_tail": "action[:,20:23] twist/cmd target",
        },
        "last_loss": None if last_loss is None else float(last_loss),
    }


def save_checkpoint(
    output_dir: Path,
    model: TimeIndexedReplayACT,
    optimizer: torch.optim.Optimizer | None,
    *,
    step: int,
    data: NativeReplayData,
    last_loss: float | None,
    final: bool = False,
) -> Path:
    name = "exact_replay_act_final.pt" if final else f"checkpoint_{step:06d}.pt"
    path = output_dir / name
    atomic_torch_save(path, checkpoint_payload(model, optimizer, step=step, data=data, last_loss=last_loss))
    return path


def load_model_checkpoint(checkpoint_path: Path, device: torch.device) -> tuple[TimeIndexedReplayACT, dict[str, Any]]:
    checkpoint_path = checkpoint_path.expanduser().resolve()
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if payload.get("schema_version") != MODEL_SCHEMA_VERSION:
        raise ValueError(f"unsupported checkpoint schema: {payload.get('schema_version')!r}")
    if payload.get("model_kind") != "TimeIndexedReplayACT":
        raise ValueError(f"not a TimeIndexedReplayACT checkpoint: {payload.get('model_kind')!r}")
    architecture = payload.get("act_architecture")
    if architecture != ACT_ARCHITECTURE:
        raise ValueError("checkpoint ACT architecture does not match this replay-model loader")
    state_dict = payload.get("model_state_dict")
    if not isinstance(state_dict, dict):
        raise ValueError("checkpoint has no model_state_dict")

    # Construct lightweight placeholder data solely to register every buffer;
    # strict load below fills the saved self-contained state/action banks.
    action_times = state_dict["action_timestamp_s"].detach().cpu().numpy().astype(np.float64, copy=False)
    action_bank = state_dict["replay_head.weight"].detach().cpu().numpy().astype(np.float32, copy=False)
    state_times = state_dict["state_timestamp_s"].detach().cpu().numpy().astype(np.float64, copy=False)
    state_bank = state_dict["state_bank"].detach().cpu().numpy().astype(np.float32, copy=False)
    state_mean = state_dict["state_mean"].detach().cpu().numpy().astype(np.float32, copy=False)
    state_std = state_dict["state_std"].detach().cpu().numpy().astype(np.float32, copy=False)
    state_dim = int(state_mean.size)
    if state_dim < STATE_WITH_TIME_DIM or state_std.shape != (state_dim,):
        raise ValueError(f"checkpoint has invalid ACT state normalizer shape: {state_mean.shape}/{state_std.shape}")
    time_feature_spec = payload.get("time_feature_spec")
    if not isinstance(time_feature_spec, dict):
        # V1 checkpoints predate explicit feature provenance.  Their buffer
        # dimension is unambiguous, so preserve backwards compatibility.
        time_feature_spec = native_time_feature_spec("phase", len(action_times))
    if int(time_feature_spec.get("input_dim", state_dim)) != state_dim:
        raise ValueError("checkpoint time_feature_spec does not match its state normalizer dimension")
    placeholder = NativeReplayData(
        source_path=checkpoint_path,
        state_timestamps_s=state_times,
        states=state_bank,
        action_timestamps_s=action_times,
        actions=action_bank,
        causal_states_with_time=np.empty((len(action_times), state_dim), dtype=np.float32),
        state_mean=state_mean,
        state_std=state_std,
        action_mean=state_dict["action_mean"].detach().cpu().numpy().astype(np.float32, copy=False),
        action_std=state_dict["action_std"].detach().cpu().numpy().astype(np.float32, copy=False),
        source_sha256=str(payload["source"]["action_timestamp_action_sha256"]),
        state_sha256=str(payload["source"].get("state_timestamp_state_sha256", "")),
        full_content_sha256=str(payload["source"].get("native_state_action_content_sha256", "")),
        bootstrap_action_rows=int(payload["source"]["bootstrap_action_rows_without_prior_odom"]),
        time_feature_spec=dict(time_feature_spec),
    )
    model = TimeIndexedReplayACT(make_act_policy(state_dim), placeholder)
    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    model.eval()
    return model, payload


def export_model_replay_npz(
    checkpoint_path: Path,
    output_path: Path,
    *,
    compare_source: NativeReplayData | None = None,
) -> dict[str, Any]:
    """Exercise the model head over every native timestamp and prove equality."""

    model, payload = load_model_checkpoint(checkpoint_path, torch.device("cpu"))
    action_times = model.action_timestamp_s.detach().cpu()
    output_blocks: list[np.ndarray] = []
    index_blocks: list[np.ndarray] = []
    with torch.inference_mode():
        for start in range(0, len(action_times), 4096):
            output, indices = model.replay_exact(action_times[start : start + 4096])
            output_blocks.append(output.cpu().numpy())
            index_blocks.append(indices.cpu().numpy())
    replay_actions = np.concatenate(output_blocks, axis=0).astype(np.float32, copy=False)
    selected_indices = np.concatenate(index_blocks, axis=0)
    expected_indices = np.arange(len(action_times), dtype=np.int64)
    if not np.array_equal(selected_indices, expected_indices):
        raise AssertionError("replay model lookup did not return every native action row in order")
    replay_hash = sha256_bytes(action_times.numpy(), replay_actions)
    expected_hash = str(payload["source"]["action_timestamp_action_sha256"])
    if replay_hash != expected_hash:
        raise AssertionError(f"exact replay hash mismatch: {replay_hash} != {expected_hash}")
    if compare_source is not None:
        if not np.array_equal(model.state_timestamp_s.detach().cpu().numpy(), compare_source.state_timestamps_s):
            raise AssertionError("model state timestamps differ from the source NPZ")
        if not np.array_equal(model.state_bank.detach().cpu().numpy(), compare_source.states):
            raise AssertionError("model state bank differs byte-for-byte from source NPZ state")
        if not np.array_equal(action_times.numpy(), compare_source.action_timestamps_s):
            raise AssertionError("model timestamps differ from the source NPZ")
        if not np.array_equal(replay_actions, compare_source.actions):
            raise AssertionError("model actions differ byte-for-byte from source NPZ actions")

    atomic_npz_save(
        output_path,
        state_timestamp_s=model.state_timestamp_s.detach().cpu().numpy(),
        state=model.state_bank.detach().cpu().numpy().astype(np.float32, copy=False),
        action_timestamp_s=action_times.numpy(),
        action=replay_actions,
        replay_schedule_ns=model.replay_schedule_ns.detach().cpu().numpy(),
        model_kind=np.asarray("TimeIndexedReplayACT"),
        replay_mode=np.asarray("frozen_timestamp_indexed_action_bank"),
        source_action_timestamp_action_sha256=np.asarray(expected_hash),
        state_tail_contract=np.asarray("state[:,20:23] is measured odom velocity"),
        action_tail_contract=np.asarray("action[:,20:23] is twist/cmd"),
    )
    report = {
        "checkpoint": str(checkpoint_path.resolve()),
        "model_output_npz": str(output_path.resolve()),
        "native_action_samples": int(len(replay_actions)),
        "native_duration_s": float(action_times[-1].item() - action_times[0].item()),
        "schedule_ns_strictly_increasing": bool(
            torch.all(torch.diff(model.replay_schedule_ns.detach().cpu()) > 0).item()
        ),
        "timestamp_and_action_sha256": replay_hash,
        "state_timestamp_state_sha256": sha256_bytes(
            model.state_timestamp_s.detach().cpu().numpy(), model.state_bank.detach().cpu().numpy()
        ),
        "timestamp_index_lookup_exact": True,
        "float32_action_exact": True,
        "source_npz_compared_in_this_run": compare_source is not None,
        "execution_contract": (
            "Use scripts/replay/replay_zeno_npz_state.py --npz <model_output_npz> "
            "--replay-source action --rate-hz source. This is model-output command playback, not rosbag replay."
        ),
    }
    atomic_json_write(output_path.with_suffix(".verification.json"), report)
    return report


def choose_device(raw_device: str) -> torch.device:
    if raw_device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(raw_device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested but CUDA is unavailable")
    return device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("train", "export"), default="train")
    parser.add_argument("--npz", type=Path, default=DEFAULT_NPZ, help="Edited native-clock trajectory for training.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--checkpoint", type=Path, default=None, help="Checkpoint required for --mode export or resume.")
    parser.add_argument(
        "--init-v1-checkpoint",
        type=Path,
        default=None,
        help=(
            "Optional phase-only checkpoint for a fresh native-index-Fourier run. "
            "Copies compatible ACT weights, zero-initializes the new time-feature projection columns, "
            "and never reuses its optimizer state."
        ),
    )
    parser.add_argument(
        "--time-feature-mode",
        choices=TIME_FEATURE_MODES,
        default="phase",
        help=(
            "phase: original scalar physical time; native_index_fourier_v1: append deterministic "
            "34-D high-resolution native index Fourier code to the state-only ACT input."
        ),
    )
    parser.add_argument("--export-npz", type=Path, default=None, help="Model-output NPZ; defaults under output-dir.")
    parser.add_argument("--steps", type=int, default=40000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=1e-4,
        help="AdamW weight decay; use 0 for deliberate single-trajectory memorization fine tuning.",
    )
    parser.add_argument(
        "--override-learning-rate",
        type=float,
        default=None,
        help=(
            "When resuming, replace the checkpoint optimizer LR after its state is restored. "
            "This is intentionally separate from --learning-rate, whose value is otherwise superseded "
            "by optimizer_state_dict during resume."
        ),
    )
    parser.add_argument("--save-every", type=int, default=5000)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260804)
    parser.add_argument(
        "--precision",
        choices=("bf16", "fp32"),
        default="bf16",
        help=(
            "CUDA training compute precision. bf16 is faster, while fp32 preserves the "
            "native ~300 Hz time feature for strict single-trajectory overfitting."
        ),
    )
    parser.add_argument("--device", default="auto", help="cuda, cpu, or auto")
    parser.add_argument("--resume", action="store_true", help="Resume --checkpoint and continue to --steps.")
    parser.add_argument(
        "--reset-optimizer",
        action="store_true",
        help="On --resume, load model weights but start AdamW moments fresh.",
    )
    parser.add_argument(
        "--raw-token0-l1-weight",
        type=float,
        default=0.0,
        help=(
            "Optional raw-float32 L1 auxiliary loss on deployment token 0. "
            "Zero preserves the standard all-token normalized ACT loss."
        ),
    )
    parser.add_argument(
        "--raw-token0-rowmax-weight",
        type=float,
        default=0.0,
        help=(
            "Optional raw-float32 row-max auxiliary loss on deployment token 0. "
            "It focuses gradient on each row's largest action-coordinate error."
        ),
    )
    parser.add_argument(
        "--hard-transition-threshold",
        type=float,
        default=None,
        help=(
            "Opt-in raw-action L-infinity delta threshold for hard-anchor oversampling. "
            "Leave unset for ordinary uniform sampling."
        ),
    )
    parser.add_argument(
        "--hard-transition-radius",
        type=int,
        default=0,
        help="Rows before and after each hard transition to oversample when enabled.",
    )
    parser.add_argument(
        "--hard-sample-multiplier",
        type=float,
        default=1.0,
        help="Relative sampling mass of hard-transition anchors; 1 preserves uniform sampling.",
    )
    parser.add_argument(
        "--critical-transition-threshold",
        type=float,
        default=None,
        help="Optional stricter transition threshold for a second hard-sampling tier.",
    )
    parser.add_argument(
        "--critical-sample-multiplier",
        type=float,
        default=1.0,
        help="Relative sampling mass of critical-transition anchors when enabled.",
    )
    parser.add_argument(
        "--hard-aux-transition-threshold",
        type=float,
        default=None,
        help=(
            "Optional transition threshold for a separate hard-only auxiliary batch. "
            "The primary ACT batch remains uniformly shuffled."
        ),
    )
    parser.add_argument(
        "--hard-aux-transition-radius",
        type=int,
        default=0,
        help="Symmetric row radius around each hard auxiliary transition.",
    )
    parser.add_argument(
        "--hard-aux-batch-size",
        type=int,
        default=0,
        help="Number of hard-only anchors appended to each normal ACT batch.",
    )
    parser.add_argument(
        "--hard-aux-rowmax-weight",
        type=float,
        default=0.0,
        help="Raw token-0 row-max loss coefficient applied only to the hard auxiliary batch.",
    )
    parser.add_argument(
        "--hard-aux-error-checkpoint",
        type=Path,
        default=None,
        help=(
            "Optional checkpoint to evaluate over the full trajectory before training. "
            "Rows whose learned first-token raw error exceeds --hard-aux-error-rowmax-threshold "
            "are added to the same hard auxiliary batch."
        ),
    )
    parser.add_argument(
        "--hard-aux-error-rowmax-threshold",
        type=float,
        default=None,
        help="Positive learned first-token raw row-max threshold for error-mined auxiliary rows.",
    )
    parser.add_argument(
        "--hard-aux-error-radius",
        type=int,
        default=0,
        help="Symmetric expansion radius around every error-mined auxiliary row.",
    )
    args = parser.parse_args()
    if args.steps <= 0 or args.batch_size <= 0 or args.num_workers < 0:
        parser.error("--steps/--batch-size must be positive and --num-workers non-negative")
    if args.save_every <= 0 or args.log_every <= 0 or not math.isfinite(args.learning_rate) or args.learning_rate <= 0:
        parser.error("--save-every/--log-every/--learning-rate must be positive")
    if not math.isfinite(args.weight_decay) or args.weight_decay < 0:
        parser.error("--weight-decay must be finite and non-negative")
    if not math.isfinite(args.raw_token0_l1_weight) or args.raw_token0_l1_weight < 0:
        parser.error("--raw-token0-l1-weight must be finite and non-negative")
    if not math.isfinite(args.raw_token0_rowmax_weight) or args.raw_token0_rowmax_weight < 0:
        parser.error("--raw-token0-rowmax-weight must be finite and non-negative")
    if args.hard_transition_threshold is not None and (
        not math.isfinite(args.hard_transition_threshold) or args.hard_transition_threshold < 0
    ):
        parser.error("--hard-transition-threshold must be finite and non-negative when provided")
    if args.hard_transition_radius < 0:
        parser.error("--hard-transition-radius must be non-negative")
    if not math.isfinite(args.hard_sample_multiplier) or args.hard_sample_multiplier < 1:
        parser.error("--hard-sample-multiplier must be finite and at least 1")
    if args.hard_sample_multiplier != 1.0 and args.hard_transition_threshold is None:
        parser.error("--hard-sample-multiplier above 1 requires --hard-transition-threshold")
    if args.critical_transition_threshold is not None and (
        not math.isfinite(args.critical_transition_threshold) or args.critical_transition_threshold < 0
    ):
        parser.error("--critical-transition-threshold must be finite and non-negative when provided")
    if args.critical_transition_threshold is not None and args.hard_transition_threshold is None:
        parser.error("--critical-transition-threshold requires --hard-transition-threshold")
    if (
        args.critical_transition_threshold is not None
        and args.critical_transition_threshold < args.hard_transition_threshold
    ):
        parser.error("--critical-transition-threshold must be at least --hard-transition-threshold")
    if not math.isfinite(args.critical_sample_multiplier) or args.critical_sample_multiplier < 1:
        parser.error("--critical-sample-multiplier must be finite and at least 1")
    if args.critical_sample_multiplier != 1.0 and args.critical_transition_threshold is None:
        parser.error("--critical-sample-multiplier above 1 requires --critical-transition-threshold")
    if args.hard_aux_transition_threshold is not None and (
        not math.isfinite(args.hard_aux_transition_threshold) or args.hard_aux_transition_threshold < 0
    ):
        parser.error("--hard-aux-transition-threshold must be finite and non-negative when provided")
    if args.hard_aux_transition_radius < 0:
        parser.error("--hard-aux-transition-radius must be non-negative")
    if args.hard_aux_batch_size < 0:
        parser.error("--hard-aux-batch-size must be non-negative")
    if not math.isfinite(args.hard_aux_rowmax_weight) or args.hard_aux_rowmax_weight < 0:
        parser.error("--hard-aux-rowmax-weight must be finite and non-negative")
    if args.hard_aux_error_rowmax_threshold is not None and (
        not math.isfinite(args.hard_aux_error_rowmax_threshold)
        or args.hard_aux_error_rowmax_threshold <= 0.0
    ):
        parser.error("--hard-aux-error-rowmax-threshold must be finite and positive when provided")
    if args.hard_aux_error_radius < 0:
        parser.error("--hard-aux-error-radius must be non-negative")
    if (args.hard_aux_error_checkpoint is None) != (args.hard_aux_error_rowmax_threshold is None):
        parser.error(
            "--hard-aux-error-checkpoint and --hard-aux-error-rowmax-threshold must be supplied together"
        )
    hard_aux_enabled = args.hard_aux_batch_size > 0 or args.hard_aux_rowmax_weight > 0.0
    hard_aux_has_pool_source = (
        args.hard_aux_transition_threshold is not None or args.hard_aux_error_checkpoint is not None
    )
    if hard_aux_enabled and not hard_aux_has_pool_source:
        parser.error("hard auxiliary loss requires a transition threshold or an error-mined checkpoint")
    if args.hard_aux_error_checkpoint is not None and not hard_aux_enabled:
        parser.error("error-mined auxiliary rows require a positive --hard-aux-batch-size and rowmax weight")
    if hard_aux_enabled and args.hard_aux_batch_size == 0:
        parser.error("hard auxiliary loss requires a positive --hard-aux-batch-size")
    if hard_aux_enabled and args.hard_aux_rowmax_weight == 0.0:
        parser.error("hard auxiliary loss requires a positive --hard-aux-rowmax-weight")
    if args.override_learning_rate is not None and (
        not math.isfinite(args.override_learning_rate) or args.override_learning_rate <= 0
    ):
        parser.error("--override-learning-rate must be positive when provided")
    if args.mode == "export" and args.checkpoint is None:
        parser.error("--mode export requires --checkpoint")
    if args.resume and args.checkpoint is None:
        parser.error("--resume requires --checkpoint")
    if args.resume and args.init_v1_checkpoint is not None:
        parser.error("--resume and --init-v1-checkpoint are mutually exclusive")
    if args.init_v1_checkpoint is not None and args.time_feature_mode != "native_index_fourier_v1":
        parser.error("--init-v1-checkpoint requires --time-feature-mode native_index_fourier_v1")
    return args


def initialize_index_fourier_from_v1(
    model: TimeIndexedReplayACT, data: NativeReplayData, checkpoint_path: Path
) -> dict[str, Any]:
    """Warm-start a 58-D index-Fourier ACT branch from a 24-D phase model.

    This is intentionally not ``--resume``: AdamW moments depend on the old
    input projection shape and are discarded.  The old 24 columns are copied
    exactly, all new Fourier columns start at zero, and the v2 data buffers
    keep their freshly calculated feature normalizers.  Therefore the initial
    learned ACT mapping is bit-for-bit the old phase-only mapping until the
    new time channels acquire nonzero weights.
    """

    checkpoint_path = checkpoint_path.expanduser().resolve()
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if payload.get("model_kind") != "TimeIndexedReplayACT":
        raise ValueError(f"--init-v1-checkpoint is not a TimeIndexedReplayACT: {checkpoint_path}")
    source = payload.get("source", {})
    if source.get("action_timestamp_action_sha256") != data.source_sha256:
        raise ValueError("--init-v1-checkpoint was trained on another action stream")
    if source.get("state_timestamp_state_sha256") != data.state_sha256:
        raise ValueError("--init-v1-checkpoint was trained on another state stream")
    old_state = payload.get("model_state_dict")
    if not isinstance(old_state, dict):
        raise ValueError("--init-v1-checkpoint has no model_state_dict")

    old_state_mean = old_state.get("state_mean")
    old_state_std = old_state.get("state_std")
    if not isinstance(old_state_mean, Tensor) or not isinstance(old_state_std, Tensor):
        raise ValueError("--init-v1-checkpoint has no state normalizer buffers")
    if old_state_mean.shape != (STATE_WITH_TIME_DIM,) or old_state_std.shape != (STATE_WITH_TIME_DIM,):
        raise ValueError(
            "--init-v1-checkpoint must use the original 24-D phase state input; "
            f"got normalizer shapes {tuple(old_state_mean.shape)}/{tuple(old_state_std.shape)}"
        )
    if data.causal_states_with_time.shape[1] <= STATE_WITH_TIME_DIM:
        raise ValueError("index-Fourier warm start requires additional native time feature columns")
    if not torch.equal(model.state_mean[:STATE_WITH_TIME_DIM].cpu(), old_state_mean.cpu()):
        raise AssertionError("v2 first 24 state means differ from the phase-only initialization checkpoint")
    if not torch.equal(model.state_std[:STATE_WITH_TIME_DIM].cpu(), old_state_std.cpu()):
        raise AssertionError("v2 first 24 state stds differ from the phase-only initialization checkpoint")

    target_state = model.state_dict()
    copied: list[str] = []
    skipped_normalizers: list[str] = []
    widened_projection: str | None = None
    mismatches: list[str] = []
    for name, target_value in target_state.items():
        old_value = old_state.get(name)
        if not isinstance(old_value, Tensor):
            mismatches.append(f"missing:{name}")
            continue
        if old_value.shape == target_value.shape:
            target_state[name] = old_value.detach().clone().to(dtype=target_value.dtype)
            copied.append(name)
            continue
        if name in {"state_mean", "state_std"}:
            skipped_normalizers.append(name)
            continue
        if (
            name.endswith("encoder_robot_state_input_proj.weight")
            and old_value.ndim == 2
            and target_value.ndim == 2
            and old_value.shape[0] == target_value.shape[0]
            and old_value.shape[1] == STATE_WITH_TIME_DIM
            and target_value.shape[1] == data.causal_states_with_time.shape[1]
        ):
            widened = target_value.detach().clone()
            widened.zero_()
            widened[:, :STATE_WITH_TIME_DIM] = old_value.to(dtype=target_value.dtype)
            target_state[name] = widened
            widened_projection = name
            continue
        mismatches.append(f"{name}:{tuple(old_value.shape)}->{tuple(target_value.shape)}")
    if mismatches:
        raise ValueError("unexpected v1->index-Fourier checkpoint mismatch: " + "; ".join(mismatches))
    if widened_projection is None:
        raise ValueError("could not find the ACT robot-state input projection to widen")
    if set(skipped_normalizers) != {"state_mean", "state_std"}:
        raise ValueError(f"unexpected v1 normalizer migration outcome: {skipped_normalizers}")
    model.load_state_dict(target_state, strict=True)
    return {
        "checkpoint": str(checkpoint_path),
        "source_trained_steps": int(payload.get("trained_steps", 0)),
        "copied_tensor_count": len(copied),
        "widened_projection": widened_projection,
        "new_time_projection_columns_initialized_to": 0.0,
        "optimizer_state_reused": False,
    }


def train(args: argparse.Namespace) -> None:
    data = load_native_replay(args.npz, time_feature_mode=args.time_feature_mode)
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    final_checkpoint = output_dir / "exact_replay_act_final.pt"
    if final_checkpoint.exists() and not args.resume:
        raise FileExistsError(f"refusing to overwrite completed model: {final_checkpoint}")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = choose_device(args.device)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
        # ``fp32`` is a precision mode, not merely AMP-off: TF32 retains only
        # a 10-bit mantissa and aliases many adjacent native-clock time
        # values before the first state projection.  Disable it together with
        # AMP for strict time-conditioned overfitting.
        allow_tf32 = args.precision == "bf16"
        torch.backends.cuda.matmul.allow_tf32 = allow_tf32
        torch.backends.cudnn.allow_tf32 = allow_tf32
        if args.precision == "fp32":
            torch.set_float32_matmul_precision("highest")

    dataset = NativeActionChunkDataset(data, ACT_ARCHITECTURE["chunk_size"])
    generator = torch.Generator().manual_seed(args.seed)
    anchor_sampler, hard_sampling = make_hard_transition_sampler(
        data.actions,
        threshold=args.hard_transition_threshold,
        radius=args.hard_transition_radius,
        multiplier=args.hard_sample_multiplier,
        critical_threshold=args.critical_transition_threshold,
        critical_multiplier=args.critical_sample_multiplier,
        generator=generator,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=anchor_sampler is None,
        sampler=anchor_sampler,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
        generator=generator,
    )
    hard_aux_loader: DataLoader[dict[str, Tensor]] | None = None
    hard_aux_sampling: dict[str, Any] = {
        "enabled": False,
        "transition_threshold": args.hard_aux_transition_threshold,
        "transition_radius": int(args.hard_aux_transition_radius),
        "error_checkpoint": (
            None
            if args.hard_aux_error_checkpoint is None
            else str(args.hard_aux_error_checkpoint.expanduser().resolve())
        ),
        "error_rowmax_threshold": args.hard_aux_error_rowmax_threshold,
        "error_radius": int(args.hard_aux_error_radius),
        "batch_size": int(args.hard_aux_batch_size),
        "rowmax_weight": float(args.hard_aux_rowmax_weight),
    }
    if (
        args.hard_aux_batch_size > 0
        and (args.hard_aux_transition_threshold is not None or args.hard_aux_error_checkpoint is not None)
    ):
        hard_aux_mask = np.zeros(len(data.actions), dtype=bool)
        hard_transition_rows = np.empty(0, dtype=np.int64)
        error_mined_rows = np.empty(0, dtype=np.int64)
        error_mined_report: dict[str, Any] | None = None
        if args.hard_aux_transition_threshold is not None:
            hard_transition_rows, transition_mask = transition_anchor_mask(
                data.actions,
                threshold=args.hard_aux_transition_threshold,
                radius=args.hard_aux_transition_radius,
            )
            hard_aux_mask |= transition_mask
        if args.hard_aux_error_checkpoint is not None:
            error_mined_rows, error_mask, error_mined_report = error_mined_anchor_mask(
                args.hard_aux_error_checkpoint,
                data,
                threshold=float(args.hard_aux_error_rowmax_threshold),
                radius=args.hard_aux_error_radius,
                batch_size=max(1024, args.batch_size * 64),
                device=device,
            )
            hard_aux_mask |= error_mask
        hard_anchor_indices = np.flatnonzero(hard_aux_mask)
        if len(hard_anchor_indices) < args.hard_aux_batch_size:
            raise ValueError(
                "hard auxiliary pool is smaller than --hard-aux-batch-size: "
                f"{len(hard_anchor_indices)} < {args.hard_aux_batch_size}"
            )
        hard_aux_sampling.update(
            {
                "enabled": True,
                "transition_count": int(len(hard_transition_rows)),
                "error_mined_row_count": int(len(error_mined_rows)),
                "hard_anchor_rows": int(len(hard_anchor_indices)),
                "hard_anchor_fraction_uniform": float(len(hard_anchor_indices) / len(dataset)),
                "error_mined": error_mined_report,
            }
        )
        hard_aux_generator = torch.Generator().manual_seed(args.seed + 1)
        hard_aux_loader = DataLoader(
            Subset(dataset, hard_anchor_indices.tolist()),
            batch_size=args.hard_aux_batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=device.type == "cuda",
            persistent_workers=args.num_workers > 0,
            drop_last=True,
            generator=hard_aux_generator,
        )
    model = TimeIndexedReplayACT(make_act_policy(int(data.causal_states_with_time.shape[1])), data).to(device)
    initialization: dict[str, Any] | None = None
    if args.init_v1_checkpoint is not None:
        # Initialize on the final target device only after buffers/weights are
        # already present.  The helper loads the source checkpoint on CPU and
        # copies tensors into this model without ever touching its optimizer.
        initialization = initialize_index_fourier_from_v1(model, data, args.init_v1_checkpoint)
    optimizer = torch.optim.AdamW(
        model.act_policy.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    start_step = 0
    last_loss: float | None = None
    if args.resume:
        checkpoint = torch.load(args.checkpoint.expanduser().resolve(), map_location="cpu", weights_only=False)
        source_metadata = checkpoint.get("source", {})
        if source_metadata.get("action_timestamp_action_sha256") != data.source_sha256:
            raise ValueError("resume checkpoint was trained on another action stream")
        if source_metadata.get("state_timestamp_state_sha256") != data.state_sha256:
            raise ValueError("resume checkpoint was trained on another odom-state stream")
        checkpoint_state_dim = int(checkpoint["model_state_dict"]["state_mean"].numel())
        if checkpoint_state_dim != data.causal_states_with_time.shape[1]:
            raise ValueError(
                "resume checkpoint time-feature dimension differs from this training invocation: "
                f"checkpoint={checkpoint_state_dim}, requested={data.causal_states_with_time.shape[1]}"
            )
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        if not args.reset_optimizer:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            for state in optimizer.state.values():
                for key, value in state.items():
                    if isinstance(value, Tensor):
                        state[key] = value.to(device)
        if args.override_learning_rate is not None:
            for group in optimizer.param_groups:
                group["lr"] = float(args.override_learning_rate)
        start_step = int(checkpoint["trained_steps"])
        last_loss = checkpoint.get("last_loss")
        if start_step >= args.steps:
            raise ValueError(f"resume checkpoint already has {start_step} steps, target is {args.steps}")

    metadata = {
        "model_kind": "TimeIndexedReplayACT",
        "source_npz": str(data.source_path),
        "source_action_timestamp_action_sha256": data.source_sha256,
        "source_state_timestamp_state_sha256": data.state_sha256,
        "source_native_state_action_content_sha256": data.full_content_sha256,
        "native_state_samples": int(len(data.states)),
        "native_action_samples": int(len(data.actions)),
        "native_action_duration_s": float(data.action_timestamps_s[-1] - data.action_timestamps_s[0]),
        "bootstrap_action_rows_without_prior_odom": int(data.bootstrap_action_rows),
        "input_contract": (
            "state[:23] causal ZOH; state[20:23]=measured odom velocity; "
            "state[23]=normalized physical time phase"
            + (
                "; state[24:]=deterministic native-index Fourier clock"
                if data.time_feature_spec["mode"] == "native_index_fourier_v1"
                else ""
            )
        ),
        "target_contract": "action[:23], including action[20:23]=twist/cmd",
        "act_architecture": ACT_ARCHITECTURE,
        "act_config": act_config_metadata(int(data.causal_states_with_time.shape[1])),
        "time_feature_spec": data.time_feature_spec,
        "initialization": initialization,
        "steps_target": int(args.steps),
        "batch_size": int(args.batch_size),
        "device": str(device),
        "optimizer_learning_rate": float(optimizer.param_groups[0]["lr"]),
        "optimizer_weight_decay": float(optimizer.param_groups[0]["weight_decay"]),
        "resume_learning_rate_override": args.override_learning_rate,
        "resume_optimizer_reset": bool(args.resume and args.reset_optimizer),
        "raw_token0_l1_weight": float(args.raw_token0_l1_weight),
        "raw_token0_rowmax_weight": float(args.raw_token0_rowmax_weight),
        "hard_transition_sampling": hard_sampling,
        "hard_auxiliary_sampling": hard_aux_sampling,
        "training_compute_precision": args.precision,
        "training_tf32_enabled": bool(device.type == "cuda" and args.precision == "bf16"),
        "exact_execution": "frozen timestamp-indexed float32 action bank embedded in the same ACT module",
    }
    atomic_json_write(output_dir / "metadata.json", metadata)
    print(
        "[exact-act] "
        f"native actions={len(data.actions)} duration={metadata['native_action_duration_s']:.6f}s "
        f"bootstrap_odom_rows={data.bootstrap_action_rows} device={device}"
    )

    iterator = iter(loader)
    hard_aux_iterator = iter(hard_aux_loader) if hard_aux_loader is not None else None
    model.train()
    for step in range(start_step + 1, args.steps + 1):
        try:
            primary_batch = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            primary_batch = next(iterator)
        primary_batch_size = int(primary_batch[OBS_STATE].shape[0])
        if hard_aux_iterator is not None:
            try:
                hard_aux_batch = next(hard_aux_iterator)
            except StopIteration:
                hard_aux_iterator = iter(hard_aux_loader)
                hard_aux_batch = next(hard_aux_iterator)
            batch = {
                name: torch.cat((primary_batch[name], hard_aux_batch[name]), dim=0)
                for name in primary_batch
            }
        else:
            batch = primary_batch
        batch = {name: value.to(device, non_blocking=device.type == "cuda") for name, value in batch.items()}
        optimizer.zero_grad(set_to_none=True)
        # Native action time is a continuous scalar at roughly 300 Hz.  bf16
        # quantizes its normalized values into far fewer bins, which aliases
        # nearby action switches and prevents the ordinary ACT branch from
        # overfitting the trajectory.  The exact model head remains float32 in
        # either mode; fp32 is available for precision-critical fine tuning.
        autocast_enabled = device.type == "cuda" and args.precision == "bf16"
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=autocast_enabled):
            if (
                args.raw_token0_l1_weight == 0.0
                and args.raw_token0_rowmax_weight == 0.0
                and hard_aux_iterator is None
            ):
                loss, loss_dict = model.act_policy(batch)
            else:
                # ACTPolicy.forward only supports per-dimension loss weights,
                # broadcast over every chunk token.  For the deployment token
                # we need an explicit raw-float32 objective, while retaining
                # the normal all-token loss as a stabilizing auxiliary term.
                policy = model.act_policy
                policy._set_action_loss_weights_for_step(int(policy._action_loss_schedule_step.item()))
                predicted_normalized, _ = policy.model(batch)
                primary_prediction = predicted_normalized[:primary_batch_size]
                primary_action = batch[ACTION][:primary_batch_size]
                primary_is_pad = batch["action_is_pad"][:primary_batch_size]
                primary_raw_action = batch["action_raw"][:primary_batch_size]
                absolute_error = (primary_action - primary_prediction).abs()
                valid_mask = (~primary_is_pad.unsqueeze(-1)).to(dtype=absolute_error.dtype)
                action_weights = policy._action_loss_weights.to(dtype=absolute_error.dtype)
                weighted_mask = valid_mask * action_weights
                base_l1 = (absolute_error * weighted_mask).sum() / weighted_mask.sum().clamp_min(1)
                predicted_raw = primary_prediction * model.action_std + model.action_mean
                raw_token0_l1 = (predicted_raw[:, 0].float() - primary_raw_action[:, 0].float()).abs().mean()
                loss = base_l1 + args.raw_token0_l1_weight * raw_token0_l1
                loss_dict = {
                    "l1_loss": float(base_l1.detach().item()),
                    "raw_token0_l1": float(raw_token0_l1.detach().item()),
                }
                if args.raw_token0_rowmax_weight > 0.0:
                    raw_token0_rowmax = (
                        predicted_raw[:, 0].float() - primary_raw_action[:, 0].float()
                    ).abs().amax(dim=-1).mean()
                    loss = loss + args.raw_token0_rowmax_weight * raw_token0_rowmax
                    loss_dict["raw_token0_rowmax"] = float(raw_token0_rowmax.detach().item())
                if hard_aux_iterator is not None:
                    hard_aux_predicted_raw = (
                        predicted_normalized[primary_batch_size:] * model.action_std + model.action_mean
                    )
                    hard_aux_target_raw = batch["action_raw"][primary_batch_size:]
                    hard_aux_rowmax = (
                        hard_aux_predicted_raw[:, 0].float() - hard_aux_target_raw[:, 0].float()
                    ).abs().amax(dim=-1).mean()
                    loss = loss + args.hard_aux_rowmax_weight * hard_aux_rowmax
                    loss_dict["hard_aux_rowmax"] = float(hard_aux_rowmax.detach().item())
        if not torch.isfinite(loss):
            raise FloatingPointError(f"non-finite loss at step {step}: {loss.item()}")
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.act_policy.parameters(), max_norm=10.0)
        if not torch.isfinite(grad_norm):
            raise FloatingPointError(f"non-finite gradient norm at step {step}: {grad_norm.item()}")
        optimizer.step()
        last_loss = float(loss.detach().item())

        if step % args.log_every == 0 or step == 1:
            raw_token0_text = ""
            if "raw_token0_l1" in loss_dict:
                raw_token0_text = f" raw_token0={loss_dict['raw_token0_l1']:.8f}"
            if "raw_token0_rowmax" in loss_dict:
                raw_token0_text += f" raw_token0_rowmax={loss_dict['raw_token0_rowmax']:.8f}"
            if "hard_aux_rowmax" in loss_dict:
                raw_token0_text += f" hard_aux_rowmax={loss_dict['hard_aux_rowmax']:.8f}"
            print(
                f"[exact-act] step={step}/{args.steps} loss={last_loss:.8f} "
                f"l1={loss_dict['l1_loss']:.8f}{raw_token0_text} "
                f"grad_norm={float(grad_norm):.6f}",
                flush=True,
            )
        if step % args.save_every == 0:
            checkpoint_path = save_checkpoint(
                output_dir, model, optimizer, step=step, data=data, last_loss=last_loss
            )
            atomic_json_write(
                output_dir / "training_progress.json",
                {"step": step, "target_steps": args.steps, "last_loss": last_loss, "checkpoint": str(checkpoint_path)},
            )

    final_path = save_checkpoint(
        output_dir, model, optimizer, step=args.steps, data=data, last_loss=last_loss, final=True
    )
    export_path = args.export_npz or (output_dir / "model_exact_replay_output.npz")
    report = export_model_replay_npz(final_path, export_path, compare_source=data)
    atomic_json_write(output_dir / "training_progress.json", {
        "step": args.steps,
        "target_steps": args.steps,
        "last_loss": last_loss,
        "checkpoint": str(final_path),
        "exact_replay_verification": report,
    })
    print(f"[exact-act] COMPLETE checkpoint={final_path}", flush=True)
    print(f"[exact-act] exact replay output={export_path}", flush=True)


def main() -> None:
    args = parse_args()
    if args.mode == "export":
        output_path = args.export_npz or (args.output_dir.expanduser().resolve() / "model_exact_replay_output.npz")
        report = export_model_replay_npz(args.checkpoint, output_path)
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return
    train(args)


if __name__ == "__main__":
    main()
