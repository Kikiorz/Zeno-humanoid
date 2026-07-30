#!/usr/bin/env bash
# Mirror immutable completed checkpoints from a Vast host to the local project.
#
# Vast /workspace is ephemeral.  A checkpoint is copied into a local hidden
# staging directory and renamed atomically only after rsync completes, so a
# locally visible checkpoint is always loadable.  This script intentionally
# never deletes either local or remote data.
set -euo pipefail

if (( $# > 0 )); then
  printf 'Configure this script with environment variables, not positional arguments.\n' >&2
  exit 2
fi

REMOTE_HOST="${REMOTE_HOST:-root@211.72.13.201}"
REMOTE_PORT="${REMOTE_PORT:-42449}"
REMOTE_RUN_DIR="${REMOTE_RUN_DIR:-/workspace/2027icra/outputs/train/robot8_20260726_act_dinov3_3cam_640x480_nocrop_all23_decoder7_b32x2_100k}"
LOCAL_RUN_DIR="${LOCAL_RUN_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)/outputs/train/robot8_20260726_act_dinov3_3cam_640x480_nocrop_all23_decoder7_b32x2_100k}"
POLL_SECONDS="${POLL_SECONDS:-60}"
ONCE="${ONCE:-false}"

if ! [[ "${REMOTE_PORT}" =~ ^[0-9]+$ && "${POLL_SECONDS}" =~ ^[0-9]+$ ]] || (( REMOTE_PORT < 1 || POLL_SECONDS < 1 )); then
  printf 'REMOTE_PORT and POLL_SECONDS must be positive integers.\n' >&2
  exit 2
fi
case "${ONCE}" in
  true|false) ;;
  *) printf 'ONCE must be true or false.\n' >&2; exit 2 ;;
esac

ssh_cmd=(ssh -p "${REMOTE_PORT}" -o BatchMode=yes -o ConnectTimeout=15 "${REMOTE_HOST}")
rsync_ssh="ssh -p ${REMOTE_PORT} -o BatchMode=yes -o ConnectTimeout=15"
mkdir -p "${LOCAL_RUN_DIR}/checkpoints"

remote_checkpoints() {
  "${ssh_cmd[@]}" \
    "find '${REMOTE_RUN_DIR}/checkpoints' -mindepth 1 -maxdepth 1 -type d -printf '%f\\n' 2>/dev/null | sort" \
    2>/dev/null || true
}

remote_checkpoint_ready() {
  local step="$1"
  # LeRobot updates the `last` symlink only after policy, processor, and full
  # training state save successfully.  A mere training_step.json is not a
  # completion marker because that file is written before optimizer_state.
  "${ssh_cmd[@]}" "last=\$(readlink '${REMOTE_RUN_DIR}/checkpoints/last' 2>/dev/null || true)
case \"\${last}\" in
  [0-9][0-9][0-9][0-9][0-9][0-9]) ;;
  *) exit 1 ;;
esac
[ \"\${last}\" -ge \"${step}\" ] && \\
test -s '${REMOTE_RUN_DIR}/checkpoints/${step}/pretrained_model/model.safetensors' && \\
test -s '${REMOTE_RUN_DIR}/checkpoints/${step}/training_state/optimizer_state.safetensors' && \\
test -s '${REMOTE_RUN_DIR}/checkpoints/${step}/training_state/training_step.json'" >/dev/null 2>&1
}

sync_once() {
  local step final_dir incoming_dir
  while IFS= read -r step; do
    [[ "${step}" =~ ^[0-9]{6}$ ]] || continue
    final_dir="${LOCAL_RUN_DIR}/checkpoints/${step}"
    [[ -d "${final_dir}" ]] && continue
    remote_checkpoint_ready "${step}" || continue

    incoming_dir="${LOCAL_RUN_DIR}/checkpoints/.${step}.incoming"
    mkdir -p "${incoming_dir}"
    printf '[%s] syncing checkpoint %s from %s\n' "$(date '+%F %T')" "${step}" "${REMOTE_HOST}"
    if ! rsync -a --partial --append-verify --human-readable --info=stats1,name0 \
      -e "${rsync_ssh}" \
      "${REMOTE_HOST}:${REMOTE_RUN_DIR}/checkpoints/${step}/" "${incoming_dir}/"; then
      printf '[%s] checkpoint %s sync did not complete; staging directory retained for retry\n' \
        "$(date '+%F %T')" "${step}" >&2
      continue
    fi
    if [[ ! -s "${incoming_dir}/pretrained_model/model.safetensors" || \
          ! -s "${incoming_dir}/training_state/optimizer_state.safetensors" || \
          ! -s "${incoming_dir}/training_state/training_step.json" ]]; then
      printf '[%s] checkpoint %s staging validation failed; retaining staging directory\n' \
        "$(date '+%F %T')" "${step}" >&2
      continue
    fi
    if [[ -e "${final_dir}" ]]; then
      printf '[%s] checkpoint %s was mirrored by another process; leaving staging intact\n' \
        "$(date '+%F %T')" "${step}" >&2
      continue
    fi
    mv "${incoming_dir}" "${final_dir}"
    printf '[%s] checkpoint %s mirrored atomically to %s\n' \
      "$(date '+%F %T')" "${step}" "${final_dir}"
  done < <(remote_checkpoints)
}

while true; do
  sync_once
  [[ "${ONCE}" == true ]] && break
  sleep "${POLL_SECONDS}"
done
