#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# USER may be unset in non-login shells (e.g. batch runners with `set -u`).
CURRENT_USER="${USER:-$(id -un)}"

DEFAULT_VENV_PATH="$REPO_ROOT/.venv_${CURRENT_USER}"
if [[ ! -f "$DEFAULT_VENV_PATH/bin/activate" ]]; then
  DEFAULT_VENV_PATH="$REPO_ROOT/.venv"
fi

VENV_PATH="${VENV_PATH:-$DEFAULT_VENV_PATH}"
HARNESS_DIR="${HARNESS_DIR:-$REPO_ROOT/lm-evaluation-harness}"
HARNESS_TEST="${HARNESS_TEST:-$HARNESS_DIR/test.sh}"

set_wandb_run_name_from_slurm_script() {
  local script_path="${1:-}"
  local script_name

  if [[ -z "$script_path" ]]; then
    return 0
  fi

  script_name="$(basename "$script_path")"
  case "$script_name" in
    *.slurm) ;;
    *) return 0 ;;
  esac

  script_name="${script_name%.slurm}"
  export DEEPLOOP_SLURM_SCRIPT_NAME="${DEEPLOOP_SLURM_SCRIPT_NAME:-$script_name}"
}

_runtime_caller="${BASH_SOURCE[1]:-}"
if [[ -z "${DEEPLOOP_SLURM_SCRIPT_NAME:-}" && -n "$_runtime_caller" ]]; then
  set_wandb_run_name_from_slurm_script "$_runtime_caller"
fi
unset _runtime_caller

activate_venv() {
  if [[ -x "$VENV_PATH/bin/python" ]]; then
    export VIRTUAL_ENV="$VENV_PATH"
    case ":$PATH:" in
      *":$VENV_PATH/bin:"*) ;;
      *) export PATH="$VENV_PATH/bin:$PATH" ;;
    esac
    hash -r
  else
    echo "Venv not found: $VENV_PATH"
    echo "Set VENV_PATH or create it (e.g., python3 -m venv $VENV_PATH)."
    exit 1
  fi
}

# Some venv console-script shims (notably torchrun) may embed a stale interpreter
# path after the repo or venv is moved. Route through the active Python instead.
torchrun() {
  python -m torch.distributed.run "$@"
}
