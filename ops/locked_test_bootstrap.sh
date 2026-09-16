#!/usr/bin/env bash
# Rebuild the frozen Locked Test runtime on a FRESH Vast.ai 1x RTX 5090 box.
#
# Target: the exact runtime recorded in environment.lock / environment-gpu.freeze
# (sha256 d738fb679db3682292481dfb74154b2d1d22da37630fd0156092c281ff31f821), so
# that `ops/locked_test_supervisor.py --plan-only` passes its `remote_runtime`
# preflight (Python 3.12.13 at /venv/main, torch 2.11.0+cu130 / CUDA 13.0,
# torchvision 0.26.0+cu130, lerobot 0.6.0, hf-libero 0.1.4, mujoco 3.8.1,
# transformers 5.5.4, huggingface-hub 1.18.0, egl-probe 1.0.2, hf-egl-probe 1.0.2,
# one RTX 5090 with compute capability 12.0).
#
# Runs ON the instance as root from a full clone at $CHECKOUT whose HEAD equals
# the tag calibration-locked-v1.  Every step is idempotent: it is skipped when
# its verified output already exists.  Nothing here reads or writes Locked Test
# data (manifest, authority, artifacts); the final self-check only resolves the
# pinned model snapshots offline and loads the policy.
#
# Usage (as root on the instance):
#   bash /workspace/locked-test-checkout/ops/locked_test_bootstrap.sh
# Optional overrides: WORKSPACE, CHECKOUT, VENV, HF_TOKEN (for Hub downloads).
set -euo pipefail

# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------
WORKSPACE="${WORKSPACE:-/workspace}"
CHECKOUT="${CHECKOUT:-$WORKSPACE/locked-test-checkout}"
VENV="${VENV:-/venv/main}"
PY="$VENV/bin/python"
WHEELS="$WORKSPACE/wheels"
HF_CACHE="$WORKSPACE/hf-cache"
LEROBOT_DIR="$WORKSPACE/lerobot"
ASSET_STAGING="$WORKSPACE/libero-assets-0b3ea86"
INSTALL_LOG="$WORKSPACE/install.log"
STAMP_DIR="$WORKSPACE/.bootstrap-stamps"
REQUIRED_TAG="calibration-locked-v1"
LOCK_FILE="$CHECKOUT/environment.lock"
FREEZE_FILE="$CHECKOUT/environment-gpu.freeze"
FREEZE_SHA256="d738fb679db3682292481dfb74154b2d1d22da37630fd0156092c281ff31f821"
ASSET_MANIFEST="$CHECKOUT/artifacts/manifests/libero-assets-0b3ea86.manifest.tsv"
ASSET_MANIFEST_SHA256="7abae99d7c844189e89376b4b5f80b4bb1dedae9e0390845567a452712b74492"
EXPECTED_PYTHON="3.12.13"

# Pinned wheels.  Hashes are the ones recorded in environment.lock
# (torch_wheel_sha256 / torchvision_wheel_sha256) and were cross-checked against
# PyPI's JSON (torch 2.11.0 cp312 manylinux_2_28_x86_64) and the
# download.pytorch.org cu130 index (torchvision 0.26.0+cu130) on 2026-09-16.
TORCH_WHEEL="torch-2.11.0-cp312-cp312-manylinux_2_28_x86_64.whl"
TORCH_URL="https://files.pythonhosted.org/packages/1a/c9/82638ef24d7877510f83baf821f5619a61b45568ce21c0a87a91576510aa/$TORCH_WHEEL"
TORCH_SHA256="0f68f4ac6d95d12e896c3b7a912b5871619542ec54d3649cf48cc1edd4dd2756"
TORCHVISION_WHEEL="torchvision-0.26.0+cu130-cp312-cp312-manylinux_2_28_x86_64.whl"
TORCHVISION_URL="https://download.pytorch.org/whl/cu130/torchvision-0.26.0%2Bcu130-cp312-cp312-manylinux_2_28_x86_64.whl"
TORCHVISION_SHA256="0f030a9bd8ada1a31b7111ea1589c1ecb5fa0884fee700a203e731b4cf378a98"

# System packages.  EGL/MuJoCo headless rendering needs the GLVND EGL/GL
# front-ends plus Mesa's EGL/OSMesa/DRI back-ends (the NVIDIA EGL ICD comes from
# the container runtime); glib for OpenCV; cmake/make/g++ build the egl_probe and
# hf_egl_probe sdists; tmux/zstd/rsync/curl/git are operational tooling.
APT_PACKAGES=(
  libegl1 libgl1 libgles2 libglvnd0 libegl-mesa0 libgl1-mesa-dri libosmesa6
  libglib2.0-0 libxext6 libx11-6 libxrender1 libsm6
  cmake build-essential pkg-config
  tmux zstd rsync curl git ca-certificates
)

# ---------------------------------------------------------------------------
# Logging / helpers
# ---------------------------------------------------------------------------
mkdir -p "$WORKSPACE" "$STAMP_DIR"
exec > >(tee -a "$INSTALL_LOG") 2>&1

log()  { printf '[%s] %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*"; }
step() { log "=== $* ==="; }
die()  { log "FATAL: $*"; exit 1; }

sha256_of() { sha256sum "$1" | awk '{print $1}'; }

# Read a `key = "value"` entry from environment.lock (TOML, simple string values).
lock_value() {
  local value
  value="$(grep -E "^$1 = " "$LOCK_FILE" | head -n1 | sed -E 's/^[^=]+= *"?([^"]*)"?.*$/\1/')"
  [[ -n "$value" ]] || die "environment.lock has no value for $1"
  printf '%s' "$value"
}

# Download with retries/resume into $WHEELS and verify sha256; skip if already good.
fetch_verified() {
  local url="$1" dest="$2" expected="$3"
  if [[ -f "$dest" ]] && [[ "$(sha256_of "$dest")" == "$expected" ]]; then
    log "already present and verified: $dest"
    return 0
  fi
  log "downloading $url"
  curl -fL --retry 5 --retry-delay 5 --retry-all-errors -C - -o "$dest" "$url"
  local actual
  actual="$(sha256_of "$dest")"
  [[ "$actual" == "$expected" ]] || die "sha256 mismatch for $dest: $actual != $expected"
  log "verified $dest ($expected)"
}

# Run the repo's own Python with its src on PYTHONPATH.
repo_py() { PYTHONPATH="$CHECKOUT/src" "$PY" "$@"; }

# Inline verifier for the 586-file LIBERO asset tree against the tracked manifest.
# Manifest columns: path size_bytes sha256 verification_kind revision_oid.
# Ordinary files: revision_oid is the git blob SHA-1; LFS files: it is the LFS
# SHA-256 OID.  We check size, sha256 and the kind-specific OID, and require no
# missing/extra files (huggingface_hub's `.cache/` bookkeeping dir is ignored).
verify_asset_tree() {
  local tree="$1"
  "$PY" - "$ASSET_MANIFEST" "$tree" <<'PYEOF'
import hashlib, os, sys
manifest, tree = sys.argv[1], sys.argv[2]
rows = []
with open(manifest, encoding="utf-8") as fh:
    header = fh.readline().rstrip("\n").split("\t")
    assert header == ["path", "size_bytes", "sha256", "verification_kind", "revision_oid"], header
    for line in fh:
        if line.strip():
            rows.append(line.rstrip("\n").split("\t"))
if len(rows) != 586:
    sys.exit(f"manifest lists {len(rows)} files, expected 586")
errors = []
seen = set()
for path, size, sha256, kind, oid in rows:
    seen.add(path)
    full = os.path.join(tree, path)
    if os.path.islink(full) or not os.path.isfile(full):
        errors.append(f"missing or not a regular file: {path}"); continue
    data = open(full, "rb").read()
    if len(data) != int(size):
        errors.append(f"size mismatch: {path} {len(data)} != {size}"); continue
    if hashlib.sha256(data).hexdigest() != sha256:
        errors.append(f"sha256 mismatch: {path}"); continue
    if kind == "git-blob-sha1":
        blob = hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()
        if blob != oid:
            errors.append(f"git blob sha1 mismatch: {path} {blob} != {oid}")
    elif kind.startswith("lfs"):
        if sha256 != oid:
            errors.append(f"lfs oid mismatch: {path} {sha256} != {oid}")
    else:
        errors.append(f"unknown verification_kind {kind!r} for {path}")
extra = []
for root, dirs, files in os.walk(tree):
    rel_root = os.path.relpath(root, tree)
    if rel_root == ".cache" or rel_root.startswith(".cache" + os.sep):
        continue
    dirs[:] = [d for d in dirs if not (rel_root == "." and d == ".cache")]
    for name in files:
        rel = os.path.normpath(os.path.join(rel_root, name))
        if rel not in seen:
            extra.append(rel)
if extra:
    errors.append(f"{len(extra)} unexpected files, e.g. {extra[:5]}")
if errors:
    print("\n".join(errors[:20]))
    sys.exit(f"asset verification FAILED with {len(errors)} error(s) under {tree}")
print(f"asset verification OK: {len(rows)} files under {tree}")
PYEOF
}

# ---------------------------------------------------------------------------
# 0. Preconditions
# ---------------------------------------------------------------------------
step "0. preconditions"
[[ "$(id -u)" == "0" ]] || die "run as root"
[[ -x "$PY" ]] || die "missing interpreter $PY (expected the Vast PyTorch image venv)"
actual_python="$("$PY" -c 'import platform; print(platform.python_version())')"
[[ "$actual_python" == "$EXPECTED_PYTHON" ]] || die "python $actual_python != $EXPECTED_PYTHON"
actual_prefix="$("$PY" -c 'import sys; print(sys.prefix)')"
[[ "$actual_prefix" == "$VENV" ]] || die "sys.prefix $actual_prefix != $VENV"
[[ -d "$CHECKOUT/.git" ]] || die "$CHECKOUT is not a git clone"
head_sha="$(git -C "$CHECKOUT" rev-parse HEAD)"
tag_sha="$(git -C "$CHECKOUT" rev-parse "${REQUIRED_TAG}^{commit}")" \
  || die "tag $REQUIRED_TAG is not present in $CHECKOUT (clone with tags)"
[[ "$head_sha" == "$tag_sha" ]] || die "HEAD $head_sha != $REQUIRED_TAG $tag_sha"
[[ -z "$(git -C "$CHECKOUT" status --porcelain)" ]] || die "checkout is not clean"
[[ "$(sha256_of "$FREEZE_FILE")" == "$FREEZE_SHA256" ]] || die "environment-gpu.freeze sha256 mismatch"
[[ "$(sha256_of "$ASSET_MANIFEST")" == "$ASSET_MANIFEST_SHA256" ]] || die "asset manifest sha256 mismatch"
LEROBOT_COMMIT="$(lock_value lerobot_commit)"
LEROBOT_TAG="$(lock_value lerobot_tag)"
LEROBOT_REPO="$(lock_value lerobot_repository)"
ASSETS_REPO="$(lock_value libero_assets_repo)"
ASSETS_REVISION="$(lock_value libero_assets_revision)"
POLICY_MODEL_SHA256="$(lock_value policy_model_sha256)"
log "checkout $CHECKOUT at $head_sha ($REQUIRED_TAG); python $actual_python at $VENV"
log "lerobot $LEROBOT_TAG=$LEROBOT_COMMIT; assets $ASSETS_REPO@$ASSETS_REVISION"
nvidia-smi --query-gpu=index,name,driver_version,compute_cap --format=csv,noheader \
  || die "nvidia-smi failed"

# ---------------------------------------------------------------------------
# 1. System packages
# ---------------------------------------------------------------------------
step "1. apt packages"
missing_pkgs=()
for pkg in "${APT_PACKAGES[@]}"; do
  dpkg -s "$pkg" >/dev/null 2>&1 || missing_pkgs+=("$pkg")
done
if ((${#missing_pkgs[@]} == 0)); then
  log "all apt packages present"
else
  log "installing: ${missing_pkgs[*]}"
  export DEBIAN_FRONTEND=noninteractive
  apt-get update -qq
  apt-get install -y -qq --no-install-recommends "${missing_pkgs[@]}"
fi
mkdir -p "$WHEELS" "$HF_CACHE" "$WORKSPACE/runstate" "$WORKSPACE/run-logs/locked-test" \
  "$WORKSPACE"/research-artifacts/{raw,scores,locked-test-features,postscore,evaluation}

# ---------------------------------------------------------------------------
# 2. torch / torchvision wheels
# ---------------------------------------------------------------------------
step "2. torch + torchvision"
fetch_verified "$TORCH_URL" "$WHEELS/$TORCH_WHEEL" "$TORCH_SHA256"
fetch_verified "$TORCHVISION_URL" "$WHEELS/$TORCHVISION_WHEEL" "$TORCHVISION_SHA256"
dist_version() { "$PY" -c "import importlib.metadata as m,sys; print(m.version(sys.argv[1]))" "$1" 2>/dev/null || true; }
if [[ "$(dist_version torch)" == "2.11.0" ]]; then
  log "torch 2.11.0 already installed"
else
  "$PY" -m pip install --no-deps --force-reinstall "$WHEELS/$TORCH_WHEEL"
fi
if [[ "$(dist_version torchvision)" == "0.26.0+cu130" ]]; then
  log "torchvision 0.26.0+cu130 already installed"
else
  "$PY" -m pip install --no-deps --force-reinstall "$WHEELS/$TORCHVISION_WHEEL"
fi

# ---------------------------------------------------------------------------
# 3. Remaining exact pins from environment-gpu.freeze (--no-deps)
# ---------------------------------------------------------------------------
step "3. frozen pins"
PINS="$WORKSPACE/requirements.frozen.txt"
# Drop comments, the LeRobot editable line, and the two file:// wheel lines
# (handled above); every remaining line is an exact `name==version` pin.
grep -vE '^(#|-e |torch @|torchvision @)' "$FREEZE_FILE" | sed '/^[[:space:]]*$/d' > "$PINS"
grep -qvE '^[A-Za-z0-9_.-]+==[^ ]+$' "$PINS" && die "unexpected non-exact line in $PINS: $(grep -vE '^[A-Za-z0-9_.-]+==[^ ]+$' "$PINS" | head -1)"
check_pins() {
  "$PY" - "$PINS" <<'PYEOF'
import importlib.metadata as m, sys
from packaging.version import Version
bad = []
for line in open(sys.argv[1]):
    name, _, version = line.strip().partition("==")
    try:
        actual = m.version(name)
    except m.PackageNotFoundError:
        bad.append(f"{name}: missing"); continue
    # Compare PEP 440-normalized versions: e.g. thop declares "0.1.1-2209072238"
    # while pip freeze recorded the normalized "0.1.1.post2209072238".
    if Version(actual) != Version(version):
        bad.append(f"{name}: {actual} != {version}")
print("\n".join(bad))
sys.exit(1 if bad else 0)
PYEOF
}
if check_pins >/dev/null; then
  log "all $(wc -l < "$PINS") pins already satisfied"
else
  # egl_probe / hf_egl_probe are cmake sdists declaring cmake_minimum_required 2.8.12;
  # CMake >= 4 refuses that without this override (see log.md SETUP-012).
  CMAKE_POLICY_VERSION_MINIMUM=3.5 "$PY" -m pip install --no-deps -r "$PINS"
  check_pins || die "frozen pins still unsatisfied after install"
fi

# ---------------------------------------------------------------------------
# 4. LeRobot editable at the pinned commit + source hash gate
# ---------------------------------------------------------------------------
step "4. lerobot"
if [[ ! -d "$LEROBOT_DIR/.git" ]]; then
  git clone "$LEROBOT_REPO" "$LEROBOT_DIR"
fi
if [[ "$(git -C "$LEROBOT_DIR" rev-parse HEAD)" != "$LEROBOT_COMMIT" ]]; then
  git -C "$LEROBOT_DIR" fetch --tags origin
  git -C "$LEROBOT_DIR" checkout --detach "$LEROBOT_TAG"
fi
[[ "$(git -C "$LEROBOT_DIR" rev-parse HEAD)" == "$LEROBOT_COMMIT" ]] \
  || die "lerobot HEAD != $LEROBOT_COMMIT (tag $LEROBOT_TAG moved?)"
[[ -z "$(git -C "$LEROBOT_DIR" status --porcelain)" ]] || die "lerobot worktree is dirty"
lerobot_origin="$("$PY" -c 'import importlib.util as u; s=u.find_spec("lerobot"); print(s.origin if s else "")' 2>/dev/null || true)"
if [[ "$(dist_version lerobot)" == "0.6.0" && "$lerobot_origin" == "$LEROBOT_DIR/src/lerobot/__init__.py" ]]; then
  log "lerobot 0.6.0 editable install already present"
else
  "$PY" -m pip install --no-deps -e "$LEROBOT_DIR"
fi
# Fail closed unless the installed 487-file Python tree hashes to the lock value.
repo_py - "$LOCK_FILE" <<'PYEOF'
import sys
from mech_int_vla.snapshots import load_model_input_lock, verify_lerobot_source
lock = load_model_input_lock(sys.argv[1])
print("lerobot source verified at", verify_lerobot_source(lock), lock.lerobot_python_sha256)
PYEOF

# ---------------------------------------------------------------------------
# 5. hf-libero config + pinned LIBERO assets
# ---------------------------------------------------------------------------
step "5. libero assets"
# hf-libero 0.1.4 (libero/libero/__init__.py) prompts on stdin the first time it
# is imported without ~/.libero/config.yaml, so write the default path dict
# ourselves.  Its env code resolves assets through get_assets_path(), which
# takes <site-packages>/libero/libero/assets when that directory exists and
# otherwise tries an UNPINNED Hub download; get_libero_path("assets") points to
# the same directory by default.  We fill that directory so no download happens.
LIBERO_CONFIG_DIR="${LIBERO_CONFIG_PATH:-$HOME/.libero}"
LIBERO_PKG_ROOT="$("$PY" -c 'import importlib.util as u; print(u.find_spec("libero").submodule_search_locations[0])')"
LIBERO_BENCH_ROOT="$LIBERO_PKG_ROOT/libero"
[[ -f "$LIBERO_BENCH_ROOT/__init__.py" ]] || die "hf-libero package root not found at $LIBERO_BENCH_ROOT"
if [[ ! -f "$LIBERO_CONFIG_DIR/config.yaml" ]]; then
  mkdir -p "$LIBERO_CONFIG_DIR"
  "$PY" - "$LIBERO_BENCH_ROOT" "$LIBERO_CONFIG_DIR/config.yaml" <<'PYEOF'
import os, sys, yaml
root, out = sys.argv[1], sys.argv[2]
# Mirrors libero.libero.get_default_path_dict(None) exactly.
cfg = {
    "benchmark_root": root,
    "bddl_files": os.path.join(root, "./bddl_files"),
    "init_states": os.path.join(root, "./init_files"),
    "datasets": os.path.join(root, "../datasets"),
    "assets": os.path.join(root, "./assets"),
}
with open(out, "w") as fh:
    yaml.dump(cfg, fh)
print("wrote", out, cfg)
PYEOF
fi
ASSET_DIR="$("$PY" -c 'from libero.libero import get_libero_path; import os; print(os.path.normpath(get_libero_path("assets")))' </dev/null)"
log "hf-libero asset dir: $ASSET_DIR"
[[ "$ASSET_DIR" == "$LIBERO_BENCH_ROOT/assets" ]] || die "unexpected asset dir $ASSET_DIR (get_assets_path() would look at $LIBERO_BENCH_ROOT/assets)"

if [[ -d "$ASSET_DIR" ]] && verify_asset_tree "$ASSET_DIR"; then
  log "asset dir already verified; skipping download"
else
  if [[ -d "$ASSET_STAGING" ]] && verify_asset_tree "$ASSET_STAGING"; then
    log "staging already verified"
  else
    # Full pinned snapshot of the dataset repo lerobot/libero-assets.  The earlier
    # run (log.md SETUP-013) staged a 60-file task-5 subset plus two textures; the
    # full pinned revision is a verified superset and the cleaner choice.
    log "downloading $ASSETS_REPO@$ASSETS_REVISION to $ASSET_STAGING"
    env -u HF_HUB_OFFLINE -u TRANSFORMERS_OFFLINE "$PY" - "$ASSETS_REPO" "$ASSETS_REVISION" "$ASSET_STAGING" <<'PYEOF'
import sys
from huggingface_hub import snapshot_download
repo, rev, dest = sys.argv[1:4]
path = snapshot_download(repo_id=repo, repo_type="dataset", revision=rev,
                         local_dir=dest, max_workers=8)
print("snapshot at", path)
PYEOF
    verify_asset_tree "$ASSET_STAGING"
  fi
  # Overlay-copy (not symlink) real files into the package asset dir, overwriting.
  mkdir -p "$ASSET_DIR"
  rsync -a --exclude '.cache' "$ASSET_STAGING/" "$ASSET_DIR/"
  verify_asset_tree "$ASSET_DIR"
  log "copied verified assets into $ASSET_DIR"
fi

# ---------------------------------------------------------------------------
# 6. Pinned HF snapshots (policy + base VLM) via the repo's own resolver
# ---------------------------------------------------------------------------
step "6. hf snapshots"
# runtime_cli `snapshots --download` calls resolve_snapshot_paths, which pulls
# the policy with POLICY_ALLOW_PATTERNS (config/model/processors/*.safetensors)
# and the base VLM with *.json/*.model/*.txt into the standard hub cache layout
# under --cache-dir, pins the resolved commit, and checks model.safetensors sha.
if repo_py -m mech_int_vla.runtime_cli snapshots --environment-lock "$LOCK_FILE" --cache-dir "$HF_CACHE" >/dev/null 2>&1; then
  log "snapshots already resolvable offline in $HF_CACHE"
else
  env -u HF_HUB_OFFLINE -u TRANSFORMERS_OFFLINE \
    PYTHONPATH="$CHECKOUT/src" "$PY" -m mech_int_vla.runtime_cli snapshots \
    --download --environment-lock "$LOCK_FILE" --cache-dir "$HF_CACHE"
fi
policy_model="$(find "$HF_CACHE/models--lerobot--smolvla_libero/snapshots/$(lock_value policy_revision)" -maxdepth 1 -name model.safetensors | head -n1)"
[[ -n "$policy_model" ]] || die "policy model.safetensors not found in $HF_CACHE"
actual_model_sha="$(sha256_of "$policy_model")"
[[ "$actual_model_sha" == "$POLICY_MODEL_SHA256" ]] || die "model.safetensors sha256 $actual_model_sha != $POLICY_MODEL_SHA256"
log "model.safetensors verified ($POLICY_MODEL_SHA256)"

# ---------------------------------------------------------------------------
# 7. Self-check (offline, collection environment exported only here)
# ---------------------------------------------------------------------------
step "7. self-check"
(
  export MUJOCO_GL=egl HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 CUDA_VISIBLE_DEVICES=0 \
         MUJOCO_EGL_DEVICE_ID=0 TOKENIZERS_PARALLELISM=false
  "$PY" -c 'import sys, platform; print("python", platform.python_version(), "prefix", sys.prefix)'
  log "environment-gpu.freeze sha256: $(sha256_of "$FREEZE_FILE")"
  nvidia-smi --query-gpu=index,name,driver_version,compute_cap,memory.total --format=csv,noheader
  # pip freeze diff: fail on any pinned distribution whose version differs or is
  # missing; report (do not fail on) extra distributions not in the freeze.
  "$PY" - "$FREEZE_FILE" <<'PYEOF'
import importlib.metadata as m, re, sys
from packaging.version import Version
pins, extras_ok = {}, {"pip"}
for line in open(sys.argv[1]):
    line = line.strip()
    if not line or line.startswith("#"):
        continue
    if line.startswith("-e "):
        pins["lerobot"] = "0.6.0"; continue
    if " @ file://" in line:
        name = line.split(" @ ")[0]
        pins[name] = {"torch": "2.11.0", "torchvision": "0.26.0+cu130"}[name]; continue
    name, _, version = line.partition("==")
    pins[name] = version
norm = lambda n: re.sub(r"[-_.]+", "-", n).lower()
installed = {norm(d.metadata["Name"]): d.version for d in m.distributions()}
same = lambda a, b: a is not None and Version(a) == Version(b)  # PEP 440-normalized compare
bad = [f"{n}: {installed.get(norm(n), 'MISSING')} != {v}" for n, v in pins.items() if not same(installed.get(norm(n)), v)]
extra = sorted(n for n in installed if n not in {norm(p) for p in pins} and n not in extras_ok)
print(f"pinned distributions: {len(pins)}; extras not in freeze (informational): {extra}")
if bad:
    print("\n".join(bad)); sys.exit("pip freeze differs from environment-gpu.freeze")
print("pip freeze matches environment-gpu.freeze on every pinned distribution")
PYEOF
  "$PY" -c 'import torch; print("torch", torch.__version__, "cuda", torch.version.cuda, torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0))'
  repo_py -m mech_int_vla.runtime_cli snapshots --environment-lock "$LOCK_FILE" --cache-dir "$HF_CACHE"
  repo_py -m mech_int_vla.runtime_cli load-policy --environment-lock "$LOCK_FILE" --cache-dir "$HF_CACHE" --device cuda
)
touch "$STAMP_DIR/bootstrap.ok"
step "bootstrap complete; run ops/locked_test_supervisor.py --plan-only next (see ops/locked_test_runbook.md)"
