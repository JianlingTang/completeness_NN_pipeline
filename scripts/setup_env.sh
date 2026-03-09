#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────
#  Unified environment setup for cluster_pipeline
#
#  Usage:
#    bash scripts/setup_env.sh          # full install (recommended)
#    bash scripts/setup_env.sh --quick  # Python-only, skip ext tools
#
#  What it does:
#    1. Creates (or reuses) a Python venv + pip install
#    2. Installs SExtractor (brew / apt / from source)
#    3. Installs BAOlab (from GitHub source)
#    4. Prints a summary
#
#  Note: slugpy is NOT required — SLUG FITS libraries are read by
#  the built-in cluster_pipeline.data.slug_reader (pure Python).
# ─────────────────────────────────────────────────────────────────
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
VENV_DIR="${PROJECT_ROOT}/.venv"
DEPS_DIR="${PROJECT_ROOT}/.deps"
QUICK=false

for arg in "$@"; do
  case "$arg" in
    --quick) QUICK=true ;;
  esac
done

# ── helpers ──────────────────────────────────────────────────────
green()  { printf '\033[0;32m%s\033[0m\n' "$*"; }
yellow() { printf '\033[0;33m%s\033[0m\n' "$*"; }
red()    { printf '\033[0;31m%s\033[0m\n' "$*"; }
info()   { printf '  → %s\n' "$*"; }

WARNINGS=()
warn_later() { WARNINGS+=("$1"); }

OS="$(uname -s)"

# ─────────────────────────────────────────────────────────────────
# Step 1: Python venv + pip
# ─────────────────────────────────────────────────────────────────
echo ""
green "═══  Step 1/3: Python virtual environment + packages  ═══"
if [ -d "$VENV_DIR" ]; then
  info "Reusing existing venv at ${VENV_DIR}"
else
  info "Creating venv at ${VENV_DIR} ..."
  python3 -m venv "$VENV_DIR"
fi
# shellcheck disable=SC1091
source "${VENV_DIR}/bin/activate"
info "Python: $(python3 --version) — $(which python3)"

pip install --upgrade pip -q
pip install -r "${PROJECT_ROOT}/requirements.txt" -q
info "All Python packages installed."

if python3 -c "from cluster_pipeline.data.slug_reader import read_cluster" 2>/dev/null; then
  info "Built-in SLUG reader: OK (no slugpy needed)"
else
  yellow "slug_reader import failed — make sure PYTHONPATH includes project root."
  warn_later "slug_reader import failed — run: export PYTHONPATH=${PROJECT_ROOT}"
fi

if $QUICK; then
  echo ""
  yellow "──  --quick mode: skipping SExtractor + BAOlab  ──"
  echo ""
  green "Done (quick). Activate:  source .venv/bin/activate"
  exit 0
fi

mkdir -p "$DEPS_DIR"

# ─────────────────────────────────────────────────────────────────
# Step 2: SExtractor
# ─────────────────────────────────────────────────────────────────
echo ""
green "═══  Step 2/3: SExtractor  ═══"

_sex_found() {
  command -v sex &>/dev/null || command -v source-extractor &>/dev/null || command -v sextractor &>/dev/null
}

if _sex_found; then
  info "SExtractor already installed: $(command -v sex || command -v source-extractor || command -v sextractor)"
else
  info "SExtractor not found — attempting to install..."

  installed_sex=false

  # ── try Homebrew (macOS) ──────────────────────────────────
  if [ "$OS" = "Darwin" ] && command -v brew &>/dev/null; then
    info "Installing via Homebrew..."
    if brew install sextractor 2>/dev/null; then
      installed_sex=true
    fi
  fi

  # ── try apt (Debian/Ubuntu) ──────────────────────────────
  if ! $installed_sex && command -v apt-get &>/dev/null; then
    info "Installing via apt (may ask for sudo password)..."
    if sudo apt-get install -y sextractor 2>/dev/null; then
      installed_sex=true
    elif sudo apt-get install -y source-extractor 2>/dev/null; then
      installed_sex=true
    fi
  fi

  # ── try building from source ─────────────────────────────
  if ! $installed_sex; then
    SEX_DIR="${DEPS_DIR}/sextractor"
    SEX_TAG="2.28.2"
    info "Package manager install failed — building from source (v${SEX_TAG})..."

    if [ ! -d "$SEX_DIR" ]; then
      git clone --depth 1 --branch "${SEX_TAG}" \
        https://github.com/astromatic/sextractor.git "$SEX_DIR" 2>/dev/null || \
      git clone --depth 1 \
        https://github.com/astromatic/sextractor.git "$SEX_DIR"
    fi

    (
      cd "$SEX_DIR"
      sh autogen.sh 2>/dev/null || true
      ./configure --prefix="${DEPS_DIR}/local" --disable-model-fitting 2>&1 | tail -3
      make -j"$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 2)" 2>&1 | tail -3
      make install 2>&1 | tail -1
    ) && installed_sex=true
  fi

  # ── verify ───────────────────────────────────────────────
  if $installed_sex || _sex_found; then
    # If we built locally, add to PATH for this session
    if [ -x "${DEPS_DIR}/local/bin/sex" ]; then
      export PATH="${DEPS_DIR}/local/bin:$PATH"
      info "SExtractor installed to ${DEPS_DIR}/local/bin/sex"
      info "Add to PATH permanently:  export PATH=\"${DEPS_DIR}/local/bin:\$PATH\""
    else
      info "SExtractor installed: $(command -v sex || command -v source-extractor || command -v sextractor)"
    fi
  else
    red "SExtractor installation failed."
    echo ""
    echo "  Manual install options:"
    echo "    macOS:   brew install sextractor"
    echo "    Ubuntu:  sudo apt-get install sextractor"
    echo "    conda:   conda install -c conda-forge astromatic-source-extractor"
    echo ""
    warn_later "SExtractor installation failed — install manually (see above)."
  fi
fi

# Symlink hint: some systems install as 'source-extractor' but pipeline calls 'sex'
if ! command -v sex &>/dev/null && command -v source-extractor &>/dev/null; then
  info "Hint: pipeline calls 'sex' but your system has 'source-extractor'."
  info "  Fix:  sudo ln -s \$(which source-extractor) /usr/local/bin/sex"
fi

# ─────────────────────────────────────────────────────────────────
# Step 3: BAOlab
# ─────────────────────────────────────────────────────────────────
echo ""
green "═══  Step 3/3: BAOlab  ═══"

BAOLAB_REPO="https://github.com/soerenslarsen/baolab.git"
BAOLAB_DIR="${DEPS_DIR}/baolab"
BAOLAB_TAR="baolab-0.94.4a.tar.gz"

_bl_found() {
  command -v bl &>/dev/null
}

_has_x11() {
  [ -f "/opt/X11/include/X11/Xlib.h" ] || \
  [ -f "/usr/X11/include/X11/Xlib.h" ] || \
  [ -f "/usr/include/X11/Xlib.h" ]
}

_create_x11_stubs() {
  local stub_dir="$1/x11stub"
  mkdir -p "$stub_dir/X11/Xmu"
  python3 "${SCRIPT_DIR}/generate_x11_stubs.py" "$stub_dir"
}

if _bl_found; then
  info "BAOlab already installed: $(which bl)"
else
  info "BAOlab (bl) not found — downloading and building from source..."

  if [ ! -d "$BAOLAB_DIR" ]; then
    info "Cloning BAOlab repo..."
    git clone --depth 1 "$BAOLAB_REPO" "$BAOLAB_DIR" 2>/dev/null || {
      red "Failed to clone BAOlab repo."
      warn_later "BAOlab clone failed — install manually from ${BAOLAB_REPO}"
    }
  fi

  if [ -d "$BAOLAB_DIR" ]; then
    (
      cd "$BAOLAB_DIR"

      if [ -f "$BAOLAB_TAR" ]; then
        info "Extracting ${BAOLAB_TAR}..."
        tar xzf "$BAOLAB_TAR"
        src_dir="${BAOLAB_TAR%.tar.gz}"

        if [ -d "$src_dir" ]; then
          cd "$src_dir"

          X11_INC="" X11_LIB="" X11_LINK="-lX11 -lXmu" USE_STUBS=false

          if [ "$OS" = "Darwin" ] && [ -d "/opt/X11/include" ]; then
            X11_INC="-I/opt/X11/include"; X11_LIB="-L/opt/X11/lib"
          elif [ "$OS" = "Darwin" ] && [ -d "/usr/X11/include" ]; then
            X11_INC="-I/usr/X11/include"; X11_LIB="-L/usr/X11/lib"
          elif [ -d "/usr/include/X11" ]; then
            X11_INC=""; X11_LIB=""
          else
            info "X11 not found — building in headless mode (batch commands work, no GUI)"
            USE_STUBS=true
            _create_x11_stubs "$(pwd)"
            X11_INC="-I./x11stub"; X11_LIB=""; X11_LINK=""
          fi

          CCOPTS="${X11_INC} -DUSE_FFT2D_PTHREADS -O"
          CCOPTS="${CCOPTS} -Wno-implicit-function-declaration -Wno-incompatible-pointer-types"
          LDOPTS="${X11_LIB} -lpthread"

          info "Building BAOlab..."
          make clean 2>/dev/null || true
          make CC="${CC:-cc}" OPTIONS="${CCOPTS}" LFLAGS="${LDOPTS}" 2>&1 | tail -5 || true

          if [ ! -f "bl" ] && $USE_STUBS; then
            info "Makefile link failed (expected with stubs) — relinking without X11 libs..."
            cc -O -c x11stub.c -o x11stub.o
            cc ${CCOPTS} -DARCH="\"$(uname -s)\"" -o bl \
              fitsutil.o utils.o baostr.o parse.o blhelp.o \
              baolab1.o baolab2.o baolab3.o baolab4.o baolab5.o \
              baolab6.o baolab7.o \
              wutil.o solveq.o baolab.o mksynth.o \
              mkpsf.o syntpsf.o neldmead.o \
              convol.o ishape.o lsqlin.o \
              mark.o clean.o fftsg.o fftsg2d.o alloc.o \
              cosmic.o polyfit.o xfit.o xydist.o byteorder.o \
              kfit2d.o k66.o x11stub.o \
              -lm ${LDOPTS} 2>&1
          fi

          if [ -f "bl" ]; then
            mkdir -p "${DEPS_DIR}/local/bin"
            cp bl "${DEPS_DIR}/local/bin/bl"
            chmod +x "${DEPS_DIR}/local/bin/bl"
            cp -f baolab.hlp baolab.par "${DEPS_DIR}/local/bin/" 2>/dev/null || true
            export PATH="${DEPS_DIR}/local/bin:$PATH"
            export BLPATH="${DEPS_DIR}/baolab/${src_dir}"
            info "BAOlab installed to ${DEPS_DIR}/local/bin/bl"
            $USE_STUBS && info "(headless build — no GUI, batch mode fully functional)"
            info "Add to shell config:"
            info "  export PATH=\"${DEPS_DIR}/local/bin:\$PATH\""
            info "  export BLPATH=\"${DEPS_DIR}/baolab/${src_dir}\""
          else
            red "BAOlab build produced no 'bl' binary."
            warn_later "BAOlab build failed — see errors above."
          fi
        else
          red "Could not find extracted directory: ${src_dir}"
          warn_later "BAOlab extraction failed."
        fi
      else
        red "Tarball ${BAOLAB_TAR} not found in repo."
        warn_later "BAOlab tarball not found — check ${BAOLAB_DIR}"
      fi
    )
  fi

  if ! _bl_found && ! [ -x "${DEPS_DIR}/local/bin/bl" ]; then
    echo ""
    echo "  Manual install:"
    echo "    git clone ${BAOLAB_REPO}"
    echo "    cd baolab && tar xzf ${BAOLAB_TAR}"
    echo "    cd ${BAOLAB_TAR%.tar.gz} && make"
    echo "    cp bl /usr/local/bin/"
    echo ""
    warn_later "BAOlab not installed — see manual steps above."
  fi
fi

# ─────────────────────────────────────────────────────────────────
# Summary
# ─────────────────────────────────────────────────────────────────
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

if [ ${#WARNINGS[@]} -eq 0 ]; then
  green "All dependencies installed successfully!"
else
  yellow "Setup finished with warnings:"
  for w in "${WARNINGS[@]}"; do
    echo "  ⚠  $w"
  done
fi

echo ""
echo "  Activate env:  source .venv/bin/activate"

# Show PATH hint if we installed local binaries
if [ -d "${DEPS_DIR}/local/bin" ]; then
  echo "  Local tools:   export PATH=\"${DEPS_DIR}/local/bin:\$PATH\""
fi

echo ""
echo "  Quick check:"
echo "    python -c 'from cluster_pipeline.data.slug_reader import read_cluster; print(\"SLUG reader OK\")"
command -v sex &>/dev/null && echo "    sex --version   # $(sex --version 2>&1 | head -1)" || true
[ -x "${DEPS_DIR}/local/bin/bl" ] && echo "    bl              # ${DEPS_DIR}/local/bin/bl" || true
echo ""
echo "  Run tests:     make test"
echo "  Run pipeline:  python -m cluster_pipeline.pipeline.pipeline_runner ..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
