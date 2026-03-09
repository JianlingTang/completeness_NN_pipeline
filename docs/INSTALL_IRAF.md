# Installing IRAF locally (for 5-filter photometry)

PyRAF is installed in the project `.venv` via `pip install pyraf`. To run the full pipeline with photometry, you also need **IRAF** installed and the environment variable set.

## Apple Silicon (M1/M2/M3) one-step install

If the installer package is already in the project, run on your machine:

```bash
# 1. Remove quarantine (otherwise macOS may block the install)
xattr -c .deps/iraf_download/iraf-2.18.1-1-arm64.pkg

# 2. Open the installer (will prompt for login password)
open .deps/iraf_download/iraf-2.18.1-1-arm64.pkg
```

Follow the installer; enter your password when asked. By default IRAF is installed to `/usr/local/lib/iraf/`.

## Running the full pipeline (with photometry to detection_*.npy)

After IRAF is installed, from the project root run:

```bash
chmod +x scripts/run_full_with_iraf.sh
./scripts/run_full_with_iraf.sh
```

That script sets `IRAF=/usr/local/lib/iraf` and runs:

```bash
python scripts/run_small_test.py --input_coords ngc628-c/white/input_coords_500.txt --run_photometry
```

This runs: Phase A injection, 5-filter frame setup, Phase B (detection + matching + 5-filter photometry + CI cut), then writes `detection_*.npy` and completeness diagnostics.

## Running without IRAF

You can run the same command without IRAF installed; photometry will be skipped and you will only get white-match labels and diagnostics:

```bash
.venv/bin/python scripts/run_small_test.py --input_coords ngc628-c/white/input_coords_500.txt --run_photometry
```
