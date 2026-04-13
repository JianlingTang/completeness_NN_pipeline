# HPC 部署说明 / Deploy to project root

解压此 zip 到你的**项目根目录**（例如 `/g/data/jh2/jt4478/make_LC_copy`）。  
**生成结果**（white/、physprop/、tmp_pipeline_test/ 等）始终写在项目根下。  
**输入数据**（SLUG 库、PSF、白光/5-filter FITS）通过环境变量指定，可不放在项目根。

## 环境变量 (全部可选，不设则用项目根下默认路径)

| 变量 | 含义 | 默认 |
|------|------|------|
| **COMP_SLUG_LIB_DIR** | SLUG 库目录 | ROOT/SLUG_library |
| **COMP_PSF_PATH** | PSF 目录 | ROOT/PSF_files |
| **COMP_FITS_PATH** | 含 galaxy 的目录（其下应有 ngc628-c/，内有 5-filter *drc.fits 等） | ROOT |
| **COMP_SCIFRAME** | 白光科学图 ngc628-c_white.fits 的完整路径 | COMP_FITS_PATH/ngc628-c/ngc628-c_white.fits |
| **COMP_BAO_PATH** | 含 BAOlab 可执行文件 bl 的目录 | ROOT/.deps/local/bin |
| **COMP_OUTPUT_LIB_DIR** | 额外 SLUG 输出库（非 flat 模型） | ROOT/SLUG_library |
| **COMP_TEMP_BASE_DIR** | 临时目录 | ROOT/tmp_pipeline_test |
| **IRAF** | IRAF 安装目录（做 photometry 时必设） | — |

PBS 示例（输入都在其他路径，生成在项目根）：

```bash
export COMP_SLUG_LIB_DIR="/g/data/jh2/jt4478/SLUG_library"
export COMP_PSF_PATH="/g/data/jh2/jt4478/PSF_files"
export COMP_FITS_PATH="/g/data/jh2/jt4478/ngc628_data"
export COMP_SCIFRAME="/g/data/jh2/jt4478/ngc628_data/ngc628-c/ngc628-c_white.fits"
export COMP_BAO_PATH="/g/data/jh2/jt4478/.deps/local/bin"
export IRAF="/path/to/iraf"
cd /g/data/jh2/jt4478/make_LC_copy
python scripts/run_pipeline.py --cleanup --nframe 3 --reff_list "1,3,5,7,9" --run_photometry --parallel
```

## 解压后需要准备的输入

- **SLUG_library**：由 COMP_SLUG_LIB_DIR 指向；或放在项目根下 SLUG_library/。
- **PSF_files**：由 COMP_PSF_PATH 指向；或放在项目根下 PSF_files/。
- **ngc628-c**：由 COMP_FITS_PATH 指向的目录下要有 ngc628-c/，内含 ngc628-c_white.fits 和 5 个 filter 的 *drc.fits；或设 COMP_SCIFRAME 指向白光 FITS。
- **BAOlab**：由 COMP_BAO_PATH 指向含 bl 的目录。
- **IRAF**：做 photometry 时必设 IRAF= 安装路径。
- 项目根下需有 **galaxy_filter_dict.npy**、**output.param**、**default.nnw**（zip 已含）；ngc628-c 的 **r2_wl_aa_ngc628-c.config** 等可在 COMP_FITS_PATH/ngc628-c/ 或项目根 ngc628-c/。

## 环境

- Python 3.10+
- `pip install -e .` 或 `pip install -r requirements.txt`
- photometry: `pip install pyraf`，并设置 IRAF
- SExtractor 在 PATH 或 module

详见 docs/RUNNING_ON_HPC.md。
