import numpy as np
import os
import glob
from scipy.spatial import cKDTree

# Constants
nframe_start = 0
nframe_end = 160
nframe = nframe_end - nframe_start
reff = 10
x_white = 500
thres_coor = 3
outname = "tset_18Dec"
validation = False

# Output matrices
detection_white = np.zeros((x_white, nframe, reff), dtype=np.uint8)
detected_index = {}  # (frame, re) → synthetic index list

# Directories
sex_dir = "/g/data/jh2/jt4478/make_LC_copy/ngc628-c_white-R17v100/white/s_extraction"
white_dir = "/g/data/jh2/jt4478/make_LC_copy/ngc628-c_white-R17v100/white"
matched_out = os.path.join(white_dir, "matched_coords")
os.makedirs(matched_out, exist_ok=True)

for re in range(1, reff + 1):
    for f in range(nframe_start, nframe_end):

        # ---- SEXTRACTOR COORD FILE ----
        if validation:
            sex_pattern = (
                f"sex_ngc628-c_WFC3_UVISfF336W_frame{framenum}_vframe_{f}_"
                f"{outname}_validation_reff{re:.2f}.fits.coo"
            )
        else:
            sex_pattern = (
                f"sex_ngc628-c_WFC3_UVISfF336W_frame{f}_{outname}_reff{re:.2f}.fits.coo"
            )

        sex_files = glob.glob(os.path.join(sex_dir, sex_pattern))

        # ---- synthetic positions ----
        if validation:
            syn_pattern = (
                f"white_position_{framenum}_vframe_{f}_{outname}_validation_reff{re:.2f}.txt"
            )
        else:
            syn_pattern = f"white_position_{f}_{outname}_reff{re:.2f}.txt"

        syn_files = glob.glob(os.path.join(white_dir, syn_pattern))

        if not sex_files or not syn_files:
            continue

        sex_c = np.loadtxt(sex_files[0])
        syn_c = np.loadtxt(syn_files[0])

        # Extract positions
        x_v, y_v = sex_c[:, 0], sex_c[:, 1]
        x_s, y_s = syn_c[:, 0], syn_c[:, 1]

        # KDTree match
        tree = cKDTree(np.column_stack((x_v, y_v)))
        dist, idx = tree.query(np.column_stack((x_s, y_s)), distance_upper_bound=thres_coor)

        mask = np.isfinite(dist) & (dist < thres_coor)
        detection_white[:, f, re - 1] = mask.astype(np.uint8)

        # Save detected synthetic indices
        detected_index[(f, re)] = np.where(mask)[0]

# print("White detection done:", detection_white.shape)
# np.save("detection_white_tset_18Dec.npy", detection_white)
# np.save("DETECTED_INDEX_tset_18Dec.npy", detected_index)

import numpy as np
import os
import glob
from scipy.spatial import cKDTree

filters = ["f275w", "f336w", "f435w", "f555w", "f814w"]
nf = len(filters)

# Constants
nframe = 160
reff = 10
x_white = 500
validation = False
main_dir = "/g/data/jh2/jt4478/make_LC_copy/ngc628-c_white-R17v100"

# detected_index[(frame, reff)] = array of synthetic IDs detected in white
# detected_index = np.load("DETECTED_INDEX.npy", allow_pickle=True).item()

# Output matrix
detection_filters = np.zeros((x_white, nframe, reff), dtype=np.uint8)

# Absolute magnitude cut
D_MOD = 29.98
MV_cut = -6
V_LIMIT = D_MOD + MV_cut

for re in range(1, reff + 1):
    for f in range(nframe):

        # ---- white-detected synthetic IDs ----
        syn_ids = detected_index.get((f, re), np.array([], dtype=int))
        if len(syn_ids) == 0:
            continue

        # ---- load matched white coords (reference space!) ----
        matched_file = os.path.join(
            main_dir,
            "white/matched_coords",
            f"matched_frame{f}_tset_18Dec_reff{re:.2f}.txt"
        )
        if not os.path.exists(matched_file):
            continue

        matched_c = np.loadtxt(matched_file)[:, :2]

        # sanity check: matched coords must align with syn_ids
        if matched_c.shape[0] != len(syn_ids):
            raise RuntimeError(
                f"Mismatch: matched coords {matched_c.shape[0]} != syn_ids {len(syn_ids)} "
                f"(frame={f}, reff={re})"
            )

        # Track filter-level detection (only for white-detected ones)
        good = np.zeros((len(syn_ids), nf), dtype=np.uint8)

        for j, filt in enumerate(filters):

            # ---- locate filter coo file (only contains PASSING objects) ----
            if filt != "f555w":
                pattern = (
                    f"{filt}/photometry/"
                    f"coo_ngc628-c*{filt}_frame{f}_tset_18Dec_reff{re:.2f}.fits.coo"
                )
            else:
                pattern = (
                    f"{filt}/CI/"
                    f"ci_cut_*ngc628-c*{filt}_frame{f}_tset_18Dec_reff{re:.2f}.fits.coo"
                )

            files = glob.glob(os.path.join(main_dir, pattern))
            if not files:
                continue

            sex_c = np.loadtxt(files[0])
            if sex_c.ndim == 1:
                sex_c = sex_c[None, :]

            # ---- KDTree: does this white object have a valid photometry in this filter? ----
            tree = cKDTree(sex_c[:, :2])
            dist, idx = tree.query(matched_c, distance_upper_bound=3)

            detect_mask = np.isfinite(dist) & (dist < 3)

            # ---- photometric QC ----
            if detect_mask.any():
                mags = sex_c[idx[detect_mask], 2]
                magerr = sex_c[idx[detect_mask], 3]

                qc = magerr <= 0.3
                if filt == "f555w":
                    qc &= (mags < V_LIMIT)

                detect_mask[detect_mask] &= qc

            good[:, j] = detect_mask.astype(np.uint8)

        # ---- ≥4 filters requirement ----
        N_good = good.sum(axis=1)
        detection_filters[syn_ids, f, re - 1] = (N_good >= 4).astype(np.uint8)

# np.save("detection_filters_tset_18Dec.npy", detection_filters)
# print("✔ detection_filters shape =", detection_filters.shape)
