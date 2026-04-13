#!/usr/bin/env python3
"""
Bootstrap one LEGUS galaxy/pointing directory under a project root:
1) create <project_root>/<galaxy>/
2) download LEGUS FITS archives/files + catalogue readme
3) extract .tar.gz and decompress .fits.gz
4) create <galaxy>_white.fits using pure Python

Example
-------
python scripts/setup_legus_galaxy.py --galaxy ngc1313-e
"""

from __future__ import annotations

import argparse
import fnmatch
import gzip
import re
import shutil
import tarfile
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

from make_white_light import make_white_from_filters


def fetch_html(url: str) -> str:
    try:
        with urllib.request.urlopen(url) as r:
            return r.read().decode("utf-8", errors="ignore")
    except urllib.error.URLError as e:
        raise RuntimeError(f"Failed to open URL: {url}\n{e}") from e


def discover_data_urls(index_url: str) -> list[str]:
    html = fetch_html(index_url)
    links = re.findall(r'href=["\']([^"\']+)["\']', html, flags=re.IGNORECASE)
    out: list[str] = []
    for href in links:
        href_l = href.lower()
        if href_l.endswith((".tar.gz", ".tgz", ".fits.gz", ".fits")):
            out.append(urllib.parse.urljoin(index_url, href))
    # keep stable order and unique
    seen: set[str] = set()
    uniq = []
    for u in out:
        if u not in seen:
            uniq.append(u)
            seen.add(u)
    # Prefer *_drc.tar.gz when listed so we do not queue every per-filter HLSP FITS on the index page.
    drc_tar = [
        u
        for u in uniq
        if Path(urllib.parse.urlparse(u).path).name.lower().endswith("_drc.tar.gz")
    ]
    if drc_tar:
        return drc_tar
    return uniq


def resolve_template_urls(url_template: str, galaxy: str) -> list[str]:
    """
    Resolve a URL template into one or more concrete URLs.

    Supports wildcard templates that include '*' in the filename part by
    scraping the parent directory index and matching links with fnmatch.
    """
    url = url_template.format(galaxy=galaxy)
    parsed = urllib.parse.urlparse(url)
    name = Path(parsed.path).name
    if "*" not in name:
        return [url]

    base_path = str(Path(parsed.path).parent).rstrip("/") + "/"
    index_url = urllib.parse.urlunparse(
        (parsed.scheme, parsed.netloc, base_path, "", "", "")
    )
    html = fetch_html(index_url)
    links = re.findall(r'href=["\']([^"\']+)["\']', html, flags=re.IGNORECASE)
    out: list[str] = []
    for href in links:
        href_name = Path(urllib.parse.urlparse(href).path).name
        if fnmatch.fnmatch(href_name, name):
            out.append(urllib.parse.urljoin(index_url, href))
    # stable unique
    seen: set[str] = set()
    uniq: list[str] = []
    for u in out:
        if u not in seen:
            uniq.append(u)
            seen.add(u)
    return uniq


def download(url: str, dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        urllib.request.urlretrieve(url, str(dest))
        return dest
    except Exception as e:  # noqa: BLE001
        raise RuntimeError(f"Download failed: {url}\n-> {dest}\n{e}") from e


def extract_file(path: Path, target_dir: Path) -> None:
    if path.name.lower().endswith((".tar.gz", ".tgz")):
        with tarfile.open(path, "r:gz") as tf:
            tf.extractall(target_dir)
    elif path.name.lower().endswith(".fits.gz"):
        out = path.with_suffix("")  # drop .gz
        with gzip.open(path, "rb") as f_in, open(out, "wb") as f_out:
            shutil.copyfileobj(f_in, f_out)


def galaxy_has_hlsp_science_fits(gal_dir: Path) -> bool:
    """
    True if at least one LEGUS HLSP science FITS exists under gal_dir (recursive),
    excluding downloads/ (often partial archives) and *_white*.fits.
    """
    for p in gal_dir.rglob("*.fits"):
        if "downloads" in p.parts:
            continue
        n = p.name.lower()
        if "white" in n:
            continue
        if n.startswith("hlsp_legus") or (n.startswith("hlsp") and "legus" in n):
            return True
    return False


def has_local_catalog_support_files(gal_dir: Path, galaxy: str) -> bool:
    """
    True if required catalogue/config sidecar files already exist locally.
    Accepts both full galaxy id and base id (e.g. ngc628-c -> ngc628).
    """
    galaxy_base = galaxy.split("-")[0]
    suffixes = [galaxy, galaxy_base] if galaxy_base != galaxy else [galaxy]

    def any_match(glob_pat: str) -> bool:
        return any(gal_dir.glob(glob_pat))

    has_auto = any(any_match(f"automatic_catalog*_{s}.readme") for s in suffixes)
    has_apcorr = any(any_match(f"avg_aperture_correction*_{s}.txt") for s in suffixes)
    has_header = any(any_match(f"header_info*_{s}.txt") for s in suffixes)
    has_r2 = any(any_match(f"r2_wl_aa*{s}.config") for s in suffixes)
    return has_auto and has_apcorr and has_header and has_r2


def sync_key_catalog_files_to_root(galaxy: str, gal_dir: Path) -> None:
    """
    Ensure key per-galaxy metadata files are present in galaxy root, because
    pipeline code commonly looks there first.
    """
    key_names = [
        f"automatic_catalog_{galaxy}.readme",
        f"avg_aperture_correction_{galaxy}.txt",
        f"header_info_{galaxy}.txt",
        f"r2_wl_aa_{galaxy}.config",
    ]
    for name in key_names:
        # Already in root
        root_path = gal_dir / name
        if root_path.exists():
            continue
        # Search recursively in downloaded/extracted subdirs
        matches = list(gal_dir.rglob(name))
        matches = [m for m in matches if m.is_file() and m != root_path]
        if not matches:
            continue
        # Prefer shortest path (usually catalog_generation_files/name)
        src = sorted(matches, key=lambda p: len(str(p)))[0]
        shutil.copy2(src, root_path)
        print(f"[sync] {src} -> {root_path}")


def main() -> int:
    ap = argparse.ArgumentParser(description="Download LEGUS data and set up one galaxy directory.")
    ap.add_argument("--galaxy", required=True, help="Galaxy/pointing id, e.g. ngc1313-e")
    ap.add_argument(
        "--project-root",
        default="/g/data/jh2/jt4478/comp_pipeline_restructure",
        help="Project root. Galaxy data goes under <project-root>/<galaxy>/",
    )
    ap.add_argument(
        "--fits-index-url-template",
        default="https://archive.stsci.edu/hlsps/legus/{galaxy}/",
        help="Index page used to auto-discover .tar.gz/.fits(.gz) links (galaxy root, not .../images/).",
    )
    ap.add_argument(
        "--catalog-readme-url-template",
        default=(
            "https://archive.stsci.edu/hlsps/legus/{galaxy}/cluster_catalogs/deterministic/"
            "hlsp_legus*{galaxy}_multiband*padagb-mwext-avgapcor.readme"
        ),
        help="Catalog readme URL template (supports '*' wildcard in filename).",
    )
    ap.add_argument(
        "--extra-url-template",
        action="append",
        default=[],
        help=(
            "Additional file URL template(s) to download. "
            "Use {galaxy} placeholder, e.g. "
            "https://.../automatic_catalog_{galaxy}.readme"
        ),
    )
    ap.add_argument(
        "--fits-url",
        action="append",
        default=[],
        help="Optional direct FITS/tar URL(s). If omitted, links are auto-discovered from index page.",
    )
    ap.add_argument(
        "--keep-compressed",
        action="store_true",
        help="Keep downloaded .tar.gz/.fits.gz files after extraction/decompression.",
    )
    ap.add_argument(
        "--force-download",
        action="store_true",
        help="Download FITS/archives even if HLSP science FITS already exist under the galaxy dir.",
    )
    ap.add_argument(
        "--force-white",
        action="store_true",
        help="Regenerate <galaxy>_white.fits even if it already exists.",
    )
    args = ap.parse_args()

    galaxy = args.galaxy.strip().lower()
    project_root = Path(args.project_root).expanduser().resolve()
    gal_dir = (project_root / galaxy).resolve()
    downloads_dir = gal_dir / "downloads"
    gal_dir.mkdir(parents=True, exist_ok=True)
    downloads_dir.mkdir(parents=True, exist_ok=True)

    fits_urls = list(args.fits_url)
    skip_fits_blobs = (
        not args.force_download
        and not fits_urls
        and galaxy_has_hlsp_science_fits(gal_dir)
    )
    if skip_fits_blobs:
        print(
            "[skip] HLSP science FITS already present under "
            f"{gal_dir} (recursive, excl. downloads/); skipping FITS/tar downloads. "
            "Use --force-download to re-download."
        )
        fits_urls = []
    elif not fits_urls:
        index_url = args.fits_index_url_template.format(galaxy=galaxy)
        fits_urls = discover_data_urls(index_url)
        if not fits_urls:
            raise RuntimeError(
                f"No downloadable FITS/tar links found at {index_url}. "
                "Use --fits-url explicitly."
            )

    skip_meta_downloads = (
        not args.force_download
        and skip_fits_blobs
        and has_local_catalog_support_files(gal_dir, galaxy)
    )
    if skip_meta_downloads:
        print(
            "[skip] local catalogue/config files already present; "
            "skipping readme/metadata downloads and creating white only."
        )
        readme_urls = []
    else:
        readme_urls = resolve_template_urls(args.catalog_readme_url_template, galaxy)
        if not readme_urls:
            raise RuntimeError(
                "No catalog readme matched template: "
                f"{args.catalog_readme_url_template.format(galaxy=galaxy)}"
            )
    default_extra_templates = [
        (
            "https://archive.stsci.edu/hlsps/legus/{galaxy}/cluster_catalogs/deterministic/"
            "catalog_generation_files/automatic_catalog_*{galaxy}.readme"
        ),
        (
            "https://archive.stsci.edu/hlsps/legus/{galaxy}/cluster_catalogs/deterministic/"
            "catalog_generation_files/avg_aperture_correction_*{galaxy}.txt"
        ),
        (
            "https://archive.stsci.edu/hlsps/legus/{galaxy}/cluster_catalogs/deterministic/"
            "catalog_generation_files/header_info_*{galaxy}.txt"
        ),
        (
            "https://archive.stsci.edu/hlsps/legus/{galaxy}/cluster_catalogs/deterministic/"
            "catalog_generation_files/r2_wl_aa_*{galaxy}.config"
        ),
    ]
    extra_urls: list[str] = []
    if not skip_meta_downloads:
        for tpl in (default_extra_templates + list(args.extra_url_template)):
            resolved = resolve_template_urls(tpl, galaxy)
            if not resolved:
                print(f"[warn] no matches for template: {tpl.format(galaxy=galaxy)}")
                continue
            extra_urls.extend(resolved)

    downloaded: list[Path] = []
    for url in fits_urls + readme_urls + extra_urls:
        fname = Path(urllib.parse.urlparse(url).path).name
        if not fname:
            raise RuntimeError(f"Could not derive filename from URL: {url}")
        dst = downloads_dir / fname
        # Skip re-download if we already have this file in downloads or synced to root
        if url in extra_urls:
            key_in_root = gal_dir / fname
            if key_in_root.exists():
                print(f"[skip] already have {key_in_root.name}")
                continue
        if dst.exists() and dst.stat().st_size > 0:
            print(f"[skip] already have {dst}")
            downloaded.append(dst)
            continue
        print(f"[download] {url}")
        downloaded.append(download(url, dst))

    for p in downloaded:
        if p.name.lower().endswith((".tar.gz", ".tgz", ".fits.gz")):
            print(f"[extract] {p}")
            extract_file(p, gal_dir)
            if not args.keep_compressed:
                p.unlink(missing_ok=True)

    # Copy readme to galaxy root for easy discovery
    for p in downloads_dir.glob("*.readme"):
        shutil.copy2(p, gal_dir / p.name)

    # Keep key per-galaxy files in galaxy root for pipeline compatibility.
    sync_key_catalog_files_to_root(galaxy=galaxy, gal_dir=gal_dir)

    white_path = gal_dir / f"{galaxy}_white.fits"
    if white_path.exists() and not args.force_white:
        print(f"[skip] white already exists: {white_path} (use --force-white to regenerate)")
    else:
        print("[white] creating white image via pure Python")
        out_white = make_white_from_filters(galaxy=galaxy, gal_dir=gal_dir)
        print(f"[white] wrote {out_white}")

    print(f"[done] LEGUS galaxy setup complete: {gal_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

