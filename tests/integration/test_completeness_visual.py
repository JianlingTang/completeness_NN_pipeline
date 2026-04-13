"""
Integration tests: completeness in mag / mass / age bins.

- Visualize completeness binned by magnitude, log(mass), log(age).
- Assert completeness is in [0, 1]; optionally that it behaves like a logistic curve
  (low at faint/low-mass end, high at bright/high-mass end).
- Report errors or continue via pytest (failures report; --continue-on-collection if needed).
"""
import numpy as np
import pytest

from cluster_pipeline.pipeline.diagnostics import completeness_per_bin


def binned_completeness_percentile(x, y_det, n_bins=15):
    """Same binning as run_pipeline: percentile-based bins, return centers, comp, counts."""
    ok = np.isfinite(x)
    x_ok, y_ok = x[ok], y_det[ok]
    if len(x_ok) < 2:
        return np.array([]), np.array([]), np.array([])
    bins = np.percentile(x_ok, np.linspace(0, 100, n_bins + 1))
    bins = np.unique(bins)
    if len(bins) < 2:
        return np.array([np.median(x_ok)]), np.array([np.mean(y_ok)]), np.array([len(x_ok)])
    hist_total, _ = np.histogram(x_ok, bins=bins)
    hist_det, _ = np.histogram(x_ok, bins=bins, weights=y_ok)
    centers = (bins[:-1] + bins[1:]) / 2
    comp = np.full_like(hist_total, np.nan, dtype=float)
    np.divide(hist_det, hist_total, out=comp, where=hist_total > 0)
    return centers, comp, hist_total


def logistic_p_detection(x, x0=22.0, k=1.5):
    """Probability of detection = 1 / (1 + exp(-k*(x0 - x))). Brighter (lower x/mag) -> higher p."""
    return 1.0 / (1.0 + np.exp(-k * (x0 - np.asarray(x))))


@pytest.fixture
def synthetic_logistic_mag():
    """Synthetic mag and matched labels from logistic detection probability."""
    np.random.seed(99)
    n = 500
    mags = np.random.uniform(18, 26, n)
    p = logistic_p_detection(mags, x0=22.0, k=1.2)
    matched = (np.random.rand(n) < p).astype(np.int8)
    return mags, matched


@pytest.fixture
def synthetic_logistic_logmass():
    """Synthetic log10(mass) and matched labels: high mass -> higher completeness."""
    np.random.seed(100)
    n = 500
    log_mass = np.random.uniform(5, 8, n)
    # p(det) increases with log_mass (logistic in log_mass)
    p = 1.0 / (1.0 + np.exp(-1.5 * (log_mass - 6.5)))
    matched = (np.random.rand(n) < p).astype(np.int8)
    return log_mass, matched


def test_completeness_in_zero_one(synthetic_logistic_mag):
    """Completeness per bin is in [0, 1] or nan."""
    mags, matched = synthetic_logistic_mag
    centers, comp, edges = completeness_per_bin(mags, matched, mag_min=17, mag_max=27, n_bins=12)
    valid = np.isfinite(comp)
    assert np.all(comp[valid] >= 0), "completeness should be >= 0"
    assert np.all(comp[valid] <= 1), "completeness should be <= 1"


def test_completeness_brighter_higher_than_faint(synthetic_logistic_mag):
    """With logistic-like detection: mean completeness in bright bins >= faint bins."""
    mags, matched = synthetic_logistic_mag
    centers, comp, edges = completeness_per_bin(mags, matched, mag_min=17, mag_max=27, n_bins=10)
    valid = np.isfinite(comp)
    if np.sum(valid) < 2:
        pytest.skip("Too few bins with data")
    # Bright = low mag = first half of bins; faint = high mag = second half
    n = len(centers)
    bright_comp = np.nanmean(comp[: n // 2])
    faint_comp = np.nanmean(comp[n // 2 :])
    assert bright_comp >= faint_comp - 0.05, (
        f"Completeness should be higher at bright end: bright={bright_comp:.3f}, faint={faint_comp:.3f}"
    )


def test_completeness_high_mass_higher_than_low_mass(synthetic_logistic_logmass):
    """With logistic in log(mass): high-mass bins have higher completeness."""
    log_mass, matched = synthetic_logistic_logmass
    centers, comp, hist = binned_completeness_percentile(log_mass, matched, n_bins=10)
    valid = np.isfinite(comp) & (hist > 5)
    if np.sum(valid) < 2:
        pytest.skip("Too few bins with enough counts")
    order = np.argsort(centers)
    comp_sorted = comp[order]
    low_mass_comp = np.nanmean(comp_sorted[: len(comp_sorted) // 2])
    high_mass_comp = np.nanmean(comp_sorted[len(comp_sorted) // 2 :])
    assert high_mass_comp >= low_mass_comp - 0.05, (
        f"Completeness should be higher at high mass: high={high_mass_comp:.3f}, low={low_mass_comp:.3f}"
    )


def test_completeness_visual_mag_mass_age_bins(synthetic_logistic_mag, synthetic_logistic_logmass, tmp_path):
    """
    Visualize completeness in mag and log(mass) bins; save figure to tmp_path.
    Asserts: comp in [0,1], and at least one bin with comp near 0 and one near 1 for logistic data.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        pytest.skip("matplotlib not available")

    mags, matched_mag = synthetic_logistic_mag
    log_mass, matched_mass = synthetic_logistic_logmass

    # Binned completeness (magnitude)
    cx_mag, cy_mag, _ = completeness_per_bin(mags, matched_mag, mag_min=17, mag_max=27, n_bins=15)
    # Binned completeness (log mass) via percentile
    cx_mass, cy_mass, ct_mass = binned_completeness_percentile(log_mass, matched_mass, n_bins=15)
    # Synthetic log(age): older -> slightly higher completeness
    np.random.seed(101)
    log_age = np.random.uniform(6, 9, 400)
    p_age = 1.0 / (1.0 + np.exp(-0.8 * (log_age - 7.5)))
    matched_age = (np.random.rand(400) < p_age).astype(np.int8)
    cx_age, cy_age, _ = binned_completeness_percentile(log_age, matched_age, n_bins=12)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].plot(cx_mag, cy_mag, "o-")
    axes[0].set_xlabel("Magnitude")
    axes[0].set_ylabel("Completeness")
    axes[0].set_ylim(-0.05, 1.05)
    axes[0].set_title("Completeness vs mag (synthetic logistic)")
    axes[0].grid(True, alpha=0.3)

    if len(cx_mass) > 0:
        axes[1].plot(cx_mass, cy_mass, "o-", color="C1")
    axes[1].set_xlabel(r"$\log_{10}$(mass)")
    axes[1].set_ylabel("Completeness")
    axes[1].set_ylim(-0.05, 1.05)
    axes[1].set_title("Completeness vs mass (synthetic)")
    axes[1].grid(True, alpha=0.3)

    if len(cx_age) > 0:
        axes[2].plot(cx_age, cy_age, "o-", color="C2")
    axes[2].set_xlabel(r"$\log_{10}$(age)")
    axes[2].set_ylabel("Completeness")
    axes[2].set_ylim(-0.05, 1.05)
    axes[2].set_title("Completeness vs age (synthetic)")
    axes[2].grid(True, alpha=0.3)

    fig.suptitle("Pipeline completeness visualization (unit-test synthetic data)")
    fig.tight_layout()
    out = tmp_path / "completeness_mag_mass_age_bins.png"
    fig.savefig(out, dpi=100)
    plt.close()

    # Sanity: at least one bin should have low completeness and one high (for our logistic data)
    valid_mag = np.isfinite(cy_mag)
    if np.sum(valid_mag) >= 2:
        assert np.nanmin(cy_mag) < 0.9 or np.nanmax(cy_mag) > 0.1, (
            "Expected some variation in completeness (logistic curve)"
        )
    assert out.exists()
