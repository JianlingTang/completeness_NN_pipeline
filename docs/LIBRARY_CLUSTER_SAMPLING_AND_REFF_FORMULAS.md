# Library cluster sampling and r_eff(M) prescriptions

This document gives the exact equations used to sample library clusters and the formulas for the **mean** and **variance** of the effective radius \(r_{\rm eff}\) as a function of mass \(M\) for the two main prescriptions used in the pipeline.

---

## 1. How library clusters are sampled

- **Library:** SLUG cluster libraries are loaded (`flat_in_logm_cluster_phot.fits`, and optionally other `*_cluster_phot.fits` depending on `mrmodel`). Each library entry has `actual_mass`, age, \(A_V\), and photometry.
- **Mass (and other properties):** Masses come directly from the library (`actual_mass`). They are **not** drawn from a power law; the library is pre-computed (e.g. flat in \(\log M\) for `flat_in_logm`).
- **Radius:** Assigned by one of two prescriptions below. The pipeline also uses a **fixed grid** of effective radii `reff_list = [1, 2, …, 10]` pc; for each run/frame one value from this list is used (see below).

### 1.1 Flat prescription (as used in the pipeline)

- For each **(frame_id, reff)** the code takes a **contiguous block** of `ncl` clusters from the main library pool.
- Index range for frame `i_frame` and radius index `ridx` (for `eradius = reff_list[ridx]`):
  - `start = (ridx * nframe + i_frame) * ncl`
  - `end = start + ncl`
- **Radius:** \(r_{\rm eff}\) is **not** a function of \(M\). For that run it is the single value **eradius** (one of 1, 2, …, 10 pc). So for the flat pipeline prescription:
  - **Mean:** \(\mathbb{E}[r_{\rm eff} \mid M] = r_{\rm eff}^{\rm fixed} = \texttt{eradius}\) (e.g. 3 pc if that run uses 3 pc).
  - **Variance:** \(\mathrm{Var}(r_{\rm eff} \mid M) = 0\) (deterministic).

### 1.2 Krumholz19 prescription

- Radii are generated from mass using a **log-normal** relation (see below). For each library mass, `sample_k19_radii(mass, n_draw=10)` draws 10 radii; then radii are **binned** into bins centered at 1, 2, …, 10 pc with half-width 0.5 (i.e. \(r \in (r-0.5, r+0.5)\)). Clusters are assigned to exactly one \(r_{\rm eff}\) bin; each bin gets a fixed number of clusters (`N_PER_REFF`), with **global uniqueness** (each parent cluster used only once across all bins).

---

## 2. r_eff(M): exact formulas for the two prescriptions

Below, \(M\) is cluster mass in \(M_\odot\), and \(r_{\rm eff}\) is in pc. All logarithms are base 10.

---

### Prescription 1: Flat (mass–radius formula in code)

Used in `mass_to_radius(..., model="flat")` (e.g. for scripts that call this function). **Not** used to assign radius in the main flat pipeline (there, \(r_{\rm eff}\) is fixed per run as above).

- **Model:** \(\log_{10}(r_{\rm eff})\) is drawn **independently of \(M\)** from a uniform distribution:
  \[
  \log_{10}(r_{\rm eff}) \sim \mathrm{Uniform}(0, 1).
  \]
  So \(r_{\rm eff}\) is uniform on \([1, 10]\) pc.

- **Mean (in log space):**
  \[
  \mathbb{E}[\log_{10}(r_{\rm eff})] = \frac{1}{2}.
  \]

- **Variance (in log space):**
  \[
  \mathrm{Var}(\log_{10}(r_{\rm eff})) = \frac{1}{12}.
  \]

- **Mean (linear space):**
  \[
  \mathbb{E}[r_{\rm eff}] = \frac{1+10}{2} = 5.5\ \mathrm{pc}.
  \]

- **Variance (linear space):**
  \[
  \mathrm{Var}(r_{\rm eff}) = \frac{(10-1)^2}{12} = \frac{81}{12} = 6.75\ \mathrm{pc}^2.
  \]

So for the **flat (code) prescription**: mean and variance of \(r_{\rm eff}\) do **not** depend on \(M\).

---

### Prescription 2: Krumholz19

Used in `sample_k19_radii()` and in `mass_to_radius(..., model="Krumholz19")`. The pipeline’s Krumholz19 sampling uses `sample_k19_radii`; the formulas below match that function.

- **Power-law (mean in log space) + Gaussian scatter:**
  \[
  \log_{10}(r_{\rm eff}) = \mu(M) + \varepsilon,\qquad \varepsilon \sim \mathcal{N}(0, \sigma^2).
  \]

- **Coefficients (from code):**
  - Slope (power-law exponent): \(\alpha = 0.1415\)  
    So \(\mu(M) = 0.1415\,\log_{10}(M)\) (no additive constant in log space).
  - Scatter (standard deviation in log space): \(\sigma = 0.21\).

  So:
  \[
  \boxed{
  \log_{10}(r_{\rm eff}) = 0.1415\,\log_{10}(M) + \varepsilon,\qquad \varepsilon \sim \mathcal{N}(0,\,0.21^2).
  }
  \]

  Equivalently, the **power law** for the mean in linear space is:
  \[
  \mathbb{E}[r_{\rm eff} \mid M]\ \propto\ M^{0.1415}.
  \]
  (The exact prefactor in linear space is \(10^{\mathbb{E}[\varepsilon]} = 1\) if we define the “mean” in log space as above; see below for the exact linear mean.)

- **Distribution:** The conditional distribution of \(\log_{10}(r_{\rm eff})\) given \(M\) is **Gaussian** with mean \(\mu(M)\) and variance \(\sigma^2\).

- **Mean of \(\log_{10}(r_{\rm eff})\) given \(M\):**
  \[
  \boxed{
  \mathbb{E}[\log_{10}(r_{\rm eff}) \mid M] = 0.1415\,\log_{10}(M).
  }
  \]

- **Variance of \(\log_{10}(r_{\rm eff})\) given \(M\):**
  \[
  \boxed{
  \mathrm{Var}(\log_{10}(r_{\rm eff}) \mid M) = \sigma^2 = (0.21)^2 = 0.0441.
  }
  \]
  So the variance is **constant** (does not depend on \(M\)).

- **Mean of \(r_{\rm eff}\) given \(M\) (linear space):**  
  Since \(\log_{10}(r_{\rm eff}) \sim \mathcal{N}(\mu(M), \sigma^2)\), \(r_{\rm eff} = 10^{\log_{10}(r_{\rm eff})}\) is log-normal. So:
  \[
  \mathbb{E}[r_{\rm eff} \mid M] = 10^{\mu(M) + \frac{1}{2}\sigma^2 (\ln 10)^2}
  = 10^{0.1415\,\log_{10}(M) + \frac{1}{2}(0.21)^2(\ln 10)^2}.
  \]
  Let \(c = \frac{1}{2}(0.21)^2(\ln 10)^2 \approx 0.560\). Then:
  \[
  \boxed{
  \mathbb{E}[r_{\rm eff} \mid M] = 10^c\cdot M^{0.1415} \approx 3.63\cdot M^{0.1415}\ \mathrm{pc}.
  }
  \]

- **Variance of \(r_{\rm eff}\) given \(M\) (linear space):**  
  For a log-normal \(r = 10^y\) with \(y \sim \mathcal{N}(\mu, \sigma^2)\):
  \[
  \mathrm{Var}(r_{\rm eff} \mid M) = \bigl(\mathbb{E}[r_{\rm eff} \mid M]\bigr)^2 \cdot \bigl(10^{\sigma^2 (\ln 10)^2} - 1\bigr).
  \]
  With \(\sigma^2 = 0.0441\): \(10^{\sigma^2 (\ln 10)^2} \approx 1.233\), so:
  \[
  \boxed{
  \mathrm{Var}(r_{\rm eff} \mid M) \approx (3.63\cdot M^{0.1415})^2 \cdot 0.233 \approx 3.06\cdot M^{0.283}\ \mathrm{pc}^2.
  }
  \]

**Note:** In `mass_to_radius()` the variable `sigma_mr` is set to `-0.2103855`. The negative sign is a bug; the intended scatter is \(\sigma = 0.21\). The distribution of \(\varepsilon\) is still \(\mathcal{N}(0, \sigma^2)\) because the code adds `randn * sigma_mr` and the sign only flips the sign of the normal draw.

---

## 3. Summary table

| Prescription | Mean \(\log_{10}(r_{\rm eff})\) | Var(\(\log_{10}(r_{\rm eff})\)) | Distribution |
|-------------|----------------------------------|----------------------------------|--------------|
| **Flat (pipeline)** | \(r_{\rm eff}^{\rm fixed}\) (no log formula) | 0 | Deterministic: \(r_{\rm eff} = \texttt{eradius}\) |
| **Flat (code)**    | \(0.5\) (independent of \(M\)) | \(1/12\) | \(\log_{10}(r_{\rm eff}) \sim \mathrm{Uniform}(0,1)\) |
| **Krumholz19**    | \(0.1415\,\log_{10}(M)\) | \(0.21^2 = 0.0441\) | Gaussian in log space |

---

## 4. Ryon17 (for reference)

In `mass_to_radius(..., model="Ryon17")` the code uses a **deterministic** relation:

\[
\log_{10}(r_{\rm eff}) = -7.775 + 1.674\,\log_{10}(M),
\]
with a cap for \(\log_{10}(M) > 5.2\) (then \(\log_{10}(r_{\rm eff})\) is fixed at \(-7.775 + 1.674 \times 5.2\)). So:

- **Mean:** \(\mathbb{E}[\log_{10}(r_{\rm eff}) \mid M] = -7.775 + 1.674\,\log_{10}(M)\) (or capped).
- **Variance:** \(\mathrm{Var}(\log_{10}(r_{\rm eff}) \mid M) = 0\).

This is not one of the two main “flat vs Krumholz19” prescriptions used for the completeness pipeline sampling above.
