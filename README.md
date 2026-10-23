# FracPotMW 

`FracPotMW.py` is a Python pipeline for fitting Milky Way rotation curves and comparing four spherical bulge models within a common bulge–disk–halo decomposition. It combines numerical optimization with optional Bayesian nested sampling and includes robustness checks for the disk prescription, radial selection, and fractional-kernel resolution.

The code accompanies the research project **“A Bayesian Test of Fractional Gravitational Deformations in the Milky Way Bulge”** (Matheus Agenor and Thiago M. Marchesin).

## Models and assumptions

The modeled circular speed is

$$
v_{\mathrm{model}}^2(R)=v_{\mathrm{bulge}}^2(R)+v_{\mathrm{disk}}^2(R)+v_{\mathrm{halo}}^2(R).
$$

Four alternative prescriptions are available for the **spherical bulge**:

| Model name in code | Description |
| --- | --- |
| `exp_newtonian` | Spherical exponential density under the standard Poisson equation (reference model, \(s=1\)). |
| `exp_fractional` | The same exponential density with a phenomenological fractional gravitational response and a free order \(s\). |
| `sersic` | Three-dimensional Prugniel–Simien approximation to a Sérsic profile. |
| `einasto` | Three-dimensional Einasto profile. |

The exponential bulge density is

$$
\rho_{\mathrm b}(r)=\frac{M_{\mathrm b}\epsilon^3}{8\pi}\,e^{-\epsilon r},
\qquad r_{\mathrm b}=\epsilon^{-1}.
$$

Here, $M_{\mathrm b}$ is the bulge mass, $\epsilon$ has units of kpc$^{-1}$, and $r_{\mathrm b}$ is the exponential scale length in **kpc**. The fractional model is constructed to reproduce the reference model exactly at $s=1$; its reference length is $\ell_0=1$ kpc by default.

All four models use the same *functional form* for the NFW dark-matter halo, but **the halo parameters are fitted**, not held fixed. The thin exponential disk is treated in two ways:

- **Fixed disk:** $M_{\mathrm d}=6.0\times10^{10}\,M_\odot$ and $R_{\mathrm d}=5.4$ kpc.
- **Marginalized disk:** $M_{\mathrm d}$ and $R_{\mathrm d}$ are free parameters, with the same prior ranges for every bulge model.

The fractional prescription is a controlled phenomenological benchmark, not a complete theory of Galactic dynamics. The fitting assumes spherical bulge and halo components, an axisymmetric thin disk, and independent Gaussian observational errors.

## Observational datasets

The pipeline supports three published Milky Way rotation curves:

- **Sofue (2020)** — dataset key `sofue2020`.
- **Sofue & Kohno (2025)** — dataset key `sofue2025`.
- **Huang et al. (2016)** — dataset key `huang2016`.

**The observational data files are not embedded in `FracPotMW.py`.** Supply numerical tables obtained from the relevant publications, respecting their data-use conditions. The three datasets are fitted independently; their Bayesian evidences are not combined into one global Bayes factor.

### Input format

Each input file must contain numerical columns for Galactocentric radius $R$, observed circular speed $v$, and velocity uncertainty $\sigma$:

```text
# R [kpc]   v [km/s]   sigma [km/s]
0.10        100.0      5.0
0.20        120.0      6.0
0.30        140.0      6.5
# ... more observational rows ...
```

The numbers above illustrate the **file format only**; they are not observational measurements. A real dataset must contain at least five valid rows. Whitespace-, comma-, semicolon-, and tab-delimited numeric tables are supported, with `#` comments. Headers should be commented out.

By default, columns **0, 1, 2** are interpreted as $(R,v,\sigma)$ for three-column tables, or columns **0, 1, 3** for tables with four or more columns. Column indices start at zero. Override the mapping when needed with `--cols-sofue2020`, `--cols-sofue2025`, or `--cols-huang2016`, using a string such as `0,1,2`.

The script assumes radii are in kpc unless explicitly specified or automatically detected as pc (the `auto` setting converts radii when the largest finite value exceeds 500). To avoid ambiguity, explicitly provide `--radius-unit-<dataset> kpc` or `pc` for the corresponding dataset.

A convenient directory layout is:

```text
FracPotMW/
├── FracPotMW.py
├── README.md
└── input/
    ├── MW_Sofue2020.dat
    ├── MW_SofueKohno2025.dat
    └── MW_Huang2016.dat
```

These filenames are recognized automatically. Other filenames can be used with the explicit `--data-sofue2020`, `--data-sofue2025`, and `--data-huang2016` options.

## Requirements

- **Python 3.10+**
- `numpy`
- `scipy`
- `pandas`
- `matplotlib`
- `dynesty` for Bayesian evidence and posterior samples (**the default sampler**)
- `corner` (optional) for corner plots
- `mpi4py` and an MPI installation (optional) for distributed execution

Install the standard dependencies with:

```bash
python -m pip install numpy scipy pandas matplotlib dynesty corner
```

For MPI execution, install and configure an MPI implementation (e.g., Open MPI or MPICH) and then install `mpi4py` according to your system's MPI configuration. MPI is not required for serial runs.

## Running the code

Run the **full default analysis** (three datasets, four bulge models, both disk assumptions, robustness tests, and `dynesty`):

```bash
python FracPotMW.py --input-dir input --outdir results
```

The default analysis can be computationally expensive. For an initial **optimization-only** run on one dataset, without additional robustness cases or corner plots:

```bash
python FracPotMW.py \
  --input-dir input \
  --datasets sofue2020 \
  --disk-mode fixed \
  --sampler optimize \
  --kernel-check-nk 0 \
  --corner-plots none \
  --outdir results
```

For **Bayesian inference on a single dataset** using the production fractional-kernel resolution:

```bash
python FracPotMW.py \
  --input-dir input \
  --datasets sofue2020 \
  --disk-mode both \
  --sampler dynesty \
  --nlive 600 \
  --dlogz 0.05 \
  --frac-nk 1536 \
  --outdir results
```

For parallel execution across independent fitting cases (MPI ranks):

```bash
mpiexec -n 4 python FracPotMW.py --input-dir input --outdir results
```

The MPI implementation distributes independent cases between ranks; it does not parallelize `dynesty` internally. Local `--workers` default to one to avoid oversubscription with MPI.

Display all command-line options with:

```bash
python FracPotMW.py --help
```

### Useful options

| Option | Default | Purpose |
| --- | --- | --- |
| `--datasets` | All three | Choose one or more datasets. |
| `--models` | All four | Select the bulge models to fit. |
| `--disk-mode` | `both` | `fixed`, `marginalized`, or `both`. |
| `--sampler` | `dynesty` | Use `dynesty` or `optimize` only. |
| `--frac-nk` | `1536` | Quadrature nodes for the fractional kernel in production runs. |
| `--kernel-check-nk` | `384` | Resolution-check nodes; set to `0` to disable. |
| `--nlive` | `600` | Live points for nested sampling. |
| `--dlogz` | `0.05` | Nested-sampling stopping tolerance. |
| `--seed` | `12345` | Base random seed. |
| `--corner-plots` | `fractional` | `none`, `fractional`, or `all`. |
| `--skip-case-plots` | off | Skip per-case rotation-curve plots. |
| `--skip-tests` | off | Skip built-in numerical consistency checks. |

Parameter-prior bounds, the fractional-order interval, and numerical integration limits are also configurable through command-line options. Consult `--help` and the saved `run_config.json` for exact settings.

## Robustness analyses

When the relevant datasets are selected, the pipeline constructs additional fits to test:

1. **Inner radial cuts** for Sofue & Kohno (2025): $R_{\min}=0.05, 0.10, 0.20, 0.50, 1.00$ kpc.
2. **Matched radial coverage:** $0.10\le R\le25.57$ kpc for Sofue (2020) and Sofue & Kohno (2025).
3. **Fractional-kernel resolution:** production $N_q=1536$ versus a lower-resolution $N_q=384$ comparison, by default.
4. **Disk-model dependence:** fixed and jointly fitted disk mass/scale length.

The code performs internal checks of the Newtonian $s=1$ limit and fractional-kernel numerical validity before fitting, unless `--skip-tests` is provided. These checks are not a substitute for independent scientific validation.

## Output files

Each run creates a timestamped directory inside `--outdir`, for example:

```text
results/
└── run_YYYYMMDD_HHMMSS/
    ├── run_config.json
    ├── input_preflight.csv
    ├── summary_metrics_all_cases.csv
    ├── bestfit_parameters_all_cases.csv
    ├── posterior_summary_all_cases.csv
    ├── radial_cut_summary.csv
    ├── common_range_summary.csv
    ├── kernel_resolution_fit_comparison.csv
    ├── disk_fixed_vs_marginalized.csv
    ├── marginalized_disk_parameters.csv
    ├── report_robustness.md
    ├── disk_fixed/
    │   ├── baseline_sofue2020/
    │   │   ├── summary_metrics.csv
    │   │   ├── bestfit_parameters.csv
    │   │   ├── posterior_summary.csv
    │   │   ├── results_exp_newtonian.npz
    │   │   ├── ...
    │   │   └── report.md
    │   └── ...
    └── disk_marginalized/
        └── ...
```

The primary fit statistics are $\chi^2$, reduced $\chi^2$, AIC, BIC, and (with `dynesty`) $\ln Z$. Posterior summaries contain parameter quantiles and, for the fractional model, $P(s>1)$. **`--sampler optimize` does not calculate Bayesian evidence or posterior credible intervals.**

The inferred evidence depends on the adopted priors; compare models only under the specified assumptions and shared prior conventions.

##  Citation

If this code contributes to a publication, please cite the associated study:

> Matheus Agenor and Thiago M. Marchesin, *A Bayesian Test of Fractional Gravitational Deformations in the Milky Way Bulge* (manuscript).

