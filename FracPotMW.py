# -*- coding: utf-8 -*-
"""Rotation-curve fits of the Milky Way: exponential, fractional, Sersic and Einasto bulges.

Fits Sofue (2020), Sofue & Kohno (2025), and Huang et al. (2016).
The NFW halo is common to all models. Both fixed and fitted exponential
thin disks are supported. Nested sampling uses dynesty; MPI distributes
independent fits, if mpi4py is installed.

Usage: python FracPotMW.py --input-dir input --outdir results
       mpiexec -n 4 python FracPotMW_public.py --input-dir input

Required data: columns R [kpc], v [km/s], sigma [km/s]; by default,
columns 0, 1, and 3 are used for 4-column tables, or 0, 1, 2 for 3.
For the exponential bulge epsilon has units kpc^-1 (not kpc).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
import traceback
import dataclasses
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import special
from scipy.optimize import differential_evolution, minimize
from scipy.special import i0, i1, k0, k1

G_ASTRO = 6.67430e-11 * 1.98847e30 / (3.08567758128e19 * 1000.0**2)
DISK_MASS_MSUN = 6.0e10
DISK_SCALE_KPC = 5.4
RNG_SEED = 12345
DEFAULT_FRAC_NK_PRODUCTION = 1536
DEFAULT_KERNEL_CHECK_NK = 384
COMMON_RMIN_KPC = 0.10
COMMON_RMAX_KPC = 25.57
SOFUE2025_RMIN_CUTS = (0.05, 0.10, 0.20, 0.50, 1.00)

# Models and likelihood

@dataclasses.dataclass(frozen=True)
class Parameter:
    name: str
    lower: float
    upper: float
    label: str
    unit: str = ""

@dataclasses.dataclass
class FitResult:
    model_name: str
    theta_best: np.ndarray
    loglike_best: float
    chi2: float
    aic: float
    bic: float
    red_chi2: float
    success: bool
    message: str
    param_names: List[str]
    logz: float = np.nan
    logzerr: float = np.nan
    samples_equal: Optional[np.ndarray] = None
    loglikes: Optional[np.ndarray] = None

@dataclasses.dataclass
class DataSet:
    radius_kpc: np.ndarray
    velocity_kms: np.ndarray
    sigma_kms: np.ndarray

    @property
    def n(self) -> int:
        return int(self.radius_kpc.size)

def v2_disk_thin_exponential(
    r_kpc: np.ndarray,
    mass_msun: float = DISK_MASS_MSUN,
    scale_kpc: float = DISK_SCALE_KPC,
) -> np.ndarray:
    """Velocidade ao quadrado para disco exponencial fino.

    Fórmula padrão:
        v_d^2(R) = 2 G M_d/R_d * y^2 [I0(y)K0(y)-I1(y)K1(y)],
        y = R/(2 R_d).

    Essa é a mesma normalização usada nos scripts originais. O texto do paper
    deve refletir exatamente esta fórmula.
    """
    r = np.asarray(r_kpc, dtype=float)
    y = r / (2.0 * scale_kpc)
    term = y**2 * (i0(y) * k0(y) - i1(y) * k1(y))
    v2 = (2.0 * G_ASTRO * mass_msun / scale_kpc) * term
    return np.where(np.isfinite(v2), np.maximum(v2, 0.0), np.nan)

def v2_halo_nfw(r_kpc: np.ndarray, rho_s_msun_pc3: float, r_s_kpc: float) -> np.ndarray:
    """Halo NFW em termos de rho_s [Msun/pc^3] e r_s [kpc]."""
    r = np.asarray(r_kpc, dtype=float)
    r_safe = np.maximum(r, 1e-12)
    rho_s_msun_kpc3 = rho_s_msun_pc3 * 1.0e9
    x = r_safe / r_s_kpc
    mass_enc = 4.0 * np.pi * rho_s_msun_kpc3 * r_s_kpc**3 * (
        np.log1p(x) - x / (1.0 + x)
    )
    v2 = G_ASTRO * mass_enc / r_safe
    return np.where(np.isfinite(v2), np.maximum(v2, 0.0), np.nan)

def mass_fraction_exponential_sphere(r_kpc: np.ndarray, epsilon_kpc_inv: float) -> np.ndarray:
    """M(<r)/M_total para rho(r) = M epsilon^3 exp(-epsilon r)/(8 pi)."""
    x = np.maximum(epsilon_kpc_inv * np.asarray(r_kpc, dtype=float), 0.0)
    frac = 1.0 - np.exp(-x) * (1.0 + x + 0.5 * x**2)
    return np.clip(frac, 0.0, 1.0)

def v2_bulge_exponential_newtonian(
    r_kpc: np.ndarray,
    mass_msun: float,
    epsilon_kpc_inv: float,
) -> np.ndarray:
    r = np.asarray(r_kpc, dtype=float)
    r_safe = np.maximum(r, 1e-12)
    frac = mass_fraction_exponential_sphere(r_safe, epsilon_kpc_inv)
    return G_ASTRO * mass_msun * frac / r_safe

def v2_bulge_sersic_prugniel_simien(
    r_kpc: np.ndarray,
    mass_msun: float,
    r_e_kpc: float,
    n_sersic: float,
) -> np.ndarray:
    """Bojo Sérsic 3D via aproximação Prugniel-Simien, normalizado por M_total."""
    r = np.asarray(r_kpc, dtype=float)
    r_safe = np.maximum(r, 1e-12)
    n = max(float(n_sersic), 1e-6)
    p = 1.0 - 0.6097 / n + 0.05463 / n**2
    b_n = 2.0 * n - 1.0 / 3.0 + 4.0 / (405.0 * n) + 46.0 / (25515.0 * n**2)
    x = b_n * np.power(r_safe / r_e_kpc, 1.0 / n)
    shape = (3.0 - p) * n
    frac = special.gammainc(shape, np.clip(x, 1e-300, 1e300))
    return G_ASTRO * mass_msun * frac / r_safe

def v2_bulge_einasto(
    r_kpc: np.ndarray,
    mass_msun: float,
    r_e_kpc: float,
    n_einasto: float,
) -> np.ndarray:
    """Bojo Einasto 3D normalizado por M_total."""
    r = np.asarray(r_kpc, dtype=float)
    r_safe = np.maximum(r, 1e-12)
    n = max(float(n_einasto), 1e-6)
    d_n = 3.0 * n - 1.0 / 3.0 + 0.0079 / n
    x = d_n * np.power(r_safe / r_e_kpc, 1.0 / n)
    frac = special.gammainc(3.0 * n, np.clip(x, 1e-300, 1e300))
    return G_ASTRO * mass_msun * frac / r_safe

class FractionalKernel:
    """Fractional kernel in q=k/epsilon, normalized to recover s=1 exactly.

    The oscillatory integral uses a Gauss-Legendre rule in log(q); large
    epsilon*r uses its asymptotic form. The s=1 denominator is analytical.
    """

    def __init__(
        self,
        nk: int = 384,
        qmin: float = 1e-6,
        qmax: float = 100.0,
        ell0_kpc: float = 1.0,
        asymptotic_transition_width: float = 6.0,
    ) -> None:
        if nk < 32:
            raise ValueError("nk deve ser >= 32 para a quadratura fracionária.")
        if not np.isfinite(qmin) or qmin <= 0.0:
            raise ValueError("qmin deve ser positivo e finito.")
        if not np.isfinite(qmax) or qmax <= qmin:
            raise ValueError("qmax deve ser finito e maior que qmin.")
        if not np.isfinite(ell0_kpc) or ell0_kpc <= 0.0:
            raise ValueError("ell0_kpc deve ser positivo e finito.")
        if asymptotic_transition_width <= 0.0:
            raise ValueError("asymptotic_transition_width deve ser positivo.")

        self.nk = int(nk)
        self.qmin = float(qmin)
        self.qmax = float(qmax)
        self.ell0_kpc = float(ell0_kpc)
        self.asymptotic_transition_width = float(asymptotic_transition_width)

        xg, wg = np.polynomial.legendre.leggauss(self.nk)
        log_min = math.log(self.qmin)
        log_max = math.log(self.qmax)
        half = 0.5 * (log_max - log_min)
        mid = 0.5 * (log_max + log_min)
        self._q = np.exp(mid + half * xg)
        self._w_logq = half * wg

    def _logq_nodes(self) -> Tuple[np.ndarray, np.ndarray]:
        return self._q, self._w_logq

    @staticmethod
    def _mass_fraction_from_x(x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        return np.clip(1.0 - np.exp(-x) * (1.0 + x + 0.5 * x**2), 0.0, 1.0)

    @classmethod
    def _dimensionless_integral_newtonian(cls, x: np.ndarray) -> np.ndarray:
        return -0.5 * np.pi * cls._mass_fraction_from_x(x)

    @staticmethod
    def _asymptotic_coefficient(s: float) -> float:
        ss = float(s)
        if abs(ss - 1.0) < 1e-7:
            return -0.5 * np.pi
        coeff = -(3.0 - 2.0 * ss) * special.gamma(2.0 - 2.0 * ss) * np.sin(np.pi * ss)
        return float(coeff)

    @staticmethod
    def _smoothstep(t: np.ndarray) -> np.ndarray:
        tc = np.clip(t, 0.0, 1.0)
        return tc * tc * (3.0 - 2.0 * tc)

    def _transition_bounds(self, s: float) -> Tuple[float, float]:
        """Switch smoothly from quadrature to the large-radius asymptotic form."""
        ss = float(np.clip(s, 0.3, 1.0))
        center = 9.0 + 15.0 * (ss - 0.3) / 0.7
        half_width = 0.5 * self.asymptotic_transition_width
        return center - half_width, center + half_width

    def dimensionless_integral(self, x: np.ndarray, s: float) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        x_safe = np.maximum(x, 1e-12)
        ss = float(s)

        if not np.isfinite(ss):
            return np.full_like(x_safe, np.nan)

        if abs(ss - 1.0) < 1e-12:
            return self._dimensionless_integral_newtonian(x_safe)

        q, w = self._logq_nodes()
        phase = x_safe[:, None] * q[None, :]
        numerator = phase * np.cos(phase) - np.sin(phase)
        q_weight = np.power(q, 2.0 - 2.0 * ss) / np.power(1.0 + q**2, 2.0)
        integral_quad = np.sum(numerator * q_weight[None, :] * w[None, :], axis=1)

        coeff = self._asymptotic_coefficient(ss)
        integral_asym = coeff * np.power(x_safe, 2.0 * ss - 2.0)

        x_lo, x_hi = self._transition_bounds(ss)
        blend = self._smoothstep((x_safe - x_lo) / (x_hi - x_lo))
        integral = (1.0 - blend) * integral_quad + blend * integral_asym
        integral[~np.isfinite(integral)] = np.nan
        return integral

    def raw_shape(
        self,
        r_kpc: np.ndarray,
        epsilon_kpc_inv: float,
        s: float,
    ) -> np.ndarray:
        r = np.asarray(r_kpc, dtype=float)
        r_safe = np.maximum(r, 1e-10)
        eps = float(epsilon_kpc_inv)
        ss = float(s)

        if not np.isfinite(eps) or eps <= 0.0 or not np.isfinite(ss):
            return np.full_like(r_safe, np.nan, dtype=float)

        x = eps * r_safe
        integral = self.dimensionless_integral(x, ss)
        raw = -np.power(eps, -2.0 * ss - 2.0) * integral / r_safe
        raw[~np.isfinite(raw)] = np.nan
        return raw

    def raw_shape_newtonian_exact(
        self,
        r_kpc: np.ndarray,
        epsilon_kpc_inv: float,
    ) -> np.ndarray:
        r = np.asarray(r_kpc, dtype=float)
        r_safe = np.maximum(r, 1e-10)
        eps = float(epsilon_kpc_inv)
        x = eps * r_safe
        integral_1 = self._dimensionless_integral_newtonian(x)
        raw_1 = -np.power(eps, -4.0) * integral_1 / r_safe
        raw_1[~np.isfinite(raw_1)] = np.nan
        return raw_1

    def v2_fractional_exponential(
        self,
        r_kpc: np.ndarray,
        mass_msun: float,
        epsilon_kpc_inv: float,
        s: float,
    ) -> np.ndarray:
        v2_newt = v2_bulge_exponential_newtonian(r_kpc, mass_msun, epsilon_kpc_inv)
        ss = float(s)
        if abs(ss - 1.0) < 1e-12:
            return v2_newt

        raw_s = self.raw_shape(r_kpc, epsilon_kpc_inv, ss)
        raw_1 = self.raw_shape_newtonian_exact(r_kpc, epsilon_kpc_inv)
        ratio = (raw_s / raw_1) * self.ell0_kpc ** (2.0 * (1.0 - ss))
        v2 = v2_newt * ratio

        invalid = (
            (~np.isfinite(v2))
            | (~np.isfinite(ratio))
            | (raw_1 <= 0.0)
            | (raw_s <= 0.0)
            | (ratio <= 0.0)
        )
        if np.any(invalid):
            v2 = np.asarray(v2, dtype=float)
            v2[invalid] = np.nan
        return v2

def default_priors(args: argparse.Namespace) -> Dict[str, List[Parameter]]:
    """Priors comuns a todos os modelos.

    Os limites do halo são configuráveis pela linha de comando, mas são sempre
    aplicados igualmente aos quatro modelos para preservar a comparação justa.
    """
    common_mass = Parameter("log10_Mb", args.logmb_min, args.logmb_max, r"$\log_{10}(M_{\rm b}/M_\odot)$")
    halo_rho = Parameter(
        "log10_rho_s",
        args.halo_logrho_min,
        args.halo_logrho_max,
        r"$\log_{10}[\rho_{\rm s}/(M_\odot\,{\rm pc}^{-3})]$",
    )
    halo_rs = Parameter(
        "log10_r_s",
        args.halo_logrs_min,
        args.halo_logrs_max,
        r"$\log_{10}(r_{\rm s}/{\rm kpc})$",
    )

    log_epsilon = Parameter(
        "log10_epsilon_exp",
        args.logepsilon_min,
        args.logepsilon_max,
        r"$\log_{10}(\epsilon/{\rm kpc}^{-1})$",
    )
    s_frac = Parameter("s", args.s_min, args.s_max, r"$s$")
    log_re = Parameter("log10_R_e", args.logre_min, args.logre_max, r"$\log_{10}(R_{\rm e}/{\rm kpc})$")
    n_shape = Parameter("n", args.n_min, args.n_max, r"$n$")

    return {
        "exp_newtonian": [common_mass, log_epsilon, halo_rho, halo_rs],
        "exp_fractional": [common_mass, log_epsilon, s_frac, halo_rho, halo_rs],
        "sersic": [common_mass, log_re, n_shape, halo_rho, halo_rs],
        "einasto": [common_mass, log_re, n_shape, halo_rho, halo_rs],
    }

def transform_unit_cube(u: np.ndarray, params: Sequence[Parameter]) -> np.ndarray:
    u = np.asarray(u, dtype=float)
    return np.array([p.lower + ui * (p.upper - p.lower) for ui, p in zip(u, params)], dtype=float)

def bounds_for(params: Sequence[Parameter]) -> List[Tuple[float, float]]:
    return [(p.lower, p.upper) for p in params]

def unpack_common_halo(theta_tail: Sequence[float]) -> Tuple[float, float]:
    log10_rho_s, log10_r_s = theta_tail
    return 10.0**log10_rho_s, 10.0**log10_r_s

def v2_bulge_by_model(
    model_name: str,
    r_kpc: np.ndarray,
    theta: np.ndarray,
    frac_kernel: FractionalKernel,
) -> np.ndarray:
    log10_Mb = theta[0]
    mass = 10.0**log10_Mb

    if model_name == "exp_newtonian":
        epsilon_exp = 10.0**theta[1]
        return v2_bulge_exponential_newtonian(r_kpc, mass, epsilon_exp)

    if model_name == "exp_fractional":
        epsilon_exp = 10.0**theta[1]
        s = theta[2]
        return frac_kernel.v2_fractional_exponential(r_kpc, mass, epsilon_exp, s)

    if model_name == "sersic":
        r_e = 10.0**theta[1]
        n = theta[2]
        return v2_bulge_sersic_prugniel_simien(r_kpc, mass, r_e, n)

    if model_name == "einasto":
        r_e = 10.0**theta[1]
        n = theta[2]
        return v2_bulge_einasto(r_kpc, mass, r_e, n)

    raise ValueError(f"Modelo desconhecido: {model_name}")

# Data and fitting

try:
    from mpi4py import MPI  # type: ignore
    MPI_AVAILABLE = True
    MPI_COMM = MPI.COMM_WORLD
    MPI_RANK = int(MPI_COMM.Get_rank())
    MPI_SIZE = int(MPI_COMM.Get_size())
except Exception:
    MPI_AVAILABLE = False
    MPI_RANK = 0
    MPI_SIZE = 1

    class _SerialComm:
        def bcast(self, value, root=0):
            return value
        def barrier(self):
            return None
        Barrier = barrier
        def gather(self, value, root=0):
            return [value]

    MPI_COMM = _SerialComm()

@dataclasses.dataclass(frozen=True)
class DatasetSpec:
    key: str
    label: str
    source: str
    candidates: Tuple[str, ...]
    default_cols: Tuple[int, int, int]

DATASET_SPECS: Dict[str, DatasetSpec] = {
    "sofue2020": DatasetSpec(
        key="sofue2020",
        label="Sofue (2020)",
        source="Milky Way Grand Rotation Curve / tabela fornecida pelo usuário",
        candidates=(
            "MW_Sofue2020.dat", "Sofue2020.dat", "sofue2020.dat",
            "sofue_2020.dat", "MW.dat", "MW2020.dat",
            "sofue2020.txt", "sofue2020.csv",
        ),
        default_cols=(0, 1, 3),
    ),
    "sofue2025": DatasetSpec(
        key="sofue2025",
        label="Sofue & Kohno (2025)",
        source="The inner rotation curve of the Milky Way / curva unificada",
        candidates=(
            "MW_SofueKohno2025.dat", "SofueKohno2025.dat",
            "sofue2025.dat", "sofue_2025.dat", "Sofue2025.dat",
            "sofue_kohno_2025.dat", "MW2025.dat",
            "sofue2025.txt", "sofue2025.csv",
        ),
        default_cols=(0, 1, 3),
    ),
    "huang2016": DatasetSpec(
        key="huang2016",
        label="Huang et al. (2016)",
        source="MNRAS 463, 2623 / Table 3",
        candidates=(
            "MW_Huang2016.dat", "Huang2016.dat", "huang2016.dat",
            "huang_2016.dat", "huang2016.txt", "huang2016.csv",
            "Huang_Table3.dat",
        ),
        default_cols=(0, 1, 3),
    ),
}

@dataclasses.dataclass(frozen=True)
class RobustnessCase:
    case_id: str
    dataset_key: str
    disk_mode: str
    rmin_kpc: Optional[float]
    rmax_kpc: Optional[float]
    frac_nk: int
    category: str
    models: Tuple[str, ...]
    note: str = ""

@dataclasses.dataclass
class LoadedDataset:
    spec: DatasetSpec
    path: Path
    data: DataSet

def mpi_print(message: str, *, root_only: bool = False) -> None:
    if root_only and MPI_RANK != 0:
        return
    prefix = f"[rank {MPI_RANK:02d}/{MPI_SIZE:02d}] " if MPI_SIZE > 1 else ""
    print(prefix + message, flush=True)

def stable_seed(base: int, *parts: str) -> int:
    text = "|".join(parts).encode("utf-8")
    value = int.from_bytes(hashlib.sha256(text).digest()[:4], "little")
    return int((int(base) + value) % (2**31 - 1))

def _parse_cols(text: Optional[str], default: Tuple[int, int, int]) -> Tuple[int, int, int]:
    if text is None:
        return default
    vals = [int(x.strip()) for x in text.split(",")]
    if len(vals) != 3 or min(vals) < 0:
        raise ValueError("Colunas devem ser informadas como R,V,sigma, por exemplo 0,1,3.")
    return tuple(vals)  # type: ignore[return-value]

def _read_numeric_table(path: Path) -> np.ndarray:
    """Read numeric whitespace tables, with CSV fallback."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)

    attempts = [
        (None, "whitespace"),
        (",", "comma"),
        (";", "semicolon"),
        ("\t", "tab"),
    ]
    errors = []
    for delimiter, label in attempts:
        try:
            arr = np.loadtxt(
                path,
                comments="#",
                delimiter=delimiter,
                ndmin=2,
                dtype=float,
            )
            arr = np.asarray(arr, dtype=float)
            if arr.ndim == 2 and arr.shape[0] >= 3 and arr.shape[1] >= 3:
                return arr
            errors.append(f"{label}: shape={getattr(arr, 'shape', None)}")
        except Exception as exc:
            errors.append(f"{label}: {type(exc).__name__}: {exc}")

    raise ValueError(
        f"Não consegui ler a tabela numérica {path}. "
        "Esperava >=3 colunas numéricas (R, V, sigma). Tentativas: "
        + " | ".join(errors)
    )

def _resolve_dataset_path(args: argparse.Namespace, key: str) -> Path:
    explicit = getattr(args, f"data_{key}")
    if explicit:
        p = Path(explicit).expanduser().resolve()
        if not p.exists():
            raise FileNotFoundError(f"Arquivo de {key} não encontrado: {p}")
        return p

    input_dir = Path(args.input_dir).expanduser().resolve()
    spec = DATASET_SPECS[key]
    for name in spec.candidates:
        p = input_dir / name
        if p.exists():
            return p
    candidates = "\n  - ".join(str(input_dir / x) for x in spec.candidates)
    raise FileNotFoundError(
        f"Não encontrei os dados de {key}. Use --data-{key.replace('_','-')} CAMINHO "
        f"ou coloque um dos arquivos esperados em --input-dir:\n  - {candidates}"
    )

def load_named_dataset(args: argparse.Namespace, key: str) -> LoadedDataset:
    spec = DATASET_SPECS[key]
    path = _resolve_dataset_path(args, key)
    arr = _read_numeric_table(path)
    cols_text = getattr(args, f"cols_{key}")
    if cols_text is None:
        cols = (0, 1, 3 if arr.shape[1] >= 4 else 2)
    else:
        cols = _parse_cols(cols_text, spec.default_cols)
    if max(cols) >= arr.shape[1]:
        raise ValueError(
            f"{key}: colunas {cols} incompatíveis com tabela de {arr.shape[1]} colunas em {path}."
        )
    r = np.asarray(arr[:, cols[0]], dtype=float)
    v = np.asarray(arr[:, cols[1]], dtype=float)
    sig = np.asarray(arr[:, cols[2]], dtype=float)

    unit = getattr(args, f"radius_unit_{key}")
    finite_r = r[np.isfinite(r)]
    if unit == "pc" or (unit == "auto" and finite_r.size and np.nanmax(finite_r) > 500.0):
        r = r / 1000.0

    ok = np.isfinite(r) & np.isfinite(v) & np.isfinite(sig) & (r > 0.0) & (sig > 0.0)
    r, v, sig = r[ok], v[ok], sig[ok]
    order = np.argsort(r)
    data = DataSet(r[order], v[order], sig[order])
    if data.n < 5:
        raise ValueError(f"{key}: apenas {data.n} pontos válidos após leitura.")
    return LoadedDataset(spec=spec, path=path, data=data)

def subset_dataset(data: DataSet, rmin: Optional[float], rmax: Optional[float]) -> DataSet:
    mask = np.ones(data.n, dtype=bool)
    if rmin is not None:
        mask &= data.radius_kpc >= float(rmin) - 1e-12
    if rmax is not None:
        mask &= data.radius_kpc <= float(rmax) + 1e-12
    out = DataSet(data.radius_kpc[mask], data.velocity_kms[mask], data.sigma_kms[mask])
    if out.n < 5:
        raise ValueError(
            f"Corte radial deixou apenas {out.n} pontos (rmin={rmin}, rmax={rmax})."
        )
    return out

def priors_for_disk_mode(args: argparse.Namespace, disk_mode: str) -> Dict[str, List[Parameter]]:
    base = default_priors(args)
    if disk_mode == "fixed":
        return base
    if disk_mode != "marginalized":
        raise ValueError(f"disk_mode inválido: {disk_mode}")

    disk_mass = Parameter(
        "log10_Md",
        args.disk_logmass_min,
        args.disk_logmass_max,
        r"$\log_{10}(M_{\rm d}/M_\odot)$",
    )
    disk_scale = Parameter(
        "log10_Rd",
        args.disk_logrd_min,
        args.disk_logrd_max,
        r"$\log_{10}(R_{\rm d}/{\rm kpc})$",
    )
    out: Dict[str, List[Parameter]] = {}
    for model, pars in base.items():
        out[model] = list(pars[:-2]) + [disk_mass, disk_scale] + list(pars[-2:])
    return out

def v_components(model_name, r_kpc, theta, frac_kernel, disk_mode):
    """Bulge, disk and NFW halo contributions in km/s."""
    rho_s, r_s = unpack_common_halo(theta[-2:])
    if disk_mode == "fixed":
        md, rd = DISK_MASS_MSUN, DISK_SCALE_KPC
    else:
        md, rd = 10.0**theta[-4], 10.0**theta[-3]
    v2_b = v2_bulge_by_model(model_name, r_kpc, theta, frac_kernel)
    v2_d = v2_disk_thin_exponential(r_kpc, mass_msun=md, scale_kpc=rd)
    v2_h = v2_halo_nfw(r_kpc, rho_s, r_s)
    v2_total = v2_b + v2_d + v2_h
    return {
        "bulge": np.sqrt(np.maximum(v2_b, 0.0)),
        "disk": np.sqrt(np.maximum(v2_d, 0.0)),
        "halo": np.sqrt(np.maximum(v2_h, 0.0)),
        "total": np.sqrt(np.maximum(v2_total, 0.0)),
        "v2_total": v2_total,
    }

def log_likelihood(
    theta: np.ndarray,
    model_name: str,
    data: DataSet,
    frac_kernel: FractionalKernel,
    disk_mode: str,
) -> float:
    try:
        comp = v_components(model_name, data.radius_kpc, theta, frac_kernel, disk_mode)
        vm = comp["total"]
        if np.any(~np.isfinite(vm)) or np.any(~np.isfinite(comp["v2_total"])) or np.any(comp["v2_total"] <= 0.0):
            return -np.inf
        chi2 = np.sum(((data.velocity_kms - vm) / data.sigma_kms) ** 2)
        return -0.5 * float(chi2) if np.isfinite(chi2) else -np.inf
    except Exception:
        return -np.inf

def fit_de(
    model_name: str,
    data: DataSet,
    params: Sequence[Parameter],
    frac_kernel: FractionalKernel,
    disk_mode: str,
    de_maxiter: int,
    de_popsize: int,
    seed: int,
    workers: int = 1,
    initial_guesses: Optional[Sequence[np.ndarray]] = None,
) -> Tuple[np.ndarray, float, bool, str]:
    bounds = bounds_for(params)

    def objective(x: np.ndarray) -> float:
        ll = log_likelihood(x, model_name, data, frac_kernel, disk_mode)
        return -ll if np.isfinite(ll) else 1e100

    result_de = differential_evolution(
        objective,
        bounds=bounds,
        maxiter=de_maxiter,
        popsize=de_popsize,
        tol=0.01,
        mutation=(0.5, 1.2),
        recombination=0.8,
        updating="immediate" if workers == 1 else "deferred",
        workers=workers,
        seed=seed,
        polish=False,
        disp=False,
    )
    candidates: List[Tuple[np.ndarray, float, bool, str]] = [
        (np.asarray(result_de.x, dtype=float), float(result_de.fun), bool(result_de.success), f"DE: {result_de.message}")
    ]
    starts = [np.asarray(result_de.x, dtype=float)]
    if initial_guesses:
        for guess in initial_guesses:
            g = np.asarray(guess, dtype=float)
            if g.shape == result_de.x.shape and np.all(np.isfinite(g)):
                starts.append(np.array([np.clip(x, lo, hi) for x, (lo, hi) in zip(g, bounds)]))
    for start in starts:
        polished = minimize(
            objective, x0=start, method="L-BFGS-B", bounds=bounds,
            options={"maxiter": 3000, "ftol": 1e-11, "gtol": 1e-8},
        )
        if np.all(np.isfinite(polished.x)) and np.isfinite(polished.fun):
            candidates.append((
                np.asarray(polished.x, dtype=float), float(polished.fun),
                bool(polished.success or result_de.success),
                f"DE: {result_de.message}; polish: {polished.message}",
            ))
    theta, fval, success, message = min(candidates, key=lambda item: item[1])
    return theta, -fval, success, message

def polish(
    model_name: str,
    data: DataSet,
    params: Sequence[Parameter],
    frac_kernel: FractionalKernel,
    disk_mode: str,
    start: np.ndarray,
) -> Tuple[np.ndarray, float, bool, str]:
    bounds = bounds_for(params)
    x0 = np.asarray(start, dtype=float)
    if x0.shape != (len(params),) or np.any(~np.isfinite(x0)):
        return x0, -np.inf, False, "Ponto inicial inválido para polish."
    x0 = np.array([np.clip(x, lo, hi) for x, (lo, hi) in zip(x0, bounds)], dtype=float)

    def objective(x: np.ndarray) -> float:
        ll = log_likelihood(x, model_name, data, frac_kernel, disk_mode)
        return -ll if np.isfinite(ll) else 1e100

    result = minimize(
        objective, x0=x0, method="L-BFGS-B", bounds=bounds,
        options={"maxiter": 4000, "ftol": 1e-12, "gtol": 1e-8},
    )
    theta = np.asarray(result.x, dtype=float)
    ll = log_likelihood(theta, model_name, data, frac_kernel, disk_mode)
    ok = np.all(np.isfinite(theta)) and np.isfinite(ll)
    return theta, float(ll), bool(ok), f"polish: {result.message}"

def sample_dynesty(
    model_name: str,
    data: DataSet,
    params: Sequence[Parameter],
    frac_kernel: FractionalKernel,
    disk_mode: str,
    nlive: int,
    dlogz: float,
    seed: int,
    print_progress: bool = False,
):
    try:
        import dynesty  # type: ignore
    except Exception as exc:
        mpi_print(f"[aviso] dynesty indisponível ({exc}); evidência omitida.")
        return np.nan, np.nan, None, None, None, -np.inf

    ndim = len(params)
    def ptform(u: np.ndarray) -> np.ndarray:
        return transform_unit_cube(u, params)
    def loglike(x: np.ndarray) -> float:
        return log_likelihood(x, model_name, data, frac_kernel, disk_mode)

    sampler = dynesty.NestedSampler(
        loglike, ptform, ndim, nlive=nlive, sample="rwalk", bound="multi",
        rstate=np.random.default_rng(seed),
    )
    sampler.run_nested(dlogz=dlogz, print_progress=print_progress)
    results = sampler.results
    logz = float(results.logz[-1])
    logzerr = float(results.logzerr[-1]) if hasattr(results, "logzerr") else np.nan
    try:
        from dynesty import utils as dyfunc  # type: ignore
        weights = np.exp(results.logwt - results.logz[-1])
        samples_equal = dyfunc.resample_equal(results.samples, weights)
    except Exception:
        samples_equal = np.asarray(results.samples)
    raw_samples = np.asarray(results.samples, dtype=float)
    loglikes = np.asarray(results.logl, dtype=float) if hasattr(results, "logl") else None
    theta_nested_best = None
    ll_nested_best = -np.inf
    if loglikes is not None and loglikes.size:
        finite = np.isfinite(loglikes)
        if np.any(finite):
            idxs = np.flatnonzero(finite)
            idx = int(idxs[np.argmax(loglikes[finite])])
            theta_nested_best = np.asarray(raw_samples[idx], dtype=float)
            ll_nested_best = float(loglikes[idx])
    return logz, logzerr, samples_equal, loglikes, theta_nested_best, ll_nested_best

def map_warm_start(
    source: FitResult,
    target_params: Sequence[Parameter],
    s_default: float = 1.0,
) -> np.ndarray:
    source_map = {name: float(val) for name, val in zip(source.param_names, source.theta_best)}
    vals = []
    for p in target_params:
        if p.name in source_map:
            vals.append(source_map[p.name])
        elif p.name == "s":
            vals.append(s_default)
        else:
            vals.append(0.5 * (p.lower + p.upper))
    return np.asarray(vals, dtype=float)

def compute_fit_result(
    model_name: str,
    data: DataSet,
    params: Sequence[Parameter],
    theta_best: np.ndarray,
    loglike_best: float,
    success: bool,
    message: str,
    logz: float = np.nan,
    logzerr: float = np.nan,
    samples_equal: Optional[np.ndarray] = None,
    loglikes: Optional[np.ndarray] = None,
) -> FitResult:
    k = len(params)
    n = data.n
    chi2 = -2.0 * loglike_best
    aic = 2.0 * k - 2.0 * loglike_best
    bic = k * np.log(n) - 2.0 * loglike_best
    dof = max(n - k, 1)
    red_chi2 = chi2 / dof
    return FitResult(
        model_name=model_name,
        theta_best=np.asarray(theta_best, dtype=float),
        loglike_best=float(loglike_best),
        chi2=float(chi2),
        aic=float(aic),
        bic=float(bic),
        red_chi2=float(red_chi2),
        success=success,
        message=message,
        param_names=[p.name for p in params],
        logz=logz,
        logzerr=logzerr,
        samples_equal=samples_equal,
        loglikes=loglikes,
    )

# Tables and figures

def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p

def write_priors_csv(outdir: Path, priors: Dict[str, List[Parameter]], model_names: Sequence[str]) -> None:
    rows = []
    for model in model_names:
        for p in priors[model]:
            rows.append({
                "model": model,
                "parameter": p.name,
                "lower": p.lower,
                "upper": p.upper,
                "label": p.label,
                "unit": p.unit,
            })
    pd.DataFrame(rows).to_csv(outdir / "priors.csv", index=False)

def save_result_npz(outdir: Path, result: FitResult) -> None:
    payload = {
        "theta_best": result.theta_best,
        "loglike_best": result.loglike_best,
        "chi2": result.chi2,
        "aic": result.aic,
        "bic": result.bic,
        "red_chi2": result.red_chi2,
        "param_names": np.array(result.param_names, dtype=object),
        "logz": result.logz,
        "logzerr": result.logzerr,
    }
    if result.samples_equal is not None:
        payload["samples_equal"] = result.samples_equal
    if result.loglikes is not None:
        payload["loglikes"] = result.loglikes
    np.savez(outdir / f"results_{result.model_name}.npz", **payload)

def param_table_rows(results: Sequence[FitResult]) -> pd.DataFrame:
    rows = []
    for res in results:
        for name, val in zip(res.param_names, res.theta_best):
            rows.append({"model": res.model_name, "parameter": name, "bestfit": val})
    return pd.DataFrame(rows)

def metrics_table(results: Sequence[FitResult]) -> pd.DataFrame:
    rows = []
    best_bic = min(r.bic for r in results)
    best_aic = min(r.aic for r in results)
    finite_logz = [r.logz for r in results if np.isfinite(r.logz)]
    best_logz = max(finite_logz) if finite_logz else np.nan
    for res in sorted(results, key=lambda x: x.bic):
        rows.append({
            "model": res.model_name,
            "n_params": len(res.param_names),
            "logL_max": res.loglike_best,
            "chi2": res.chi2,
            "red_chi2": res.red_chi2,
            "AIC": res.aic,
            "delta_AIC": res.aic - best_aic,
            "BIC": res.bic,
            "delta_BIC": res.bic - best_bic,
            "logZ": res.logz,
            "logZerr": res.logzerr,
            "delta_logZ": res.logz - best_logz if np.isfinite(res.logz) and np.isfinite(best_logz) else np.nan,
            "success": res.success,
        })
    return pd.DataFrame(rows)

MODEL_LABELS = {
    "exp_newtonian": r"Exponential reference ($s=1$)",
    "exp_fractional": r"Exponential fractional",
    "sersic": "Sérsic--PS",
    "einasto": r"Einasto",
}

def set_publication_style() -> None:
    """Configuração global compatível com figuras de artigo, sem exigir TeX."""
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "STIXGeneral", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 14,
        "axes.labelsize": 17,
        "axes.titlesize": 15,
        "xtick.labelsize": 13,
        "ytick.labelsize": 13,
        "legend.fontsize": 13,
        "axes.linewidth": 1.25,
        "lines.linewidth": 2.0,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.top": True,
        "ytick.right": True,
        "xtick.major.size": 7,
        "ytick.major.size": 7,
        "xtick.minor.size": 4,
        "ytick.minor.size": 4,
        "xtick.major.width": 1.15,
        "ytick.major.width": 1.15,
        "xtick.minor.width": 0.9,
        "ytick.minor.width": 0.9,
        "legend.frameon": False,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.04,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })

def _style_axis(ax: plt.Axes) -> None:
    ax.tick_params(which="both", direction="in", top=True, right=True)
    ax.minorticks_on()
    for spine in ax.spines.values():
        spine.set_linewidth(1.25)

def maybe_plot_corner(outdir: Path, result: FitResult, priors: Sequence[Parameter]) -> None:
    if result.samples_equal is None:
        return
    try:
        import corner  # type: ignore
    except Exception as exc:
        print(f"[aviso] corner indisponível ({exc}); corner plot não gerado.", flush=True)
        return

    samples = np.asarray(result.samples_equal, dtype=float)
    samples = samples[np.all(np.isfinite(samples), axis=1)]
    if samples.shape[0] < 50:
        return
    if samples.shape[0] > 50000:
        rng = np.random.default_rng(1234)
        samples = samples[rng.choice(samples.shape[0], 50000, replace=False)]

    labels = [p.label for p in priors]
    fig = corner.corner(
        samples,
        labels=labels,
        truths=result.theta_best,
        bins=35,
        smooth=0.8,
        smooth1d=0.8,
        quantiles=[0.16, 0.50, 0.84],
        show_titles=True,
        title_fmt=".3f",
        plot_datapoints=False,
        plot_density=True,
        fill_contours=True,
        levels=(0.393, 0.865, 0.989),
        label_kwargs={"fontsize": 15},
        title_kwargs={"fontsize": 12},
        hist_kwargs={"linewidth": 1.5},
        contour_kwargs={"linewidths": 1.15},
        truth_color="crimson",
        max_n_ticks=4,
    )
    for ax in fig.get_axes():
        ax.tick_params(which="both", direction="in", top=True, right=True, labelsize=10)
        for spine in ax.spines.values():
            spine.set_linewidth(1.0)
    fig.savefig(outdir / f"corner_{result.model_name}.png", dpi=300)
    fig.savefig(outdir / f"corner_{result.model_name}.pdf")
    plt.close(fig)

def dataframe_to_markdown(df: pd.DataFrame) -> str:
    """Gera tabela Markdown sem depender do pacote opcional tabulate."""
    shown = df.copy()
    for col in shown.columns:
        if pd.api.types.is_float_dtype(shown[col]):
            shown[col] = shown[col].map(lambda x: "nan" if not np.isfinite(x) else f"{x:.6g}")
        else:
            shown[col] = shown[col].astype(str)
    headers = [str(c) for c in shown.columns]
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in shown.itertuples(index=False, name=None):
        lines.append("| " + " | ".join(str(v) for v in row) + " |")
    return "\n".join(lines)

def posterior_rows(case: RobustnessCase, result: FitResult) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    samples = result.samples_equal
    for j, name in enumerate(result.param_names):
        row: Dict[str, object] = {
            "case_id": case.case_id,
            "dataset": case.dataset_key,
            "disk_mode": case.disk_mode,
            "category": case.category,
            "frac_nk": case.frac_nk,
            "model": result.model_name,
            "parameter": name,
            "bestfit": float(result.theta_best[j]),
            "q025": np.nan, "q16": np.nan, "q50": np.nan, "q84": np.nan, "q975": np.nan,
            "p_gt_1": np.nan,
        }
        if samples is not None and samples.ndim == 2 and samples.shape[1] > j and samples.shape[0] > 10:
            vals = np.asarray(samples[:, j], dtype=float)
            vals = vals[np.isfinite(vals)]
            if vals.size:
                q = np.quantile(vals, [0.025, 0.16, 0.50, 0.84, 0.975])
                row.update({"q025":q[0], "q16":q[1], "q50":q[2], "q84":q[3], "q975":q[4]})
                if name == "s":
                    row["p_gt_1"] = float(np.mean(vals > 1.0))
        rows.append(row)
    return rows

def plot_case_rotation_grid(
    outdir: Path,
    data: DataSet,
    results: Sequence[FitResult],
    frac_kernel: FractionalKernel,
    disk_mode: str,
) -> None:
    preferred = ["exp_newtonian", "exp_fractional", "sersic", "einasto"]
    by = {r.model_name: r for r in results}
    ordered = [by[m] for m in preferred if m in by]
    if not ordered:
        return
    r_plot = np.geomspace(max(1e-3, data.radius_kpc.min()*0.75), data.radius_kpc.max()*1.1, 600)
    ncols = len(ordered)
    fig, axes = plt.subplots(2, ncols, figsize=(5.0*ncols, 8.0), sharex="col")
    if ncols == 1:
        axes = np.asarray(axes).reshape(2, 1)
    for j, res in enumerate(ordered):
        ax, axr = axes[0,j], axes[1,j]
        cp = v_components(res.model_name, r_plot, res.theta_best, frac_kernel, disk_mode)
        cd = v_components(res.model_name, data.radius_kpc, res.theta_best, frac_kernel, disk_mode)
        residuals = (data.velocity_kms-cd["total"])/data.sigma_kms
        ax.errorbar(data.radius_kpc, data.velocity_kms, yerr=data.sigma_kms, fmt="o", ms=4, mfc="white", mec="black", ecolor="0.4", capsize=1.5)
        ax.plot(r_plot, cp["total"], label="Total")
        ax.plot(r_plot, cp["bulge"], ls="--", label="Bulge")
        ax.plot(r_plot, cp["disk"], ls="-.", label="Disk")
        ax.plot(r_plot, cp["halo"], ls=":", label="NFW halo")
        ax.set_xscale("log")
        ax.set_title(MODEL_LABELS.get(res.model_name,res.model_name))
        if j==0:
            ax.set_ylabel(r"$v_c\,[\mathrm{km\,s^{-1}}]$")
            ax.legend()
        axr.axhline(0.0, lw=1.0)
        axr.errorbar(data.radius_kpc, residuals, yerr=np.ones_like(residuals), fmt="o", ms=3.7, mfc="white", mec="black", ecolor="0.5", capsize=1.2)
        axr.set_xscale("log")
        axr.set_xlabel(r"$R\,[\mathrm{kpc}]$")
        if j==0:
            axr.set_ylabel(r"$(v_{obs}-v_{mod})/\sigma$")
        _style_axis(ax); _style_axis(axr)
    fig.suptitle(f"disk={disk_mode}; N={data.n}; R={data.radius_kpc.min():.3g}--{data.radius_kpc.max():.3g} kpc")
    fig.tight_layout()
    fig.savefig(outdir / "rotation_curves_residuals.png", dpi=220)
    fig.savefig(outdir / "rotation_curves_residuals.pdf")
    plt.close(fig)

def write_case_report(
    outdir: Path,
    case: RobustnessCase,
    loaded: LoadedDataset,
    data: DataSet,
    results: Sequence[FitResult],
    args: argparse.Namespace,
) -> None:
    metrics = metrics_table(results)
    lines = [
        f"# Caso {case.case_id}\n\n",
        f"- Dataset: **{loaded.spec.label}** (`{case.dataset_key}`)\n",
        f"- Fonte: {loaded.spec.source}\n",
        f"- Arquivo: `{loaded.path}`\n",
        f"- Disco: **{case.disk_mode}**\n",
        f"- N: {data.n}\n",
        f"- Faixa: {data.radius_kpc.min():.6g}--{data.radius_kpc.max():.6g} kpc\n",
        f"- Nq: {case.frac_nk}\n",
        f"- Categoria: {case.category}\n",
        f"- Nota: {case.note}\n\n",
        "## Métricas\n\n", dataframe_to_markdown(metrics), "\n",
    ]
    if case.disk_mode == "fixed":
        lines += [f"\nDisco fixo: Md={DISK_MASS_MSUN:.6g} Msun; Rd={DISK_SCALE_KPC:.6g} kpc.\n"]
    else:
        lines += [
            "\nO disco foi marginalizado simultaneamente com bojo e halo. ",
            f"Priors: log10(Md/Msun)=[{args.disk_logmass_min},{args.disk_logmass_max}], ",
            f"log10(Rd/kpc)=[{args.disk_logrd_min},{args.disk_logrd_max}].\n",
        ]
    (outdir / "report.md").write_text("".join(lines), encoding="utf-8")

def load_fit_result_npz(path: Path, model_name: str) -> FitResult:
    """Reconstrói um FitResult salvo por save_result_npz."""
    z = np.load(path, allow_pickle=True)
    names = [str(x) for x in z["param_names"].tolist()]
    samples = np.asarray(z["samples_equal"], dtype=float) if "samples_equal" in z.files else None
    loglikes = np.asarray(z["loglikes"], dtype=float) if "loglikes" in z.files else None
    return FitResult(
        model_name=model_name,
        theta_best=np.asarray(z["theta_best"], dtype=float),
        loglike_best=float(z["loglike_best"]),
        chi2=float(z["chi2"]),
        aic=float(z["aic"]),
        bic=float(z["bic"]),
        red_chi2=float(z["red_chi2"]),
        success=True,
        message="loaded from npz",
        param_names=names,
        logz=float(z["logz"]),
        logzerr=float(z["logzerr"]),
        samples_equal=samples,
        loglikes=loglikes,
    )

def plot_baseline_multirc_6x4(run_root: Path, args: argparse.Namespace, disk_mode: str) -> None:
    """Figura multi-RC compatível com a execução original, separada por hipótese de disco."""
    preferred = [m for m in ["exp_newtonian", "exp_fractional", "sersic", "einasto"] if m in args.models]
    if not preferred:
        return
    dataset_keys = [k for k in ["sofue2020", "sofue2025", "huang2016"] if k in args.datasets]
    if not dataset_keys:
        return

    nrows = 2 * len(dataset_keys)
    fig, axes = plt.subplots(nrows, len(preferred), figsize=(5.0 * len(preferred), 3.7 * nrows), sharex=False)
    if nrows == 1 and len(preferred) == 1:
        axes = np.array([[axes]])
    elif len(preferred) == 1:
        axes = np.asarray(axes).reshape(nrows, 1)
    elif nrows == 1:
        axes = np.asarray(axes).reshape(1, len(preferred))

    markers = {"sofue2020": "o", "sofue2025": "s", "huang2016": "^"}
    plotted_any = False
    for di, key in enumerate(dataset_keys):
        try:
            loaded = load_named_dataset(args, key)
        except Exception as exc:
            mpi_print(
                f"[aviso] figura multi-RC ({disk_mode}): não foi possível carregar {key}: {exc}",
                root_only=True,
            )
            continue
        data = loaded.data
        case_dir = run_root / f"disk_{disk_mode}" / f"baseline_{key}"
        if not case_dir.exists():
            continue
        kernel = FractionalKernel(
            nk=args.frac_nk,
            qmin=args.frac_qmin,
            qmax=args.frac_qmax,
            ell0_kpc=args.ell0_kpc,
            asymptotic_transition_width=args.frac_asym_width,
        )
        r_plot = np.geomspace(max(1e-3, data.radius_kpc.min() * 0.75), data.radius_kpc.max() * 1.1, 700)
        for j, model in enumerate(preferred):
            npz = case_dir / f"results_{model}.npz"
            if not npz.exists():
                continue
            plotted_any = True
            res = load_fit_result_npz(npz, model)
            cp = v_components(model, r_plot, res.theta_best, kernel, disk_mode)
            cd = v_components(model, data.radius_kpc, res.theta_best, kernel, disk_mode)
            residuals = (data.velocity_kms - cd["total"]) / data.sigma_kms
            ax = axes[2 * di, j]
            axr = axes[2 * di + 1, j]
            ax.errorbar(
                data.radius_kpc, data.velocity_kms, yerr=data.sigma_kms,
                fmt=markers.get(key, "o"), ms=4.0, mfc="white", mec="black",
                ecolor="0.45", capsize=1.4, label=loaded.spec.label,
            )
            ax.plot(r_plot, cp["total"], label="Total")
            ax.plot(r_plot, cp["bulge"], ls="--", label="Bulge")
            ax.plot(r_plot, cp["disk"], ls="-.", label="Disk")
            ax.plot(r_plot, cp["halo"], ls=":", label="NFW halo")
            ax.set_xscale("log")
            if di == 0:
                ax.set_title(MODEL_LABELS.get(model, model))
            if j == 0:
                ax.set_ylabel(f"{loaded.spec.label}\n" + r"$v_c\,[\mathrm{km\,s^{-1}}]$")
            if di == 0 and j == 0:
                ax.legend(fontsize=10)
            _style_axis(ax)

            axr.axhline(0.0, lw=1.0)
            axr.errorbar(
                data.radius_kpc, residuals, yerr=np.ones_like(residuals),
                fmt=markers.get(key, "o"), ms=3.5, mfc="white", mec="black",
                ecolor="0.55", capsize=1.0,
            )
            axr.set_xscale("log")
            axr.set_xlabel(r"$R\,[\mathrm{kpc}]$")
            if j == 0:
                axr.set_ylabel(r"$(v_{obs}-v_{mod})/\sigma$")
            _style_axis(axr)

    if not plotted_any:
        plt.close(fig)
        return
    fig.suptitle(f"Milky Way rotation curves — disk {disk_mode}", y=0.997)
    fig.tight_layout(rect=(0, 0, 1, 0.988))
    out = ensure_dir(run_root / f"disk_{disk_mode}")
    fig.savefig(out / "rotation_curves_residuals_6x4.png", dpi=250)
    fig.savefig(out / "rotation_curves_residuals_6x4.pdf")
    plt.close(fig)

def write_disk_baseline_products(
    run_root: Path,
    args: argparse.Namespace,
    metrics: pd.DataFrame,
    params: pd.DataFrame,
    post: pd.DataFrame,
    disk_mode: str,
) -> None:
    """Replica os produtos multiRC principais para fixed e marginalized."""
    out = ensure_dir(run_root / f"disk_{disk_mode}")
    if not metrics.empty:
        m = metrics[(metrics["disk_mode"] == disk_mode) & (metrics["category"] == "baseline")].copy()
        m.to_csv(out / "summary_metrics_all_datasets.csv", index=False)
    else:
        m = pd.DataFrame()
    if not params.empty:
        p = params[(params["disk_mode"] == disk_mode) & (params["category"] == "baseline")].copy()
        p.to_csv(out / "bestfit_parameters_all_datasets.csv", index=False)
    if not post.empty:
        pp = post[(post["disk_mode"] == disk_mode) & (post["category"] == "baseline")].copy()
        pp.to_csv(out / "posterior_summary_all_datasets.csv", index=False)

    if not m.empty:
        try:
            plot_baseline_multirc_6x4(run_root, args, disk_mode)
        except Exception as exc:
            mpi_print(
                f"[aviso] agregação: falha não fatal ao gerar figura multi-RC ({disk_mode}): {exc}",
                root_only=True,
            )
    lines = [
        f"# Relatório automático — três curvas de rotação — disco {disk_mode}\n\n",
        f"Data de geração: {datetime.now().isoformat(timespec='seconds')}\n\n",
        f"Kernel de produção: Nq={args.frac_nk}.\n\n",
    ]
    if disk_mode == "fixed":
        lines.append(f"Disco fixo: Md={DISK_MASS_MSUN:.6g} Msun; Rd={DISK_SCALE_KPC:.6g} kpc.\n\n")
    else:
        lines.append(
            "Disco marginalizado simultaneamente com bojo e halo: "
            f"log10(Md/Msun)=[{args.disk_logmass_min},{args.disk_logmass_max}], "
            f"log10(Rd/kpc)=[{args.disk_logrd_min},{args.disk_logrd_max}].\n\n"
        )
    if not m.empty:
        lines += ["## Métricas baseline\n\n", dataframe_to_markdown(m), "\n"]
    (out / "report_multiRC.md").write_text("".join(lines), encoding="utf-8")

def _read_case_csvs(run_root: Path, filename: str) -> pd.DataFrame:
    frames=[]
    for p in sorted(run_root.glob(f"disk_*/*/{filename}")):
        try: frames.append(pd.read_csv(p))
        except Exception: pass
    return pd.concat(frames,ignore_index=True) if frames else pd.DataFrame()

def _fractional_pair_summary(metrics: pd.DataFrame, post: pd.DataFrame, case_filter: pd.Series) -> Dict[str, float]:
    sub=metrics[case_filter]
    frac=sub[sub.model=="exp_fractional"]
    ref=sub[sub.model=="exp_newtonian"]
    out={k:np.nan for k in ["chi2_frac","chi2_ref","delta_chi2_ref_minus_frac","bic_frac","bic_ref","delta_bic_ref_minus_frac","logz_frac","logz_ref","delta_logz_frac_minus_ref","s_best","s_q025","s_q16","s_q50","s_q84","s_q975","p_s_gt_1"]}
    if len(frac):
        f=frac.iloc[0]; out.update({"chi2_frac":f.chi2,"bic_frac":f.BIC,"logz_frac":f.logZ})
    if len(ref):
        r=ref.iloc[0]; out.update({"chi2_ref":r.chi2,"bic_ref":r.BIC,"logz_ref":r.logZ})
    if len(frac) and len(ref):
        out["delta_chi2_ref_minus_frac"]=float(ref.iloc[0].chi2-frac.iloc[0].chi2)
        out["delta_bic_ref_minus_frac"]=float(ref.iloc[0].BIC-frac.iloc[0].BIC)
        if np.isfinite(frac.iloc[0].logZ) and np.isfinite(ref.iloc[0].logZ):
            out["delta_logz_frac_minus_ref"]=float(frac.iloc[0].logZ-ref.iloc[0].logZ)
    if not post.empty:
        ids=set(sub.case_id.astype(str))
        pp=post[(post.case_id.astype(str).isin(ids)) & (post.model=="exp_fractional") & (post.parameter=="s")]
        if len(pp):
            q=pp.iloc[0]; out.update({"s_best":q.bestfit,"s_q025":q.q025,"s_q16":q.q16,"s_q50":q.q50,"s_q84":q.q84,"s_q975":q.q975,"p_s_gt_1":q.p_gt_1})
    return out

def aggregate_results(run_root: Path, args: argparse.Namespace, cases: Sequence[RobustnessCase], failures: Sequence[Dict[str,object]]) -> None:
    metrics=_read_case_csvs(run_root,"summary_metrics.csv")
    params=_read_case_csvs(run_root,"bestfit_parameters.csv")
    post=_read_case_csvs(run_root,"posterior_summary.csv")
    metrics.to_csv(run_root/"summary_metrics_all_cases.csv",index=False)
    params.to_csv(run_root/"bestfit_parameters_all_cases.csv",index=False)
    post.to_csv(run_root/"posterior_summary_all_cases.csv",index=False)

    for _disk_mode in (["fixed", "marginalized"] if args.disk_mode == "both" else [args.disk_mode]):
        write_disk_baseline_products(run_root, args, metrics, params, post, _disk_mode)

    manifest_rows=[]
    for p in sorted(run_root.glob("disk_*/*/case_manifest.json")):
        try: manifest_rows.append(json.loads(p.read_text(encoding="utf-8")))
        except Exception: pass
    pd.DataFrame(manifest_rows).to_csv(run_root/"run_manifest.csv",index=False)
    if failures:
        pd.DataFrame(failures).to_csv(run_root/"failures.csv",index=False)

    radial_rows=[]
    for disk in ("fixed","marginalized"):
        if metrics.empty: continue
        ids=[]
        base=f"baseline_sofue2025"
        if ((metrics.case_id==base)&(metrics.disk_mode==disk)).any(): ids.append((base,np.nan))
        for rmin in SOFUE2025_RMIN_CUTS:
            cid=f"sofue2025_rmin_{str(rmin).replace('.', 'p')}"
            if ((metrics.case_id==cid)&(metrics.disk_mode==disk)).any(): ids.append((cid,rmin))
        for cid,rmin in ids:
            submask=(metrics.case_id==cid)&(metrics.disk_mode==disk)
            sub=metrics[submask]
            actual=float(sub.rmin_actual.iloc[0]) if len(sub) else np.nan
            row={"disk_mode":disk,"case_id":cid,"rmin_requested":rmin,"rmin_actual":actual}
            row.update(_fractional_pair_summary(metrics,post,submask))
            radial_rows.append(row)
    radial=pd.DataFrame(radial_rows)
    radial.to_csv(run_root/"radial_cut_summary.csv",index=False)

    common_rows=[]
    for disk in ("fixed","marginalized"):
        for key,cid in [("sofue2020","sofue2020_common_0p10_25p57"),("sofue2025","sofue2025_rmin_0p1")]:
            mask=(metrics.case_id==cid)&(metrics.disk_mode==disk) if not metrics.empty else pd.Series(dtype=bool)
            if len(metrics) and mask.any():
                row={"disk_mode":disk,"dataset":key,"case_id":cid,"rmin":COMMON_RMIN_KPC,"rmax":COMMON_RMAX_KPC}
                row.update(_fractional_pair_summary(metrics,post,mask)); common_rows.append(row)
    pd.DataFrame(common_rows).to_csv(run_root/"common_range_summary.csv",index=False)

    kernel_rows=[]
    for disk in ("fixed","marginalized"):
        for nk,cid in [(args.kernel_check_nk,f"kernelcheck_sofue2025_nk{args.kernel_check_nk}"),(args.frac_nk,"baseline_sofue2025")]:
            mask=(metrics.case_id==cid)&(metrics.disk_mode==disk) if not metrics.empty else pd.Series(dtype=bool)
            if len(metrics) and mask.any():
                row={"disk_mode":disk,"frac_nk":nk,"case_id":cid}
                row.update(_fractional_pair_summary(metrics,post,mask)); kernel_rows.append(row)
    pd.DataFrame(kernel_rows).to_csv(run_root/"kernel_resolution_fit_comparison.csv",index=False)

    disk_rows=[]
    if not metrics.empty:
        for cid in sorted(set(metrics.case_id.astype(str))):
            if cid.startswith("kernelcheck_"): continue
            for model in sorted(set(metrics[metrics.case_id==cid].model.astype(str))):
                f=metrics[(metrics.case_id==cid)&(metrics.disk_mode=="fixed")&(metrics.model==model)]
                m=metrics[(metrics.case_id==cid)&(metrics.disk_mode=="marginalized")&(metrics.model==model)]
                if len(f) and len(m):
                    fr, mr=f.iloc[0],m.iloc[0]
                    disk_rows.append({
                        "case_id":cid,"dataset":fr.dataset,"model":model,
                        "chi2_fixed":fr.chi2,"chi2_marginalized":mr.chi2,
                        "delta_chi2_fixed_minus_marg":fr.chi2-mr.chi2,
                        "BIC_fixed":fr.BIC,"BIC_marginalized":mr.BIC,
                        "delta_BIC_fixed_minus_marg":fr.BIC-mr.BIC,
                        "logZ_fixed":fr.logZ,"logZ_marginalized":mr.logZ,
                        "delta_logZ_marg_minus_fixed":mr.logZ-fr.logZ if np.isfinite(fr.logZ) and np.isfinite(mr.logZ) else np.nan,
                    })
    pd.DataFrame(disk_rows).to_csv(run_root/"disk_fixed_vs_marginalized.csv",index=False)

    disk_post=[]
    if not post.empty:
        for (cid,model),grp in post[(post.disk_mode=="marginalized") & (post.parameter.isin(["log10_Md","log10_Rd"]))].groupby(["case_id","model"]):
            row={"case_id":cid,"model":model}
            for _,rr in grp.iterrows():
                stem="Md_Msun" if rr.parameter=="log10_Md" else "Rd_kpc"
                for qname in ["bestfit","q025","q16","q50","q84","q975"]:
                    val=rr[qname]
                    row[f"{stem}_{qname}"]=10.0**val if np.isfinite(val) else np.nan
            disk_post.append(row)
    pd.DataFrame(disk_post).to_csv(run_root/"marginalized_disk_parameters.csv",index=False)

    if not radial.empty:
        fig,ax=plt.subplots(figsize=(7.4,5.3))
        plotted_s = False
        for disk,grp in radial.groupby("disk_mode"):
            g=grp.sort_values("rmin_actual")
            x=g.rmin_actual.to_numpy(dtype=float)
            y=np.where(np.isfinite(g.s_q50.to_numpy(dtype=float)),
                       g.s_q50.to_numpy(dtype=float),
                       g.s_best.to_numpy(dtype=float))
            good=np.isfinite(x) & (x > 0.0) & np.isfinite(y)
            if not np.any(good):
                continue
            ax.plot(x[good],y[good],marker="o",label=disk)
            plotted_s = True
            lo=g.s_q16.to_numpy(dtype=float); hi=g.s_q84.to_numpy(dtype=float)
            good_band=good & np.isfinite(lo) & np.isfinite(hi)
            if np.any(good_band):
                ax.fill_between(x[good_band],lo[good_band],hi[good_band],alpha=0.18)
        if plotted_s:
            ax.axhline(1.0,ls="--",lw=1.1)
            ax.set_xscale("log"); ax.set_xlabel(r"$R_{\min}$ [kpc]"); ax.set_ylabel(r"$s$")
            ax.legend(title="Disk"); _style_axis(ax); fig.tight_layout()
            fig.savefig(run_root/"radial_cut_s_vs_rmin.png",dpi=250); fig.savefig(run_root/"radial_cut_s_vs_rmin.pdf")
        plt.close(fig)

        finite_bayes = np.isfinite(radial.delta_logz_frac_minus_ref.to_numpy(dtype=float))
        if np.any(finite_bayes):
            fig,ax=plt.subplots(figsize=(7.4,5.3))
            plotted = False
            for disk,grp in radial.groupby("disk_mode"):
                g=grp.sort_values("rmin_actual")
                good=np.isfinite(g.delta_logz_frac_minus_ref.to_numpy(dtype=float)) & (g.rmin_actual.to_numpy(dtype=float) > 0)
                if np.any(good):
                    ax.plot(g.rmin_actual.to_numpy(dtype=float)[good],g.delta_logz_frac_minus_ref.to_numpy(dtype=float)[good],marker="o",label=disk)
                    plotted = True
            if plotted:
                ax.axhline(0.0,ls="--",lw=1.1); ax.set_xscale("log")
                ax.set_xlabel(r"$R_{\min}$ [kpc]"); ax.set_ylabel(r"$\Delta\ln Z_{F-ref}$")
                ax.legend(title="Disk"); _style_axis(ax); fig.tight_layout()
                fig.savefig(run_root/"radial_cut_delta_logz_vs_rmin.png",dpi=250); fig.savefig(run_root/"radial_cut_delta_logz_vs_rmin.pdf")
            plt.close(fig)

    report=[
        "# Relatório de robustez — modelo fracionário da Via Láctea\n\n",
        f"MPI: available={MPI_AVAILABLE}, ranks={MPI_SIZE}.\n\n",
        f"Kernel de produção: Nq={args.frac_nk}; controle: Nq={args.kernel_check_nk}.\n\n",
        "## Testes executados\n",
        "1. Sofue & Kohno 2025 com cortes Rmin=0.05, 0.10, 0.20, 0.50 e 1.00 kpc.\n",
        "2. Sofue 2020 e Sofue & Kohno 2025 em 0.10--25.57 kpc.\n",
        "3. Produção com quadratura fracionária de alta resolução e controle de resolução.\n",
        "4. Cada teste repetido com disco fixo e com Md/Rd marginalizados.\n\n",
        "## Arquivos principais\n",
        "- `radial_cut_summary.csv`\n- `common_range_summary.csv`\n- `kernel_resolution_fit_comparison.csv`\n",
        "- `disk_fixed_vs_marginalized.csv`\n- `marginalized_disk_parameters.csv`\n",
        "- `summary_metrics_all_cases.csv`\n- `posterior_summary_all_cases.csv`\n\n",
        "A evidência entre disco fixo e marginalizado depende explicitamente dos priors de Md e Rd; os limites usados estão registrados em `run_config.json`.\n",
    ]
    (run_root/"report_robustness.md").write_text("".join(report),encoding="utf-8")

# Robustness runs

def run_internal_tests(frac_kernel: FractionalKernel) -> None:
    r = np.geomspace(0.02, 200.0, 160)
    mass = 1.7e10
    eps = 2.0
    v2n = v2_bulge_exponential_newtonian(r, mass, eps)
    if not np.all(np.isfinite(v2n)) or np.any(v2n < 0.0):
        raise RuntimeError("Teste falhou: bojo exponencial Newtoniano inválido.")

    frac_large = mass_fraction_exponential_sphere(np.array([200.0 / eps]), eps)[0]
    if frac_large < 0.999999:
        raise RuntimeError("Teste falhou: massa exponencial não converge para M_total.")

    v2f = frac_kernel.v2_fractional_exponential(r, mass, eps, 1.0)
    rel = np.nanmax(np.abs(v2f - v2n) / np.maximum(v2n, 1e-30))
    if rel > 1e-12:
        raise RuntimeError(f"Teste falhou: limite s=1 não reproduz Newtoniano. rel={rel}")

    raw1 = frac_kernel.raw_shape_newtonian_exact(r, eps)
    expected = 0.5 * np.pi * eps**(-4.0) * mass_fraction_exponential_sphere(r, eps) / r
    rel_raw1 = np.nanmax(np.abs(raw1 - expected) / np.maximum(np.abs(expected), 1e-30))
    if rel_raw1 > 1e-12:
        raise RuntimeError(f"Teste falhou: raw Newtoniano analítico inconsistente. rel={rel_raw1}")

    if frac_kernel.nk >= 256:
        x = np.geomspace(1e-3, 3e3, 240)
        for s_test in np.linspace(0.3, 1.3, 11):
            integral = frac_kernel.dimensionless_integral(x, float(s_test))
            if np.any(~np.isfinite(integral)) or np.any(integral >= 0.0):
                raise RuntimeError(f"Teste falhou: integral híbrida inválida para s={s_test:.3f}.")

def execute_case(
    case: RobustnessCase,
    args: argparse.Namespace,
    run_root: Path,
    loaded_cache: Dict[str, LoadedDataset],
) -> Dict[str, object]:
    if case.dataset_key not in loaded_cache:
        loaded_cache[case.dataset_key] = load_named_dataset(args, case.dataset_key)
    loaded = loaded_cache[case.dataset_key]
    data = subset_dataset(loaded.data, case.rmin_kpc, case.rmax_kpc)
    outdir = ensure_dir(run_root / f"disk_{case.disk_mode}" / case.case_id)
    kernel = FractionalKernel(
        nk=case.frac_nk, qmin=args.frac_qmin, qmax=args.frac_qmax,
        ell0_kpc=args.ell0_kpc, asymptotic_transition_width=args.frac_asym_width,
    )
    if not args.skip_tests:
        run_internal_tests(kernel)

    priors = priors_for_disk_mode(args, case.disk_mode)
    write_priors_csv(outdir, priors, case.models)
    results: List[FitResult] = []
    by_name: Dict[str, FitResult] = {}
    local_workers = int(args.workers)
    if MPI_SIZE > 1 and local_workers > 1 and not args.allow_mpi_local_workers:
        local_workers = 1

    for model_index, model in enumerate(case.models):
        params = priors[model]
        initial_guesses: List[np.ndarray] = []
        if model == "exp_fractional" and "exp_newtonian" in by_name:
            initial_guesses.append(map_warm_start(by_name["exp_newtonian"], params, s_default=1.0))
        seed = stable_seed(args.seed, case.case_id, case.disk_mode, str(case.frac_nk), model)
        mpi_print(f"{case.case_id} [{case.disk_mode}] -> {model} (N={data.n}, Nq={case.frac_nk})")
        theta, ll, success, message = fit_de(
            model, data, params, kernel, case.disk_mode,
            args.de_maxiter, args.de_popsize, seed, workers=local_workers,
            initial_guesses=initial_guesses,
        )
        logz=logzerr=np.nan; samples_equal=loglikes=None
        optimizer_ll = float(ll)
        if args.sampler == "dynesty":
            logz, logzerr, samples_equal, loglikes, theta_nested, ll_nested = sample_dynesty(
                model, data, params, kernel, case.disk_mode,
                args.nlive, args.dlogz, stable_seed(seed, "dynesty"),
                print_progress=bool(args.dynesty_progress and MPI_SIZE == 1),
            )
            if np.isfinite(ll_nested):
                gap = optimizer_ll - ll_nested
                if gap > args.dynesty_maxlogl_gap:
                    mpi_print(f"[aviso] {case.case_id}/{model}: dynesty perdeu modo do DE por {gap:.3f} em logL; logZ rejeitado.")
                    logz=logzerr=np.nan; samples_equal=None
                if theta_nested is not None and ll_nested > ll:
                    tp, lp, okp, mp = polish(model, data, params, kernel, case.disk_mode, theta_nested)
                    if okp and lp >= ll_nested:
                        cand_t, cand_ll, cand_msg = tp, lp, f"dynesty max + {mp}; {message}"
                    else:
                        cand_t, cand_ll, cand_msg = theta_nested, ll_nested, f"dynesty max bruto; {message}"
                    if cand_ll > ll:
                        theta, ll, success, message = np.asarray(cand_t), float(cand_ll), True, cand_msg
            tol = max(0.1, 5.0*float(logzerr) if np.isfinite(logzerr) else 0.1)
            if np.isfinite(logz) and logz > ll + tol:
                mpi_print(f"[aviso] {case.case_id}/{model}: logZ > logLmax; logZ rejeitado.")
                logz=logzerr=np.nan; samples_equal=None

        res = compute_fit_result(
            model, data, params, theta, ll, success, message,
            logz=logz, logzerr=logzerr, samples_equal=samples_equal, loglikes=loglikes,
        )
        results.append(res); by_name[model]=res; save_result_npz(outdir,res)

    metrics = metrics_table(results)
    params_df = param_table_rows(results)
    post_df = pd.DataFrame([row for res in results for row in posterior_rows(case,res)])
    metrics.insert(0,"case_id",case.case_id); metrics.insert(1,"dataset",case.dataset_key)
    metrics.insert(2,"disk_mode",case.disk_mode); metrics.insert(3,"category",case.category)
    metrics.insert(4,"frac_nk",case.frac_nk); metrics.insert(5,"rmin_actual",data.radius_kpc.min())
    metrics.insert(6,"rmax_actual",data.radius_kpc.max()); metrics.insert(7,"n_data",data.n)
    params_df.insert(0,"case_id",case.case_id); params_df.insert(1,"dataset",case.dataset_key)
    params_df.insert(2,"disk_mode",case.disk_mode); params_df.insert(3,"category",case.category)
    params_df.insert(4,"frac_nk",case.frac_nk)
    metrics.to_csv(outdir/"summary_metrics.csv",index=False)
    params_df.to_csv(outdir/"bestfit_parameters.csv",index=False)
    post_df.to_csv(outdir/"posterior_summary.csv",index=False)

    if not args.skip_case_plots:
        plot_case_rotation_grid(outdir,data,results,kernel,case.disk_mode)
    if args.corner_plots != "none":
        for res in results:
            if args.corner_plots == "all" or res.model_name == "exp_fractional":
                maybe_plot_corner(outdir,res,priors[res.model_name])
    write_case_report(outdir,case,loaded,data,results,args)
    manifest = {
        "case_id":case.case_id,"dataset":case.dataset_key,"disk_mode":case.disk_mode,
        "category":case.category,"frac_nk":case.frac_nk,"rmin_requested":case.rmin_kpc,
        "rmax_requested":case.rmax_kpc,"rmin_actual":float(data.radius_kpc.min()),
        "rmax_actual":float(data.radius_kpc.max()),"n_data":int(data.n),"models":list(case.models),
        "note":case.note,"status":"ok",
    }
    (outdir/"case_manifest.json").write_text(json.dumps(manifest,indent=2,ensure_ascii=False)+"\n",encoding="utf-8")
    return manifest

def build_cases(args: argparse.Namespace) -> List[RobustnessCase]:
    models = tuple(args.models)
    disk_modes = ("fixed","marginalized") if args.disk_mode == "both" else (args.disk_mode,)
    cases: List[RobustnessCase] = []
    for disk in disk_modes:
        for key in args.datasets:
            cases.append(RobustnessCase(
                case_id=f"baseline_{key}", dataset_key=key, disk_mode=disk,
                rmin_kpc=None, rmax_kpc=None, frac_nk=args.frac_nk,
                category="baseline", models=models, note="Faixa radial completa do dataset.",
            ))
        if "sofue2025" in args.datasets:
            for rmin in SOFUE2025_RMIN_CUTS:
                tag = str(rmin).replace(".","p")
                rmax = COMMON_RMAX_KPC if abs(rmin-COMMON_RMIN_KPC)<1e-12 else None
                note = "Teste de sensibilidade ao corte radial interno."
                if abs(rmin-COMMON_RMIN_KPC)<1e-12:
                    note += " Este caso também representa Sofue 2025 na faixa comum 0.10--25.57 kpc."
                cases.append(RobustnessCase(
                    case_id=f"sofue2025_rmin_{tag}", dataset_key="sofue2025", disk_mode=disk,
                    rmin_kpc=rmin, rmax_kpc=rmax, frac_nk=args.frac_nk,
                    category="radial_cut", models=models, note=note,
                ))
        if "sofue2020" in args.datasets:
            cases.append(RobustnessCase(
                case_id="sofue2020_common_0p10_25p57", dataset_key="sofue2020", disk_mode=disk,
                rmin_kpc=COMMON_RMIN_KPC, rmax_kpc=COMMON_RMAX_KPC, frac_nk=args.frac_nk,
                category="common_range", models=models,
                note="Comparação Sofue 2020 vs Sofue 2025 exatamente na faixa 0.10--25.57 kpc.",
            ))
        if args.kernel_check_nk > 0 and args.kernel_check_nk != args.frac_nk and "sofue2025" in args.datasets:
            cases.append(RobustnessCase(
                case_id=f"kernelcheck_sofue2025_nk{args.kernel_check_nk}", dataset_key="sofue2025", disk_mode=disk,
                rmin_kpc=None, rmax_kpc=None, frac_nk=args.kernel_check_nk,
                category="kernel_check", models=("exp_newtonian","exp_fractional"),
                note=f"Controle de resolução do kernel contra a produção Nq={args.frac_nk}.",
            ))
    return cases

def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser=argparse.ArgumentParser(
        description="Robustez multi-curva: cortes radiais, faixa comum, Nq=1536, disco fixo/marginalizado e MPI."
    )
    parser.add_argument("--input-dir",default="input",help="Diretório com as tabelas das curvas.")
    parser.add_argument("--datasets",nargs="+",default=["sofue2020","sofue2025","huang2016"],choices=list(DATASET_SPECS))
    parser.add_argument("--data-sofue2020",default=None)
    parser.add_argument("--data-sofue2025",default=None)
    parser.add_argument("--data-huang2016",default=None)
    parser.add_argument("--cols-sofue2020",default=None,help="Índices R,V,sigma; default 0,1,3.")
    parser.add_argument("--cols-sofue2025",default=None,help="Índices R,V,sigma; default 0,1,3 para tabelas de 4 colunas (como no multiRC original), ou 0,1,2 para 3 colunas.")
    parser.add_argument("--cols-huang2016",default=None,help="Índices R,V,sigma; default 0,1,3 para tabelas de 4 colunas (como no multiRC original), ou 0,1,2 para 3 colunas.")
    for key in DATASET_SPECS:
        parser.add_argument(f"--radius-unit-{key}",choices=["auto","kpc","pc"],default="auto")

    parser.add_argument("--outdir",default="output")
    parser.add_argument("--disk-mode",choices=["fixed","marginalized","both"],default="both")
    parser.add_argument("--models",nargs="+",default=["exp_newtonian","exp_fractional","sersic","einasto"],choices=["exp_newtonian","exp_fractional","sersic","einasto"])
    parser.add_argument("--sampler",choices=["optimize","dynesty"],default="dynesty")
    parser.add_argument("--de-maxiter",type=int,default=180)
    parser.add_argument("--de-popsize",type=int,default=18)
    parser.add_argument("--workers",type=int,default=1,help="Workers locais por rank; com MPI, default efetivo=1 para evitar oversubscription.")
    parser.add_argument("--allow-mpi-local-workers",action="store_true",help="Permite workers>1 dentro de cada rank MPI (use com cautela).")
    parser.add_argument("--frac-nk",type=int,default=DEFAULT_FRAC_NK_PRODUCTION,help="Nós de produção; default 1536.")
    parser.add_argument("--kernel-check-nk",type=int,default=DEFAULT_KERNEL_CHECK_NK,help="Execução de controle do kernel, default 384; 0 desativa.")
    parser.add_argument("--frac-qmin",type=float,default=1e-6)
    parser.add_argument("--frac-qmax",type=float,default=100.0)
    parser.add_argument("--ell0-kpc",type=float,default=1.0)
    parser.add_argument("--frac-asym-width",type=float,default=6.0)

    parser.add_argument("--logmb-min",type=float,default=7.0); parser.add_argument("--logmb-max",type=float,default=11.0)
    parser.add_argument("--logepsilon-min",type=float,default=-1.3); parser.add_argument("--logepsilon-max",type=float,default=1.3)
    parser.add_argument("--s-min",type=float,default=0.3); parser.add_argument("--s-max",type=float,default=1.3)
    parser.add_argument("--logre-min",type=float,default=-1.3); parser.add_argument("--logre-max",type=float,default=1.0)
    parser.add_argument("--n-min",type=float,default=0.5); parser.add_argument("--n-max",type=float,default=6.0)
    parser.add_argument("--halo-logrho-min",type=float,default=-5.0); parser.add_argument("--halo-logrho-max",type=float,default=5.0)
    parser.add_argument("--halo-logrs-min",type=float,default=-1.0); parser.add_argument("--halo-logrs-max",type=float,default=3.5)

    parser.add_argument("--disk-logmass-min",type=float,default=10.30,help="Prior uniforme em log10(Md/Msun), mínimo.")
    parser.add_argument("--disk-logmass-max",type=float,default=10.95,help="Prior uniforme em log10(Md/Msun), máximo.")
    parser.add_argument("--disk-logrd-min",type=float,default=math.log10(2.0),help="Prior uniforme em log10(Rd/kpc), mínimo; default Rd=2 kpc.")
    parser.add_argument("--disk-logrd-max",type=float,default=math.log10(7.0),help="Prior uniforme em log10(Rd/kpc), máximo; default Rd=7 kpc.")

    parser.add_argument("--nlive",type=int,default=600); parser.add_argument("--dlogz",type=float,default=0.05)
    parser.add_argument("--dynesty-maxlogl-gap",type=float,default=2.0)
    parser.add_argument("--dynesty-progress",action="store_true")
    parser.add_argument("--seed",type=int,default=RNG_SEED)
    parser.add_argument("--skip-tests",action="store_true")
    parser.add_argument("--skip-case-plots",action="store_true")
    parser.add_argument("--corner-plots",choices=["none","fractional","all"],default="fractional")
    return parser.parse_args(argv)

def validate_args(args: argparse.Namespace) -> None:
    if args.frac_nk < 32: raise ValueError("--frac-nk deve ser >=32")
    if args.disk_logmass_min >= args.disk_logmass_max: raise ValueError("Prior de Md inválido")
    if args.disk_logrd_min >= args.disk_logrd_max: raise ValueError("Prior de Rd inválido")
    if not (args.disk_logmass_min <= math.log10(DISK_MASS_MSUN) <= args.disk_logmass_max):
        mpi_print("[aviso] o valor do disco fixo Md=6e10 Msun está fora do prior do caso marginalizado.",root_only=True)
    if not (args.disk_logrd_min <= math.log10(DISK_SCALE_KPC) <= args.disk_logrd_max):
        mpi_print("[aviso] o valor do disco fixo Rd=5.4 kpc está fora do prior do caso marginalizado.",root_only=True)

def main(argv: Optional[Sequence[str]] = None) -> int:
    args=parse_args(argv); validate_args(args)
    np.random.seed(stable_seed(args.seed,str(MPI_RANK)))
    set_publication_style()
    t0=time.time()

    if MPI_RANK==0:
        stamp=datetime.now().strftime("run_%Y%m%d_%H%M%S")
        root=ensure_dir(Path(args.outdir)/stamp)
        config=vars(args).copy(); config.update({"mpi_available":MPI_AVAILABLE,"mpi_size":MPI_SIZE,"production_frac_nk":args.frac_nk,"common_range_kpc":[COMMON_RMIN_KPC,COMMON_RMAX_KPC],"sofue2025_rmin_cuts_kpc":list(SOFUE2025_RMIN_CUTS)})
        (root/"run_config.json").write_text(json.dumps(config,indent=2,ensure_ascii=False,default=str)+"\n",encoding="utf-8")
    else:
        root=None
    root=Path(MPI_COMM.bcast(str(root) if root is not None else None,root=0))
    MPI_COMM.Barrier()

    cases=build_cases(args)
    if MPI_RANK==0:
        mpi_print("="*88,root_only=True)
        mpi_print("ROBUSTEZ MW v2 — cortes radiais + faixa comum + Nq alto + disco fixo/marginalizado + MPI",root_only=True)
        mpi_print(f"MPI ranks: {MPI_SIZE} | casos: {len(cases)} | Nq produção: {args.frac_nk}",root_only=True)
        mpi_print(f"Saída: {root}",root_only=True)
        if not MPI_AVAILABLE:
            mpi_print("[aviso] mpi4py não disponível: execução serial. Para paralelizar, instale mpi4py e use mpirun/mpiexec.",root_only=True)
        if MPI_SIZE>1 and args.workers>1 and not args.allow_mpi_local_workers:
            mpi_print("[aviso] --workers>1 foi desativado dentro dos ranks para evitar oversubscription.",root_only=True)
        mpi_print("="*88,root_only=True)

    if MPI_RANK == 0:
        preflight_errors=[]
        preflight_rows=[]
        for key in args.datasets:
            try:
                ld=load_named_dataset(args,key)
                preflight_rows.append({
                    "dataset":key, "path":str(ld.path), "n":ld.data.n,
                    "rmin_kpc":float(ld.data.radius_kpc.min()),
                    "rmax_kpc":float(ld.data.radius_kpc.max()),
                    "sigma_min":float(ld.data.sigma_kms.min()),
                    "sigma_max":float(ld.data.sigma_kms.max()),
                })
                mpi_print(
                    f"[dados] {key}: {ld.path.name} | N={ld.data.n} | "
                    f"R={ld.data.radius_kpc.min():.6g}--{ld.data.radius_kpc.max():.6g} kpc",
                    root_only=True,
                )
            except Exception as exc:
                preflight_errors.append({"dataset":key,"error":repr(exc),"traceback":traceback.format_exc()})
                mpi_print(f"[ERRO DE ENTRADA] {key}: {exc}",root_only=True)
        pd.DataFrame(preflight_rows).to_csv(root/"input_preflight.csv",index=False)
        if preflight_errors:
            (root/"input_preflight_errors.json").write_text(
                json.dumps(preflight_errors,indent=2,ensure_ascii=False)+"\n",encoding="utf-8"
            )
    else:
        preflight_errors=None
    preflight_errors=MPI_COMM.bcast(preflight_errors,root=0)
    if preflight_errors:
        MPI_COMM.Barrier()
        return 2

    my_cases=cases[MPI_RANK::MPI_SIZE]
    loaded_cache: Dict[str,LoadedDataset]={}
    local_ok=[]; local_fail=[]
    for case in my_cases:
        try:
            local_ok.append(execute_case(case,args,root,loaded_cache))
        except Exception as exc:
            err={"case_id":case.case_id,"dataset":case.dataset_key,"disk_mode":case.disk_mode,"rank":MPI_RANK,"error":repr(exc),"traceback":traceback.format_exc()}
            local_fail.append(err)
            faildir=ensure_dir(root/f"disk_{case.disk_mode}"/case.case_id)
            (faildir/"FAILED.json").write_text(json.dumps(err,indent=2,ensure_ascii=False)+"\n",encoding="utf-8")
            mpi_print(f"[ERRO] {case.case_id}: {exc}")

    gathered_ok=MPI_COMM.gather(local_ok,root=0); gathered_fail=MPI_COMM.gather(local_fail,root=0)
    MPI_COMM.Barrier()
    if MPI_RANK==0:
        all_ok=[x for batch in gathered_ok for x in batch]
        all_fail=[x for batch in gathered_fail for x in batch]
        aggregate_results(root,args,cases,all_fail)
        elapsed=(time.time()-t0)/60.0
        mpi_print("="*88,root_only=True)
        mpi_print(f"Concluído: {len(all_ok)}/{len(cases)} casos | falhas={len(all_fail)} | {elapsed:.2f} min",root_only=True)
        mpi_print(f"Resultados: {root}",root_only=True)
        mpi_print("="*88,root_only=True)
        return 0 if not all_fail else 2
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
