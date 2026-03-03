# sa_step4_sobol_head.py
#
# Step 4 — Sobol as 2 cases (1 building × 2 weather)
# Goal: compare variance shares and interactions between TRY_A and TRY_B
#
# Design:
# - 1 variant (e.g., V3)
# - 1 building
# - 2 cases: TRY_A and TRY_B
# - k_sob inputs (default: 5)
# - N_sob base sample size (default: 128)
# - Total runs per case: (k+2)*N  (Saltelli 2002 style: A, B, and k mixed matrices)
# - Seeds: Ssob = 1 (seed effects handled in Step 2)
#
# Assumptions:
# - Your runner (sim_wrapper / sa_sim_runner) already:
#   - rebuilds TEASER each run (force_teaser_rebuild=True)
#   - applies WWR by scaling AWin & ATransparent in zone records (wwr_factor)
#   - writes setpoint tables from zone_control
#   - applies record overrides via record_overrides_global (regex on zone records)
#   - outputs overall.json + timeseries.csv + model_export
#
# IMPORTANT:
# - "infiltration" only has an effect if the record key you set exists in the exported ZoneBaseRecord
#   (e.g. baseACH or maxUserACH). If you want a single "infiltration factor" you can map it to
#   baseACH and/or other ACH-related keys.

import os
import json
import pickle
import copy
import re
from dataclasses import dataclass
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path

import numpy as np
import pandas as pd

from sim_wrapper import SAParams, run_many
from utils import LPGSelectionConfig


# ───────────────────────────────────────────────────────────────
# Repo-relative paths
# ───────────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent

AIXLIB_MO = (REPO_ROOT / "external" / "AixLib" / "AixLib" / "package.mo").resolve()

DATA_DIR = (REPO_ROOT / "data").resolve()
BUILDING_DATA_CSV = (DATA_DIR / "building_data" / "building_data.csv").resolve()
BUILDING_DATA_PKL = (DATA_DIR / "building_data" / "building_data.pkl").resolve()

TEASER_BASE = (REPO_ROOT / "teaser_models" / "sensitivity_analysis").resolve()

# Step 4 outputs to separate folder
OUT_BASE = (REPO_ROOT / "results" / "sa_step4_sobol").resolve()

SIM_MODEL_PKG_PREFIX = "sensitivity_analysis"

WEATHER_DIR = (DATA_DIR / "weather").resolve()
MOS_FILES: Dict[str, str] = {
    "TRY_A": str((WEATHER_DIR / "TRY_A.mos").resolve()),
    "TRY_B": str((WEATHER_DIR / "TRY_B.mos").resolve()),
}

YEAR = 2021
START_SIM = 0
END_SIM = 365 * 24 * 3600

N_PROC = 1


# ───────────────────────────────────────────────────────────────
# Step 4 configuration (Sobol)
# ───────────────────────────────────────────────────────────────
SOBOL_VARIANT_KEY = "V3"
SOBOL_WEATHERS = ["TRY_A", "TRY_B"]

# Choose exactly 1 building; if None => first from PKL
BUILDING_ID_FILTER: Optional[str] = None

# Sobol setup
K_SOB = 5
N_SOB = 128
S_SEED = 1  # one usage draw per point

# Saltelli-style design:
# - A matrix: N × k
# - B matrix: N × k
# - A_Bi matrices: N × k (A with column i from B)
# Total model evaluations = (k + 2) * N per case
DESIGN_SEED = 20240229

# Dry-run
DRY_RUN = True
DRY_RUN_PREVIEW_N = 12


# ───────────────────────────────────────────────────────────────
# Variants (minimal: only need idx + apartments)
# ───────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class VariantSpec:
    key: str
    idx: int
    n_apartments: int


VARIANTS: Dict[str, VariantSpec] = {
    "V1": VariantSpec("V1", 1, 6),
    "V2": VariantSpec("V2", 2, 6),
    "V3": VariantSpec("V3", 3, 6),
    "V4": VariantSpec("V4", 4, 6),
    "V5": VariantSpec("V5", 5, 12),
    "V6": VariantSpec("V6", 6, 12),
    "V7": VariantSpec("V7", 7, 6),
    "V8": VariantSpec("V8", 8, 6),
}


def sim_models_dir_for_variant(v: VariantSpec) -> str:
    return str((TEASER_BASE / f"Var_{v.idx}").resolve())


def sim_model_pkg_name_for_variant(v: VariantSpec) -> str:
    return f"{SIM_MODEL_PKG_PREFIX}.Var_{v.idx}"


# ───────────────────────────────────────────────────────────────
# LPG config builder (same curated templates as Step 2/3)
# ───────────────────────────────────────────────────────────────
HOUSEHOLD_TEMPLATES = [
    ("CHR07", 1), ("CHR09", 1), ("CHR10", 1), ("CHR13", 1),
    ("CHR23", 1), ("CHR24", 1), ("CHR30", 1), ("OR01", 1),
    ("CHR01", 2), ("CHR02", 2), ("CHR16", 2), ("CHR17", 2),
    ("CHR03", 3), ("CHR52", 3),
    ("CHR27", 4),
    ("CHR41", 5),
]
HOUSEHOLD_SIZE_PROBS = {1: 0.40, 2: 0.35, 3: 0.15, 4: 0.07, 5: 0.03}
TEMPLATE_TO_PERSONS = {code: int(persons) for code, persons in HOUSEHOLD_TEMPLATES}


def make_lpg_cfg(n_apartments: int, tp_key: str = "TP_BER21", seed_mode: str = "random_r") -> LPGSelectionConfig:
    return LPGSelectionConfig(
        n_apartments=int(n_apartments),
        size_probs=dict(HOUSEHOLD_SIZE_PROBS),
        template_to_persons=dict(TEMPLATE_TO_PERSONS),
        tp_mode="fixed",
        tp_fixed=str(tp_key),
        lpg_seed_mode=str(seed_mode),
        r_values=(1, 2, 3, 4, 5),
    )


# ───────────────────────────────────────────────────────────────
# Building loader (PKL required for TEASER build)
# ───────────────────────────────────────────────────────────────
def load_building_data(pkl_path: Path, csv_path: Path) -> List[Dict[str, Any]]:
    pkl_path = Path(pkl_path)
    csv_path = Path(csv_path)

    if pkl_path.is_file():
        with open(pkl_path, "rb") as fh:
            data = pickle.load(fh)

        out: List[Dict[str, Any]] = []
        if isinstance(data, list):
            for b in data:
                if not isinstance(b, dict):
                    continue
                bid = b.get("building_id") or b.get("Building ID") or b.get("id")
                if bid is None:
                    continue
                out.append({"building_id": str(bid), "payload": b})
            return out
        raise ValueError("PKL has unexpected format (not a list).")

    # CSV fallback (not recommended for Step 4)
    if not csv_path.is_file():
        raise FileNotFoundError(f"Neither PKL nor CSV found: {pkl_path} / {csv_path}")

    df = pd.read_csv(csv_path)
    id_col = "Building ID" if "Building ID" in df.columns else ("building_id" if "building_id" in df.columns else None)
    if id_col is None:
        raise KeyError("CSV needs a building id column: 'Building ID' or 'building_id'")

    out: List[Dict[str, Any]] = []
    for bid, grp in df.groupby(id_col):
        out.append({"building_id": str(bid), "rows": grp.reset_index(drop=True).to_dict(orient="records")})
    return out


def index_payloads_by_id(building_data: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for b in building_data:
        bid = str(b.get("building_id", "")).strip()
        payload = b.get("payload")
        if bid and isinstance(payload, dict):
            out[bid] = payload
    return out


# ───────────────────────────────────────────────────────────────
# Zone helpers + deterministic zone_control for Sobol
# ───────────────────────────────────────────────────────────────
def _extract_zone_names(payload: Dict[str, Any]) -> List[str]:
    polys = payload.get("polygons", {})
    storeys = polys.get("storeys", []) if isinstance(polys, dict) else []
    names: List[str] = []
    for st in storeys:
        for z in st.get("zones", []) or []:
            zn = z.get("name")
            if zn:
                names.append(str(zn))
    if not names:
        return ["ThermalZone"]
    return list(dict.fromkeys(names))


def _zone_index(zname: str) -> Optional[int]:
    m = re.search(r"Zone_(\d+)$", str(zname))
    return int(m.group(1)) if m else None


def _is_core_zone(zname: str, n_cores: int) -> bool:
    idx = _zone_index(zname)
    return (idx is not None) and (1 <= idx <= int(n_cores))


def build_zone_control_deterministic(
    payload: Dict[str, Any],
    mean_K: float,
    spread_K: float,
    core_setpoint_C: float = 8.0,
    clip_K: float = 6.0,
) -> Dict[str, Any]:
    bd = payload.get("building_data", {}) if isinstance(payload.get("building_data"), dict) else {}
    zone_names = _extract_zone_names(payload)
    n_cores = int(bd.get("bldg:n_cores", 1) or 1)

    core_sp_K = float(core_setpoint_C + 273.15)

    non_core = [zn for zn in zone_names if not _is_core_zone(zn, n_cores)]
    m = max(1, len(non_core))
    pattern = np.linspace(-1.0, 1.0, m)

    zc = {
        "default": {"heated": True, "heat_setpoint_K": float(mean_K), "cooled": False, "cool_setpoint_K": 0.0},
        "zones": {},
        "rules": [],
    }

    j = 0
    for zn in zone_names:
        if _is_core_zone(zn, n_cores):
            zc["zones"][zn] = {"heated": False, "heat_setpoint_K": core_sp_K, "cooled": False, "cool_setpoint_K": 0.0}
        else:
            sp = float(mean_K + spread_K * float(pattern[j]))
            sp = float(np.clip(sp, mean_K - clip_K, mean_K + clip_K))
            zc["zones"][zn] = {"heated": True, "heat_setpoint_K": sp, "cooled": False, "cool_setpoint_K": 0.0}
            j += 1

    return zc


# ───────────────────────────────────────────────────────────────
# Sobol inputs (k = 5)
# ───────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class SobolParam:
    name: str
    lo: float
    hi: float
    kind: str         # "wwr", "setpoints", "record_global", "sa_params"
    target_key: str   # where it maps


SOBOL_PARAMS: List[SobolParam] = [
    SobolParam("wwr_factor",     0.8, 1.2, "wwr",       "wwr_factor"),
    SobolParam("tset_mean_C",   18.0, 22.0, "setpoints", "tset_mean_C"),
    SobolParam("tset_spread_K",  0.5, 2.0, "setpoints", "tset_spread_K"),
    # Infiltration mapping example: baseACH (must exist in records to have effect)
    SobolParam("baseACH",       0.05, 0.8, "record_global", "baseACH"),
    SobolParam("gains_scale",    0.8, 1.2, "sa_params", "gains_scale"),
]

if len(SOBOL_PARAMS) != K_SOB:
    raise ValueError(f"SOBOL_PARAMS len={len(SOBOL_PARAMS)} but K_SOB={K_SOB}")


def _map_unit(u: float, lo: float, hi: float) -> float:
    return float(lo + float(u) * (hi - lo))


def apply_sobol_vector(
    base_payload: Dict[str, Any],
    x_unit: np.ndarray,
) -> Tuple[Dict[str, Any], Dict[str, Any], float, float]:
    """
    Applies Sobol parameter vector to:
      - payload building_data (setpoints zone_control)
      - record_overrides_global (wwr_factor + infiltration/other record keys)
      - gains_scale (SAParams)
    Returns:
      enriched_payload, record_overrides_global, gains_scale, wwr_factor
    """
    payload = copy.deepcopy(base_payload)
    bd = payload.setdefault("building_data", {})
    if not isinstance(bd, dict):
        bd = {}
        payload["building_data"] = bd

    record_overrides_global: Dict[str, Any] = {}
    gains_scale = 1.0
    wwr_factor = 1.0

    # defaults for setpoints
    tset_mean_C = 20.0
    tset_spread_K = 1.0

    for i, p in enumerate(SOBOL_PARAMS):
        u = float(np.clip(float(x_unit[i]), 0.0, 1.0))

        if p.kind == "wwr":
            wwr_factor = _map_unit(u, p.lo, p.hi)
            record_overrides_global[p.target_key] = float(wwr_factor)

        elif p.kind == "sa_params" and p.target_key == "gains_scale":
            gains_scale = _map_unit(u, p.lo, p.hi)

        elif p.kind == "setpoints":
            if p.target_key == "tset_mean_C":
                tset_mean_C = _map_unit(u, p.lo, p.hi)
            elif p.target_key == "tset_spread_K":
                tset_spread_K = _map_unit(u, p.lo, p.hi)

        elif p.kind == "record_global":
            record_overrides_global[p.target_key] = _map_unit(u, p.lo, p.hi)

    # build deterministic zone_control
    mean_K = float(tset_mean_C + 273.15)
    bd["sa_tset_mean_K"] = float(mean_K)
    bd["sa_tset_spread_K"] = float(tset_spread_K)
    bd["sa_zone_control"] = build_zone_control_deterministic(payload, mean_K=mean_K, spread_K=float(tset_spread_K))

    # wwr_factor should always exist for the runner
    record_overrides_global["wwr_factor"] = float(wwr_factor)

    return payload, record_overrides_global, float(gains_scale), float(wwr_factor)


# ───────────────────────────────────────────────────────────────
# Sobol design generation: Saltelli-style matrices A, B, A_Bi
# ───────────────────────────────────────────────────────────────
def sobol_saltelli_design(
    n: int,
    k: int,
    rng: np.random.Generator,
) -> Dict[str, Any]:
    """
    Returns dict with:
      - A: (n,k)
      - B: (n,k)
      - AB: list of k matrices, each (n,k)
    All in unit space [0,1].
    """
    A = rng.random((n, k))
    B = rng.random((n, k))

    AB = []
    for i in range(k):
        M = A.copy()
        M[:, i] = B[:, i]
        AB.append(M)

    return {"A": A, "B": B, "AB": AB}


def iter_saltelli_points(design: Dict[str, Any]) -> List[Tuple[str, int, Optional[int], np.ndarray]]:
    """
    Yields points in the standard order:
      - A rows: tag="A", row=j, col=None
      - B rows: tag="B", row=j, col=None
      - ABi rows: tag="AB", row=j, col=i

    Returns list of tuples:
      (block_tag, row_id, mix_col, x_unit)
    """
    A = design["A"]
    B = design["B"]
    AB = design["AB"]

    n, k = A.shape
    out: List[Tuple[str, int, Optional[int], np.ndarray]] = []

    for j in range(n):
        out.append(("A", int(j), None, A[j, :].copy()))
    for j in range(n):
        out.append(("B", int(j), None, B[j, :].copy()))
    for i in range(k):
        for j in range(n):
            out.append(("AB", int(j), int(i), AB[i][j, :].copy()))

    return out


# ───────────────────────────────────────────────────────────────
# Build tasks per case (weather)
# ───────────────────────────────────────────────────────────────
def build_tasks_sobol_case(
    variant_key: str,
    building_id: str,
    base_payload: Dict[str, Any],
    weather_key: str,
    design: Dict[str, Any],
    seed: int,
) -> List[Dict[str, Any]]:
    if variant_key not in VARIANTS:
        raise KeyError(f"Unknown variant: {variant_key}")
    v = VARIANTS[variant_key]

    if weather_key not in MOS_FILES:
        raise KeyError(f"Unknown weather_key: {weather_key}")
    mos_path = MOS_FILES[weather_key]

    sim_models_dir = sim_models_dir_for_variant(v)
    sim_pkg = sim_model_pkg_name_for_variant(v)

    tasks: List[Dict[str, Any]] = []

    points = iter_saltelli_points(design)
    for run_idx, (block_tag, row_id, mix_col, x_unit) in enumerate(points):
        enriched_payload, rec_glob, gains_scale, wwr_factor = apply_sobol_vector(base_payload, x_unit)
        zone_control = enriched_payload.get("building_data", {}).get("sa_zone_control", None)

        sa_params = SAParams(
            gains_scale=float(gains_scale),
            zone_weights=None,
            rng_seed=int(seed),  # one seed for all Sobol points (per your plan)
            lpg_cfg=make_lpg_cfg(n_apartments=v.n_apartments, tp_key="TP_BER21", seed_mode="random_r"),
            enable_cooling=False,
            th_zone_index=0,
            th_people_factor=0.1,
            th_lights_factor=0.1,
            th_occ_rel_factor=0.1,
            th_machines_factor=0.0,
            record_overrides_global=dict(rec_glob),
            record_overrides_by_zone=None,
        )

        out_dir = (
            OUT_BASE
            / "sobol"
            / variant_key
            / str(building_id)
            / weather_key
            / f"seed_{int(seed)}"
            / f"{block_tag}"
            / (f"col_{mix_col:02d}" if mix_col is not None else "base")
            / f"row_{row_id:04d}"
        )

        meta = {
            "step": 4,
            "method": "sobol_saltelli",
            "variant": variant_key,
            "variant_idx": v.idx,
            "building_id": str(building_id),
            "weather_key": str(weather_key),
            "mos_file_path": str(mos_path),

            "k": int(K_SOB),
            "n_sob": int(N_SOB),
            "seed": int(seed),

            "block": str(block_tag),   # "A", "B", "AB"
            "row_id": int(row_id),
            "mix_col": int(mix_col) if mix_col is not None else None,
            "run_idx": int(run_idx),

            "x_unit": [float(vv) for vv in np.asarray(x_unit, dtype=float).flatten().tolist()],
            "x_mapped": {
                "wwr_factor": float(wwr_factor),
                "gains_scale": float(gains_scale),
                "tset_mean_K": float(enriched_payload.get("building_data", {}).get("sa_tset_mean_K", np.nan)),
                "tset_spread_K": float(enriched_payload.get("building_data", {}).get("sa_tset_spread_K", np.nan)),
                "record_overrides_global": dict(rec_glob),
            },
        }

        tasks.append({
            "out_dir": str(out_dir),
            "sim_models_dir": str(sim_models_dir),
            "sim_model_pkg_name": str(sim_pkg),
            "aixlib_mo": str(AIXLIB_MO),

            "building_id": str(building_id),
            "mos_file_path": str(mos_path),
            "year": int(YEAR),
            "start_sim": int(START_SIM),
            "end_sim": int(END_SIM),

            "lpg_results_path": "",
            "sa_params": sa_params,
            "internal_gains_mode": "multizone_table",

            "building_payload": enriched_payload,
            "zone_control": zone_control,
            "task_meta": meta,

            # ensure TEASER rebuild per run
            "force_teaser_rebuild": True,
        })

    return tasks


def preview_tasks(tasks: List[Dict[str, Any]], n: int = 10) -> None:
    n = min(int(n), len(tasks))
    print(f"[DRY_RUN] tasks={len(tasks)} preview_n={n}")
    for i in range(n):
        tm = tasks[i].get("task_meta", {}) or {}
        print(json.dumps({
            "i": i,
            "out_dir": tasks[i].get("out_dir"),
            "block": tm.get("block"),
            "row_id": tm.get("row_id"),
            "mix_col": tm.get("mix_col"),
            "x_mapped": tm.get("x_mapped", {}),
        }, indent=2))


# ───────────────────────────────────────────────────────────────
# Main
# ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    if not AIXLIB_MO.is_file():
        raise FileNotFoundError(f"AIXLIB_MO not found: {AIXLIB_MO}")
    if not (BUILDING_DATA_PKL.is_file() or BUILDING_DATA_CSV.is_file()):
        raise FileNotFoundError(f"Building data not found: {BUILDING_DATA_PKL} / {BUILDING_DATA_CSV}")
    if SOBOL_VARIANT_KEY not in VARIANTS:
        raise KeyError(f"SOBOL_VARIANT_KEY={SOBOL_VARIANT_KEY} not in VARIANTS")
    for wk in SOBOL_WEATHERS:
        if wk not in MOS_FILES:
            raise KeyError(f"Weather '{wk}' not in MOS_FILES: {list(MOS_FILES.keys())}")

    building_data = load_building_data(BUILDING_DATA_PKL, BUILDING_DATA_CSV)
    payloads_by_id = index_payloads_by_id(building_data)
    if not payloads_by_id:
        raise ValueError("No payloads found from PKL. Step 4 requires PKL payloads for TEASER build.")

    if BUILDING_ID_FILTER is not None:
        if str(BUILDING_ID_FILTER) not in payloads_by_id:
            raise KeyError(f"BUILDING_ID_FILTER={BUILDING_ID_FILTER} not found in payloads.")
        building_id = str(BUILDING_ID_FILTER)
    else:
        building_id = next(iter(payloads_by_id.keys()))

    base_payload = payloads_by_id[building_id]

    rng = np.random.default_rng(int(DESIGN_SEED))
    design = sobol_saltelli_design(n=int(N_SOB), k=int(K_SOB), rng=rng)

    # one seed for all points (per case)
    sobol_seed = 1

    tasks_all: List[Dict[str, Any]] = []
    for wk in SOBOL_WEATHERS:
        tasks_case = build_tasks_sobol_case(
            variant_key=SOBOL_VARIANT_KEY,
            building_id=building_id,
            base_payload=base_payload,
            weather_key=wk,
            design=design,
            seed=int(sobol_seed),
        )
        tasks_all.extend(tasks_case)

    OUT_BASE.mkdir(parents=True, exist_ok=True)

    manifest_path = OUT_BASE / "run_manifest_step4_sobol.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump({
            "step": 4,
            "method": "sobol_saltelli",
            "repo_root": str(REPO_ROOT),
            "variant": SOBOL_VARIANT_KEY,
            "building_id": str(building_id),
            "weathers": list(SOBOL_WEATHERS),

            "k_sob": int(K_SOB),
            "n_sob": int(N_SOB),
            "runs_per_case": int((K_SOB + 2) * N_SOB),
            "runs_total": int(len(tasks_all)),
            "seed": int(sobol_seed),

            "params": [{"name": p.name, "lo": p.lo, "hi": p.hi, "kind": p.kind, "target_key": p.target_key} for p in SOBOL_PARAMS],

            "paths": {
                "AIXLIB_MO": str(AIXLIB_MO),
                "BUILDING_DATA_PKL": str(BUILDING_DATA_PKL),
                "BUILDING_DATA_CSV": str(BUILDING_DATA_CSV),
                "TEASER_BASE": str(TEASER_BASE),
                "OUT_BASE": str(OUT_BASE),
                "MOS_FILES": dict(MOS_FILES),
            },

            "dry_run": bool(DRY_RUN),
        }, f, indent=2, ensure_ascii=False)

    print(f"[STEP4] Built tasks={len(tasks_all)} total. Manifest: {manifest_path}")
    print(f"[STEP4] Runs per case={(K_SOB+2)*N_SOB}, cases={len(SOBOL_WEATHERS)}, total={len(tasks_all)}")

    if DRY_RUN:
        preview_tasks(tasks_all, n=DRY_RUN_PREVIEW_N)
        print("[STEP4] DRY_RUN=True => no simulation.")
    else:
        results = run_many(tasks_all, n_proc=int(N_PROC))
        print(f"[STEP4] Done. Results returned: {len(results)}")