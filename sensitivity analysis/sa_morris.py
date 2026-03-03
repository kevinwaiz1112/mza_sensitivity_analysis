# sa_step3_morris_head.py
#
# Step 3 — Morris screening (filter for Sobol)
# - 1 variant (e.g., V3 or V5)
# - 1 weather (fixed, e.g., TRY_A)
# - k inputs (default: 15)
# - r trajectories (default: 10)
# - seeds per point Sm (default: 2)
#
# This script only builds tasks (like Step 2) and then (optionally) runs them via run_many().
# It assumes your runner already:
# - rebuilds TEASER per run (force_teaser_rebuild=True)
# - applies WWR via record patching (AWin/ATransparent)
# - applies record_overrides_global / record_overrides_by_zone via regex patch
# - writes setpoint tables from zone_control
# - writes outputs (overall.json, timeseries.csv, model_export, zone_map.json)

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

# IMPORTANT: keep consistent with your runner module name
from sim_wrapper import SAParams, run_many
from utils import LPGSelectionConfig


# ───────────────────────────────────────────────────────────────
# Repo-relative paths (Windows + Linux)
# ───────────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent


# ───────────────────────────────────────────────────────────────
# 1) Variants (same as Step 2)
# ───────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class VariantSpec:
    key: str
    idx: int
    n_apartments: int
    expected_n_zones: Optional[int] = None


VARIANTS: Dict[str, VariantSpec] = {
    "V1": VariantSpec(key="V1", idx=1, n_apartments=6,  expected_n_zones=1),
    "V2": VariantSpec(key="V2", idx=2, n_apartments=6,  expected_n_zones=2),
    "V3": VariantSpec(key="V3", idx=3, n_apartments=6,  expected_n_zones=7),
    "V4": VariantSpec(key="V4", idx=4, n_apartments=6,  expected_n_zones=7),
    "V7": VariantSpec(key="V7", idx=7, n_apartments=6,  expected_n_zones=7),
    "V8": VariantSpec(key="V8", idx=8, n_apartments=6,  expected_n_zones=7),
    "V5": VariantSpec(key="V5", idx=5, n_apartments=12, expected_n_zones=13),
    "V6": VariantSpec(key="V6", idx=6, n_apartments=12, expected_n_zones=13),
}


# ───────────────────────────────────────────────────────────────
# 2) Paths / Inputs (repo-relative)
# ───────────────────────────────────────────────────────────────
AIXLIB_MO = (REPO_ROOT / "external" / "AixLib" / "AixLib" / "package.mo").resolve()

DATA_DIR = (REPO_ROOT / "data").resolve()
BUILDING_DATA_CSV = (DATA_DIR / "building_data" / "building_data.csv").resolve()
BUILDING_DATA_PKL = (DATA_DIR / "building_data" / "building_data.pkl").resolve()

TEASER_BASE = (REPO_ROOT / "teaser_models" / "sensitivity_analysis").resolve()

# Step 3 outputs go into a separate folder to keep steps clean
OUT_BASE = (REPO_ROOT / "results" / "sa_step3_morris").resolve()

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
# Step 3 configuration (Morris)
# ───────────────────────────────────────────────────────────────
# Choose exactly 1 variant + 1 weather here:
MORRIS_VARIANT_KEY = "V3"
MORRIS_WEATHER_KEY = "TRY_A"

# Choose exactly 1 building (recommended).
# If None => uses first building in PKL.
BUILDING_ID_FILTER: Optional[str] = None  # e.g. "ID_..." or whatever your PKL uses

# Morris parameters:
K = 15          # number of inputs
R = 10          # number of trajectories
SM = 2          # seeds per point (usage randomness)
LEVELS = 6      # grid levels for Morris (>=4). Using step = 1/(LEVELS-1).

# LPG base (household) behavior:
LPG_TP_KEY = "TP_BER21"
LPG_SEED_MODE = "random_r"

# For Morris screening you typically want deterministic mapping:
# - weather fixed
# - YoC & setpoints & WWR & infiltration etc are set from the design vector (no extra random draws)

# Dry-run:
DRY_RUN = True
DRY_RUN_PREVIEW_N = 12


# ───────────────────────────────────────────────────────────────
# 3) LPG config builder (same template set as Step 2)
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


def make_lpg_cfg(n_apartments: int, tp_key: str = LPG_TP_KEY, seed_mode: str = LPG_SEED_MODE) -> LPGSelectionConfig:
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
# 4) Building data loader (PKL preferred)
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

    # CSV fallback (not recommended for Step 3 because TEASER build needs payload)
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
# 5) Variant -> TEASER export paths
# ───────────────────────────────────────────────────────────────
def sim_models_dir_for_variant(v: VariantSpec) -> str:
    return str((TEASER_BASE / f"Var_{v.idx}").resolve())


def sim_model_pkg_name_for_variant(v: VariantSpec) -> str:
    return f"{SIM_MODEL_PKG_PREFIX}.Var_{v.idx}"


# ───────────────────────────────────────────────────────────────
# 6) Utilities: zones + deterministic zone_control
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
    """
    Deterministic version (no randomness):
    - Core/TH zones: heated=False, setpoint = core_setpoint_C
    - All other zones: setpoint pattern around mean:
        mean +/- spread * pattern_i
      where pattern_i is evenly spaced in [-1, 1] across non-core zones.

    This makes mean and spread identifiable in Morris/Sobol designs.
    """
    bd = payload.get("building_data", {}) if isinstance(payload.get("building_data"), dict) else {}
    zone_names = _extract_zone_names(payload)
    n_cores = int(bd.get("bldg:n_cores", 1) or 1)

    core_sp_K = float(core_setpoint_C + 273.15)

    # build pattern only for non-core zones
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
# 7) Morris parameter definition (k=15)
# ───────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class ParamDef:
    name: str
    lo: float
    hi: float
    kind: str
    target_key: str


def _map_unit_to_range(u: float, lo: float, hi: float) -> float:
    return float(lo + u * (hi - lo))


def _map_unit_to_int_choice(u: float, choices: List[int]) -> int:
    # u in [0,1] -> index in [0..len-1]
    idx = int(np.floor(u * len(choices)))
    idx = min(max(idx, 0), len(choices) - 1)
    return int(choices[idx])


# IMPORTANT:
# - "record_global" parameters must exist as keys in the TEASER zone record files.
#   They will be applied by your runner via record_overrides_global.
# - "payload_bd" parameters must be used inside teaser_export.py (YoC etc).
# - If a key does not exist in records, it will just be ignored (no effect) unless you add it to TEASER export.
#
# These 15 are a reasonable starter set. Adjust keys to match your TEASER record fields.
PARAMS: List[ParamDef] = [
    # 1) WWR scale applied by runner (AWin, ATransparent)
    ParamDef("wwr_factor", 0.8, 1.2, "wwr", "wwr_factor"),

    # 2) internal gains scaling used by runner (sa_params.gains_scale)
    ParamDef("gains_scale", 0.8, 1.2, "sa_params", "gains_scale"),

    # 3-4) setpoint mean/spread (build zone_control deterministically)
    ParamDef("tset_mean_C", 18.0, 22.0, "setpoints", "tset_mean_C"),
    ParamDef("tset_spread_K", 0.5, 2.0, "setpoints", "tset_spread_K"),

    # 5) YoC shift: -1/0/+1 relative to base class
    # (applied as building_data["sa_tabula_year_class"])
    ParamDef("yoc_shift", -1.0, 1.0, "yoc_shift", "yoc_shift"),

    # 6-15) Example record keys (must match TEASER zone record fields if you want effect)
    ParamDef("baseACH", 0.05, 0.8, "record_global", "baseACH"),
    ParamDef("maxUserACH", 0.5, 3.0, "record_global", "maxUserACH"),
    ParamDef("internalGainsMachinesSpecific", 0.1, 1.5, "record_global", "internalGainsMachinesSpecific"),
    ParamDef("lightingPowerSpecific", 3.0, 12.0, "record_global", "lightingPowerSpecific"),
    ParamDef("shadingFactor", 0.3, 1.0, "record_global", "shadingFactor"),
    ParamDef("gWin", 0.45, 0.75, "record_global", "gWin"),
    ParamDef("UWin", 0.9, 2.0, "record_global", "UWin"),
    ParamDef("hConWin", 2.0, 4.0, "record_global", "hConWin"),
    ParamDef("withAirCap", 0.0, 1.0, "record_global_bool", "withAirCap"),
    ParamDef("withAHU", 0.0, 1.0, "record_global_bool", "withAHU"),
]

if len(PARAMS) != K:
    raise ValueError(f"PARAMS list has len={len(PARAMS)} but K={K}. Adjust PARAMS or K.")


# ───────────────────────────────────────────────────────────────
# 8) Morris trajectory generator (simple, dependency-free)
# ───────────────────────────────────────────────────────────────
def morris_trajectories(
    k: int,
    r: int,
    levels: int,
    rng: np.random.Generator,
) -> List[Dict[str, Any]]:
    """
    Builds r trajectories in unit space [0,1]^k on a regular grid with `levels`.
    We use step = 1/(levels-1). Each trajectory has k+1 points.

    Returns list of:
      {
        "trajectory_id": t,
        "points": [x0, x1, ..., xk]  (each is np.ndarray shape (k,))
        "changed_dim": [None, d1, d2, ..., dk]  (which param changed from prev point)
      }
    """
    if levels < 4:
        raise ValueError("levels should be >= 4 for Morris.")
    step = 1.0 / float(levels - 1)

    grid = np.linspace(0.0, 1.0, levels)

    out = []
    for t in range(int(r)):
        # start point: choose grid values (avoid boundary issues by keeping room for +/- step)
        # We'll choose from [step, 1-step] so both directions possible.
        inner_grid = grid[1:-1] if levels > 2 else grid
        x = rng.choice(inner_grid, size=k, replace=True).astype(float)

        perm = rng.permutation(k).tolist()
        points = [x.copy()]
        changed = [None]

        for d in perm:
            x_new = x.copy()
            # decide direction; keep within [0,1]
            if x_new[d] + step <= 1.0 and x_new[d] - step >= 0.0:
                direction = rng.choice([-1.0, 1.0])
            elif x_new[d] + step <= 1.0:
                direction = 1.0
            else:
                direction = -1.0

            x_new[d] = float(np.clip(x_new[d] + direction * step, 0.0, 1.0))
            x = x_new
            points.append(x.copy())
            changed.append(int(d))

        out.append({"trajectory_id": int(t), "points": points, "changed_dim": changed})

    return out


# ───────────────────────────────────────────────────────────────
# 9) Apply parameter vector -> payload + sa_params
# ───────────────────────────────────────────────────────────────
def apply_param_vector(
    base_payload: Dict[str, Any],
    x_unit: np.ndarray,
) -> Tuple[Dict[str, Any], Dict[str, Any], float, float, int]:
    """
    Returns:
      enriched_payload,
      record_overrides_global (dict),
      gains_scale,
      wwr_factor,
      yoc_class (int)
    """
    payload = copy.deepcopy(base_payload)
    bd = payload.setdefault("building_data", {})
    if not isinstance(bd, dict):
        bd = {}
        payload["building_data"] = bd

    # base class from payload
    base_class = bd.get("tabula_year_class", bd.get("bldg:tabula_year_class", 6))
    try:
        base_class = int(base_class)
    except Exception:
        base_class = 6

    record_overrides_global: Dict[str, Any] = {}

    # defaults
    wwr_factor = 1.0
    gains_scale = 1.0
    tset_mean_C = 20.0
    tset_spread_K = 1.0
    yoc_shift = 0

    for i, p in enumerate(PARAMS):
        u = float(np.clip(float(x_unit[i]), 0.0, 1.0))

        if p.kind == "wwr":
            wwr_factor = _map_unit_to_range(u, p.lo, p.hi)

        elif p.kind == "sa_params" and p.target_key == "gains_scale":
            gains_scale = _map_unit_to_range(u, p.lo, p.hi)

        elif p.kind == "setpoints":
            if p.target_key == "tset_mean_C":
                tset_mean_C = _map_unit_to_range(u, p.lo, p.hi)
            elif p.target_key == "tset_spread_K":
                tset_spread_K = _map_unit_to_range(u, p.lo, p.hi)

        elif p.kind == "yoc_shift":
            # map u -> {-1,0,+1}
            yoc_shift = _map_unit_to_int_choice(u, [-1, 0, 1])

        elif p.kind == "record_global":
            record_overrides_global[p.target_key] = _map_unit_to_range(u, p.lo, p.hi)

        elif p.kind == "record_global_bool":
            # u>0.5 => True
            record_overrides_global[p.target_key] = bool(u > 0.5)

        else:
            # unknown kind: ignore
            pass

    # write YoC class (clamped)
    yoc_class = int(np.clip(base_class + int(yoc_shift), 1, 12))
    bd["sa_tabula_year_class"] = int(yoc_class)
    bd["sa_yoc_base_class"] = int(base_class)
    bd["sa_yoc_error_step"] = int(yoc_shift)
    bd["sa_yoc_is_correct"] = bool(yoc_shift == 0)

    # build deterministic zone_control from mean/spread
    mean_K = float(tset_mean_C + 273.15)
    bd["sa_tset_mean_K"] = float(mean_K)
    bd["sa_tset_spread_K"] = float(tset_spread_K)
    bd["sa_zone_control"] = build_zone_control_deterministic(payload, mean_K=mean_K, spread_K=float(tset_spread_K))

    # include wwr_factor in record_overrides so runner finds it
    record_overrides_global["wwr_factor"] = float(wwr_factor)

    return payload, record_overrides_global, float(gains_scale), float(wwr_factor), int(yoc_class)


# ───────────────────────────────────────────────────────────────
# 10) Build tasks (Morris)
# ───────────────────────────────────────────────────────────────
def build_tasks_morris(
    variant_key: str,
    building_id: str,
    base_payload: Dict[str, Any],
    weather_key: str,
    trajectories: List[Dict[str, Any]],
    sm: int,
    seed_base: int = 1000,
) -> List[Dict[str, Any]]:
    """
    Creates tasks for Morris:
      total runs = r * (k+1) * Sm
    """
    if variant_key not in VARIANTS:
        raise KeyError(f"Unknown variant: {variant_key}")
    v = VARIANTS[variant_key]

    sim_models_dir = sim_models_dir_for_variant(v)
    sim_pkg = sim_model_pkg_name_for_variant(v)

    if weather_key not in MOS_FILES:
        raise KeyError(f"Unknown weather_key: {weather_key}, available={list(MOS_FILES.keys())}")
    mos_path = MOS_FILES[weather_key]

    tasks: List[Dict[str, Any]] = []

    global_point_counter = 0
    for traj in trajectories:
        tid = int(traj["trajectory_id"])
        points: List[np.ndarray] = traj["points"]
        changed_dim: List[Optional[int]] = traj["changed_dim"]

        for pid, x_unit in enumerate(points):
            enriched_payload, rec_glob, gains_scale, wwr_factor, yoc_class = apply_param_vector(base_payload, x_unit)

            zone_control = enriched_payload.get("building_data", {}).get("sa_zone_control", None)

            # seeds per point: run same x twice with different LPG seeds
            for si in range(int(sm)):
                seed = int(seed_base + global_point_counter * 10 + si)

                sa_params = SAParams(
                    gains_scale=float(gains_scale),
                    zone_weights=None,
                    rng_seed=int(seed),
                    lpg_cfg=make_lpg_cfg(
                        n_apartments=v.n_apartments,
                        tp_key=LPG_TP_KEY,
                        seed_mode=LPG_SEED_MODE,
                    ),
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
                    / "morris"
                    / variant_key
                    / str(building_id)
                    / weather_key
                    / f"traj_{tid:02d}"
                    / f"point_{pid:02d}"
                    / f"seed_{seed}"
                )

                meta = {
                    "step": 3,
                    "method": "morris",
                    "variant": variant_key,
                    "variant_idx": v.idx,
                    "building_id": str(building_id),
                    "weather_key": str(weather_key),
                    "mos_file_path": str(mos_path),

                    "k": int(K),
                    "r": int(R),
                    "levels": int(LEVELS),
                    "sm": int(sm),

                    "trajectory_id": int(tid),
                    "point_id": int(pid),
                    "global_point_id": int(global_point_counter),
                    "changed_dim": changed_dim[pid] if pid < len(changed_dim) else None,
                    "changed_param": (PARAMS[int(changed_dim[pid])].name if (pid < len(changed_dim) and changed_dim[pid] is not None) else None),

                    # store the full parameter vector (unit + mapped values)
                    "x_unit": [float(vv) for vv in np.asarray(x_unit, dtype=float).flatten().tolist()],
                    "x_mapped": {
                        "wwr_factor": float(wwr_factor),
                        "gains_scale": float(gains_scale),
                        "tset_mean_K": float(enriched_payload.get("building_data", {}).get("sa_tset_mean_K", np.nan)),
                        "tset_spread_K": float(enriched_payload.get("building_data", {}).get("sa_tset_spread_K", np.nan)),
                        "sa_tabula_year_class": int(yoc_class),
                        "record_overrides_global": dict(rec_glob),
                    },

                    "seed": int(seed),
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

            global_point_counter += 1

    return tasks


# ───────────────────────────────────────────────────────────────
# 11) Dry-run preview
# ───────────────────────────────────────────────────────────────
def preview_tasks(tasks: List[Dict[str, Any]], n: int = 10) -> None:
    n = min(int(n), len(tasks))
    print(f"[DRY_RUN] tasks={len(tasks)} preview_n={n}")
    for i in range(n):
        tm = tasks[i].get("task_meta", {}) or {}
        print(json.dumps({
            "i": i,
            "out_dir": tasks[i].get("out_dir"),
            "traj": tm.get("trajectory_id"),
            "point": tm.get("point_id"),
            "seed": tm.get("seed"),
            "changed_param": tm.get("changed_param"),
            "x_mapped": tm.get("x_mapped", {}),
        }, indent=2))


# ───────────────────────────────────────────────────────────────
# 12) Main
# ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # sanity checks
    if not AIXLIB_MO.is_file():
        raise FileNotFoundError(f"AIXLIB_MO not found: {AIXLIB_MO}")
    if not (BUILDING_DATA_PKL.is_file() or BUILDING_DATA_CSV.is_file()):
        raise FileNotFoundError(f"Building data not found: {BUILDING_DATA_PKL} / {BUILDING_DATA_CSV}")
    if MORRIS_WEATHER_KEY not in MOS_FILES:
        raise KeyError(f"MORRIS_WEATHER_KEY={MORRIS_WEATHER_KEY} not in MOS_FILES: {list(MOS_FILES.keys())}")
    if MORRIS_VARIANT_KEY not in VARIANTS:
        raise KeyError(f"MORRIS_VARIANT_KEY={MORRIS_VARIANT_KEY} not in VARIANTS: {list(VARIANTS.keys())}")

    building_data = load_building_data(BUILDING_DATA_PKL, BUILDING_DATA_CSV)
    payloads_by_id = index_payloads_by_id(building_data)
    if not payloads_by_id:
        raise ValueError("No payloads found from PKL. Step 3 requires PKL payloads for TEASER build.")

    # pick building
    if BUILDING_ID_FILTER is not None:
        if str(BUILDING_ID_FILTER) not in payloads_by_id:
            raise KeyError(f"BUILDING_ID_FILTER={BUILDING_ID_FILTER} not found in payloads. Available example: {list(payloads_by_id.keys())[:5]}")
        building_id = str(BUILDING_ID_FILTER)
    else:
        building_id = next(iter(payloads_by_id.keys()))

    base_payload = payloads_by_id[building_id]

    # build Morris design
    rng = np.random.default_rng(12345)
    trajectories = morris_trajectories(k=K, r=R, levels=LEVELS, rng=rng)

    tasks = build_tasks_morris(
        variant_key=MORRIS_VARIANT_KEY,
        building_id=building_id,
        base_payload=base_payload,
        weather_key=MORRIS_WEATHER_KEY,
        trajectories=trajectories,
        sm=SM,
        seed_base=1000,
    )

    OUT_BASE.mkdir(parents=True, exist_ok=True)

    # manifest
    manifest_path = OUT_BASE / "run_manifest_step3_morris.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump({
            "step": 3,
            "method": "morris",
            "repo_root": str(REPO_ROOT),
            "variant": MORRIS_VARIANT_KEY,
            "weather_key": MORRIS_WEATHER_KEY,
            "building_id": str(building_id),

            "k": int(K),
            "r": int(R),
            "levels": int(LEVELS),
            "sm": int(SM),
            "n_tasks": int(len(tasks)),
            "expected_runs": int(R * (K + 1) * SM),

            "params": [{"name": p.name, "lo": p.lo, "hi": p.hi, "kind": p.kind, "target_key": p.target_key} for p in PARAMS],

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

    print(f"[STEP3] Built tasks={len(tasks)} (expected={R*(K+1)*SM}). Manifest: {manifest_path}")

    if DRY_RUN:
        preview_tasks(tasks, n=DRY_RUN_PREVIEW_N)
        print("[STEP3] DRY_RUN=True => no simulation.")
    else:
        # run tasks
        results = run_many(tasks, n_proc=int(N_PROC))
        print(f"[STEP3] Done. Results returned: {len(results)}")