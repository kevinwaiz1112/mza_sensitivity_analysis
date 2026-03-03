import os
import json
import pickle
import copy
import re
from dataclasses import dataclass
from typing import Dict, Any, List, Optional
from pathlib import Path

import numpy as np
import pandas as pd

# NOTE:
# - keep import name consistent with your runner file
#   If your runner module is called sa_sim_runner.py, change to:
#       from sa_sim_runner import SAParams, run_many
from sim_wrapper import SAParams, run_many

from utils import LPGSelectionConfig


# ───────────────────────────────────────────────────────────────
# Repo-relative paths (Windows + Linux)
# ───────────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent  # assumes: <repo>/<something>/sa_head.py (here: ".../sensitivity analysis/sa_head.py")


# ───────────────────────────────────────────────────────────────
# 1) Variants
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
# 2) Paths / Inputs (RELATIVE)
# ───────────────────────────────────────────────────────────────

# AixLib submodule inside repo
AIXLIB_MO = (REPO_ROOT / "external" / "AixLib" / "AixLib" / "package.mo").resolve()

# Data dir inside repo (adjust to your repo layout)
DATA_DIR = (REPO_ROOT / "data").resolve()

# Building data inside repo (adjust if needed)
BUILDING_DATA_CSV = (DATA_DIR / "building_data" / "building_data.csv").resolve()
BUILDING_DATA_PKL = (DATA_DIR / "building_data" / "building_data.pkl").resolve()

# TEASER export base directory (models live here)
TEASER_BASE = (REPO_ROOT / "teaser_models" / "sensitivity_analysis").resolve()

# Results output
OUT_BASE = (REPO_ROOT / "results" / "sa_results").resolve()

# Modelica package namespace prefix (unchanged)
SIM_MODEL_PKG_PREFIX = "sensitivity_analysis"

# Weather files inside repo (adjust filenames)
WEATHER_DIR = (DATA_DIR / "weather").resolve()
MOS_FILES: Dict[str, str] = {
    "TRY_A": str((WEATHER_DIR / "TRY_A.mos").resolve()),
    "TRY_B": str((WEATHER_DIR / "TRY_B.mos").resolve()),
    # optional:
    # "WeekCold": str((WEATHER_DIR / "WeekCold.mos").resolve()),
    # "WeekHot":  str((WEATHER_DIR / "WeekHot.mos").resolve()),
}

# Simulation window
YEAR = 2021
START_SIM = 0
END_SIM = 365 * 24 * 3600

# Parallelization
N_PROC = 1

# Dry-run
DRY_RUN = True            # True = build+validate inputs + preview json; no simulation
DRY_RUN_PREVIEW_N = 10    # preview N tasks
DRY_RUN_BUILD_SAMPLE = True  # also write per-task small preview JSON for first N


# ───────────────────────────────────────────────────────────────
# Dry-run helpers
# ───────────────────────────────────────────────────────────────
def _jsonable_sa_params(p: SAParams) -> Dict[str, Any]:
    """SAParams is a dataclass; nested LPGSelectionConfig as well."""
    d = {
        "gains_scale": float(p.gains_scale),
        "zone_weights": p.zone_weights,
        "rng_seed": int(p.rng_seed),
        "enable_cooling": bool(p.enable_cooling),
        "th_zone_index": int(p.th_zone_index),
        "th_people_factor": float(p.th_people_factor),
        "th_lights_factor": float(p.th_lights_factor),
        "th_occ_rel_factor": float(p.th_occ_rel_factor),
        "th_machines_factor": float(p.th_machines_factor),
        "record_overrides_global": dict(p.record_overrides_global or {}),
        "record_overrides_by_zone": p.record_overrides_by_zone,
    }
    cfg = p.lpg_cfg
    d["lpg_cfg"] = {
        "n_apartments": int(cfg.n_apartments),
        "size_probs": dict(cfg.size_probs),
        "template_to_persons": dict(cfg.template_to_persons),
        "tp_mode": str(cfg.tp_mode),
        "tp_fixed": str(cfg.tp_fixed),
        "lpg_seed_mode": str(cfg.lpg_seed_mode),
        "r_values": list(cfg.r_values),
    }
    return d


def _preview_task(task: Dict[str, Any]) -> Dict[str, Any]:
    """Compact preview (avoid huge payload dumps)."""
    bd = (task.get("building_payload", {}) or {}).get("building_data", {}) or {}
    zone_control = bd.get("sa_zone_control", task.get("zone_control"))
    zones = list((zone_control or {}).get("zones", {}).keys()) if isinstance(zone_control, dict) else []

    return {
        "out_dir": task.get("out_dir"),
        "variant": (task.get("task_meta") or {}).get("variant"),
        "building_id": task.get("building_id"),
        "weather_key": (task.get("task_meta") or {}).get("weather_key"),
        "seed": (task.get("task_meta") or {}).get("seed"),
        "sample_id": ((task.get("task_meta") or {}).get("sample") or {}).get("sample_id"),
        "wwr_factor": ((task.get("task_meta") or {}).get("sample") or {}).get("wwr_factor"),
        "gains_scale": ((task.get("task_meta") or {}).get("sample") or {}).get("gains_scale"),
        "sa_tabula_year_class": bd.get("sa_tabula_year_class"),
        "sa_yoc_base_class": bd.get("sa_yoc_base_class"),
        "sa_yoc_error_step": bd.get("sa_yoc_error_step"),
        "sa_tset_mean_K": bd.get("sa_tset_mean_K"),
        "sa_tset_spread_K": bd.get("sa_tset_spread_K"),
        "n_zone_control_zones": len(zones),
        "zone_names_first5": zones[:5],
        "sa_params": _jsonable_sa_params(task["sa_params"]) if "sa_params" in task else None,
    }


def dry_run_validate_and_save(tasks: List[Dict[str, Any]], out_base: Path, preview_n: int = 10) -> None:
    """
    Validates that tasks are consistent and writes preview files.
    No simulation, no runner call.
    """
    out_base = Path(out_base)
    out_base.mkdir(parents=True, exist_ok=True)

    previews: List[Dict[str, Any]] = []

    for i, t in enumerate(tasks):
        # Required keys for the runner
        required = [
            "out_dir", "sim_models_dir", "sim_model_pkg_name", "aixlib_mo",
            "building_id", "mos_file_path", "year", "start_sim", "end_sim",
            "sa_params", "internal_gains_mode",
        ]
        for k in required:
            if k not in t:
                raise KeyError(f"Task {i} missing key '{k}'")

        # LPG config checks
        cfg = t["sa_params"].lpg_cfg
        if cfg is None:
            raise ValueError(f"Task {i}: sa_params.lpg_cfg is None")

        if not cfg.template_to_persons:
            raise ValueError(f"Task {i}: lpg_cfg.template_to_persons is empty")

        ssum = float(sum(cfg.size_probs.values()))
        if not (0.999 <= ssum <= 1.001):
            raise ValueError(f"Task {i}: size_probs sum={ssum} (expected ~1)")

        # Weather file extension sanity (path may still be placeholder)
        if Path(str(t["mos_file_path"])).suffix.lower() != ".mos":
            pass

        if i < preview_n:
            previews.append(_preview_task(t))

        if DRY_RUN_BUILD_SAMPLE and i < preview_n:
            p = _preview_task(t)
            p_path = out_base / f"dryrun_task_{i:03d}.json"
            with open(p_path, "w", encoding="utf-8") as f:
                json.dump(p, f, indent=2, ensure_ascii=False)

    preview_path = out_base / "dryrun_preview.json"
    with open(preview_path, "w", encoding="utf-8") as f:
        json.dump({"n_tasks": len(tasks), "preview_n": len(previews), "preview": previews}, f, indent=2, ensure_ascii=False)

    print(f"[DRY_RUN] Built {len(tasks)} tasks. Preview saved to: {preview_path}")
    if previews:
        print("[DRY_RUN] First preview item:")
        print(json.dumps(previews[0], indent=2, ensure_ascii=False))


# ───────────────────────────────────────────────────────────────
# 3) LPG Config Builder
# ───────────────────────────────────────────────────────────────
HOUSEHOLD_TEMPLATES = [
    # 1 person
    ("CHR07", 1), ("CHR09", 1), ("CHR10", 1), ("CHR13", 1),
    ("CHR23", 1), ("CHR24", 1), ("CHR30", 1), ("OR01", 1),
    # 2 persons
    ("CHR01", 2), ("CHR02", 2), ("CHR16", 2), ("CHR17", 2),
    # 3 persons
    ("CHR03", 3), ("CHR52", 3),
    # 4 persons
    ("CHR27", 4),
    # 5 persons
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
# 4) Building-Data Loader
# ───────────────────────────────────────────────────────────────
def load_building_data(pkl_path: Path, csv_path: Path) -> List[Dict[str, Any]]:
    """
    Prefer PKL. CSV fallback.
    Returns list of dicts with keys:
      - building_id
      - payload (if PKL)
      - rows (if CSV fallback)
    """
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
    """Map building_id -> payload dict (PKL case only)."""
    out: Dict[str, Dict[str, Any]] = {}
    for b in building_data:
        bid = str(b.get("building_id", "")).strip()
        payload = b.get("payload")
        if bid and isinstance(payload, dict):
            out[bid] = payload
    return out


# ───────────────────────────────────────────────────────────────
# 5) Variant -> TEASER Export paths
# ───────────────────────────────────────────────────────────────
def sim_models_dir_for_variant(v: VariantSpec) -> str:
    # return string for runner compatibility (utils likely uses os.path)
    return str((TEASER_BASE / f"Var_{v.idx}").resolve())


def sim_model_pkg_name_for_variant(v: VariantSpec) -> str:
    return f"{SIM_MODEL_PKG_PREFIX}.Var_{v.idx}"


# ───────────────────────────────────────────────────────────────
# 6) Sampling: YoC + Setpoints + WWR + Weather + GainsScale
# ───────────────────────────────────────────────────────────────
def _get_tabula_year_class(payload: Dict[str, Any], fallback: int = 6) -> int:
    bd = payload.get("building_data", {}) if isinstance(payload.get("building_data"), dict) else {}
    v = bd.get("tabula_year_class", bd.get("bldg:tabula_year_class", fallback))
    try:
        return int(v)
    except Exception:
        return int(fallback)


def _clamp_int(x: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, int(x)))


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


def apply_yoc_and_setpoints_to_payload(
    base_payload: Dict[str, Any],
    rng: np.random.Generator,
    yoc_accuracy: float = 0.83,
    yoc_min_class: int = 1,
    yoc_max_class: int = 12,
    # Setpoints
    t0_C: float = 20.0,
    sigma_mean_K_min: float = 1.0,
    sigma_mean_K_max: float = 2.0,
    spread_K_min: float = 0.5,
    spread_K_max: float = 2.0,
    per_zone_clip_K: float = 4.0,
    core_setpoint_C: float = 8.0,
) -> Dict[str, Any]:
    """
    Adds SA fields into a copy of the payload:
      - building_data["sa_tabula_year_class"]
      - building_data["sa_zone_control"] (per-zone heating/cooling setpoints)
    """
    payload = copy.deepcopy(base_payload)
    bd = payload.setdefault("building_data", {})
    if not isinstance(bd, dict):
        bd = {}
        payload["building_data"] = bd

    # YoC: correct vs neighbor class
    base_class = _get_tabula_year_class(payload, fallback=6)
    is_correct = bool(rng.random() < float(yoc_accuracy))
    if is_correct:
        yoc_class = base_class
        yoc_error = 0
    else:
        step = int(rng.choice([-1, 1]))
        yoc_class = _clamp_int(base_class + step, yoc_min_class, yoc_max_class)
        yoc_error = step

    bd["sa_tabula_year_class"] = int(yoc_class)
    bd["sa_yoc_is_correct"] = bool(is_correct)
    bd["sa_yoc_base_class"] = int(base_class)
    bd["sa_yoc_error_step"] = int(yoc_error)

    # Setpoints: mean + spread, heterogenous per zone
    sigma_mean = float(rng.uniform(sigma_mean_K_min, sigma_mean_K_max))
    t_mean_C = float(rng.normal(loc=t0_C, scale=sigma_mean))
    t_mean_K = t_mean_C + 273.15
    t_spread_K = float(rng.uniform(spread_K_min, spread_K_max))

    zone_names = _extract_zone_names(payload)
    n_cores = int(bd.get("bldg:n_cores", 1) or 1)

    zone_control = {
        "default": {"heated": True, "heat_setpoint_K": float(t_mean_K), "cooled": False, "cool_setpoint_K": 0.0},
        "zones": {},
        "rules": [],
    }

    core_sp_K = float(core_setpoint_C + 273.15)

    for zn in zone_names:
        if _is_core_zone(zn, n_cores):
            zone_control["zones"][zn] = {
                "heated": False,
                "heat_setpoint_K": core_sp_K,
                "cooled": False,
                "cool_setpoint_K": 0.0,
            }
            continue

        z_sp = float(rng.normal(loc=t_mean_K, scale=t_spread_K))
        z_sp = float(np.clip(z_sp, t_mean_K - per_zone_clip_K, t_mean_K + per_zone_clip_K))

        zone_control["zones"][zn] = {
            "heated": True,
            "heat_setpoint_K": z_sp,
            "cooled": False,
            "cool_setpoint_K": 0.0,
        }

    bd["sa_tset_mean_K"] = float(t_mean_K)
    bd["sa_tset_spread_K"] = float(t_spread_K)
    bd["sa_zone_control"] = zone_control

    return payload


def make_samples(
    N: int,
    rng_seed: int,
    weather_options: List[str],
    wwr_min: float = 0.8,
    wwr_max: float = 1.2,
    gains_min: float = 0.8,
    gains_max: float = 1.2,
) -> List[Dict[str, Any]]:
    rng = np.random.default_rng(int(rng_seed))
    if not weather_options:
        raise ValueError("weather_options is empty")

    samples: List[Dict[str, Any]] = []
    for i in range(int(N)):
        samples.append({
            "sample_id": int(i),
            "wwr_factor": float(rng.uniform(wwr_min, wwr_max)),
            "gains_scale": float(rng.uniform(gains_min, gains_max)),
            "weather_key": str(rng.choice(weather_options)),
        })
    return samples


def make_seeds(S: int, base: int = 1) -> List[int]:
    return [int(base + i) for i in range(int(S))]


# ───────────────────────────────────────────────────────────────
# 7) Build tasks
# ───────────────────────────────────────────────────────────────
def build_tasks(
    variants: List[str],
    building_payloads_by_id: Dict[str, Dict[str, Any]],
    samples: List[Dict[str, Any]],
    seeds: List[int],
) -> List[Dict[str, Any]]:
    tasks: List[Dict[str, Any]] = []

    for vkey in variants:
        if vkey not in VARIANTS:
            raise KeyError(f"Unknown variant: {vkey}")
        v = VARIANTS[vkey]
        sim_models_dir = sim_models_dir_for_variant(v)
        sim_pkg = sim_model_pkg_name_for_variant(v)

        for bid, base_payload in building_payloads_by_id.items():
            for s in samples:
                # deterministic per (building, sample): YoC + setpoints stable across seeds
                mix_seed = (hash(str(bid)) & 0xFFFFFFFF) ^ (int(s["sample_id"]) * 2654435761)
                rng_bs = np.random.default_rng(int(mix_seed))

                enriched_payload = apply_yoc_and_setpoints_to_payload(
                    base_payload=base_payload,
                    rng=rng_bs,
                    yoc_accuracy=0.83,
                )

                zone_control = enriched_payload.get("building_data", {}).get("sa_zone_control", None)

                wkey = str(s["weather_key"])
                if wkey not in MOS_FILES:
                    raise KeyError(f"weather_key '{wkey}' not in MOS_FILES: {list(MOS_FILES.keys())}")
                mos_path = MOS_FILES[wkey]

                for seed in seeds:
                    sa_params = SAParams(
                        gains_scale=float(s["gains_scale"]),
                        zone_weights=None,
                        rng_seed=int(seed),
                        lpg_cfg=make_lpg_cfg(
                            n_apartments=v.n_apartments,
                            tp_key="TP_BER21",
                            seed_mode="random_r",
                        ),
                        enable_cooling=False,
                        th_zone_index=0,
                        th_people_factor=0.1,
                        th_lights_factor=0.1,
                        th_occ_rel_factor=0.1,
                        th_machines_factor=0.0,
                        record_overrides_global={"wwr_factor": float(s["wwr_factor"])},
                        record_overrides_by_zone=None,
                    )

                    out_dir = (
                        OUT_BASE
                        / vkey
                        / str(bid)
                        / wkey
                        / f"sample_{int(s['sample_id']):04d}"
                        / f"seed_{int(seed)}"
                    )

                    task_meta = {
                        "variant": vkey,
                        "variant_idx": v.idx,
                        "expected_n_zones": v.expected_n_zones,
                        "sample": dict(s),
                        "seed": int(seed),
                        "weather_key": wkey,
                        "mos_file_path": mos_path,
                        "sa_tabula_year_class": enriched_payload.get("building_data", {}).get("sa_tabula_year_class"),
                        "sa_tset_mean_K": enriched_payload.get("building_data", {}).get("sa_tset_mean_K"),
                        "sa_tset_spread_K": enriched_payload.get("building_data", {}).get("sa_tset_spread_K"),
                    }

                    tasks.append({
                        "out_dir": str(out_dir),
                        "sim_models_dir": str(sim_models_dir),
                        "sim_model_pkg_name": sim_pkg,
                        "aixlib_mo": str(AIXLIB_MO),
                        "building_id": str(bid),
                        "mos_file_path": str(mos_path),
                        "year": int(YEAR),
                        "start_sim": int(START_SIM),
                        "end_sim": int(END_SIM),
                        "lpg_results_path": "",
                        "sa_params": sa_params,
                        "internal_gains_mode": "multizone_table",
                        "building_payload": enriched_payload,
                        "zone_control": zone_control,
                        "task_meta": task_meta,
                    })

    return tasks


# ───────────────────────────────────────────────────────────────
# 8) Main
# ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # basic path checks (optional but helpful)
    if not AIXLIB_MO.is_file():
        raise FileNotFoundError(f"AIXLIB_MO not found: {AIXLIB_MO}")
    if not (BUILDING_DATA_PKL.is_file() or BUILDING_DATA_CSV.is_file()):
        raise FileNotFoundError(f"Building data not found: {BUILDING_DATA_PKL} / {BUILDING_DATA_CSV}")

    building_data = load_building_data(BUILDING_DATA_PKL, BUILDING_DATA_CSV)
    payloads_by_id = index_payloads_by_id(building_data)

    if not payloads_by_id:
        raise ValueError(
            "No payloads found from PKL. For Step 2 (TEASER build) the PKL format (list of payload dicts) is required."
        )

    variant_keys = ["V1", "V2", "V3", "V4", "V5", "V6", "V7", "V8"]
    weather_options = ["TRY_A", "TRY_B"]  # add "WeekCold", "WeekHot" if you enable them in MOS_FILES

    N_samples = 70
    S_seeds = 5

    samples = make_samples(N=N_samples, rng_seed=42, weather_options=weather_options)
    seeds = make_seeds(S=S_seeds, base=1)

    tasks = build_tasks(
        variants=variant_keys,
        building_payloads_by_id=payloads_by_id,
        samples=samples,
        seeds=seeds,
    )

    OUT_BASE.mkdir(parents=True, exist_ok=True)
    manifest_path = OUT_BASE / "run_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump({
            "step": 2,
            "repo_root": str(REPO_ROOT),
            "variant_keys": variant_keys,
            "weather_options": weather_options,
            "N_samples": int(N_samples),
            "S_seeds": int(S_seeds),
            "n_buildings": int(len(payloads_by_id)),
            "n_tasks": int(len(tasks)),
            "dry_run": bool(DRY_RUN),
            "paths": {
                "AIXLIB_MO": str(AIXLIB_MO),
                "BUILDING_DATA_CSV": str(BUILDING_DATA_CSV),
                "BUILDING_DATA_PKL": str(BUILDING_DATA_PKL),
                "TEASER_BASE": str(TEASER_BASE),
                "OUT_BASE": str(OUT_BASE),
                "WEATHER_DIR": str(WEATHER_DIR),
                "MOS_FILES": dict(MOS_FILES),
            },
        }, f, indent=2, ensure_ascii=False)

    print(f"[DEBUG] DRY_RUN={DRY_RUN} (type={type(DRY_RUN)})")
    if DRY_RUN:
        dry_run_validate_and_save(tasks, out_base=OUT_BASE, preview_n=DRY_RUN_PREVIEW_N)
    else:
        raise RuntimeError("DRY_RUN=False: simulation is intentionally disabled in this script.")