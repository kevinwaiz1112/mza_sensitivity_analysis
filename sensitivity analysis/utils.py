# utils.py  (minimal, cross-platform, only what SA needs)

from __future__ import annotations

import glob
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple, Optional, List

import numpy as np
import pandas as pd


# ───────────────────────────────────────────────────────────────
# Repo-relative roots
# ───────────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent  # adjust if utils.py lives deeper


# ───────────────────────────────────────────────────────────────
# LPG pool configuration (repo-relative)
# You can override with env var LPG_POOL_ROOT
# ───────────────────────────────────────────────────────────────
DEFAULT_LPG_POOL_ROOT = REPO_ROOT / "data" / "lpg" / "HH_AllTemplates_5x" / "HH_AllTemplates_5x"
LPG_POOL_ROOT = Path(os.environ.get("LPG_POOL_ROOT", str(DEFAULT_LPG_POOL_ROOT))).resolve()

TP_CODES = ["TP_BER21", "TP_BER23", "TP_BER25", "TP_DEL25", "TP_FR"]

# Filename labels in LPG exports
LABEL_ELEC = "Electricity"
LABEL_WW = "Warm Water"
LABEL_HW = "Hot Water"
LABEL_CW = "Cold Water"
LABEL_IG = "Inner Device Heat Gains"
LABEL_BAL_OUTSIDE = "BodilyActivityLevel.Outside"

HH_FIXED = "HH1"

# IMPORTANT:
# - Most LPG hourly sum profiles are in kWh/h (= kW).
# - If your Modelica model expects W, set MACHINES_SCALE = 1000.0
MACHINES_SCALE = 1.0


# ───────────────────────────────────────────────────────────────
# Dataclasses
# ───────────────────────────────────────────────────────────────
@dataclass
class LPGSelectionConfig:
    n_apartments: int
    size_probs: Dict[int, float]                # e.g. {1:0.4,2:0.35,3:0.15,4:0.07,5:0.03}
    template_to_persons: Dict[str, int]         # mapping template -> persons
    tp_mode: str = "fixed"                      # "fixed" or "random"
    tp_fixed: str = "TP_BER21"
    lpg_seed_mode: str = "random_r"             # "random_r" or "mean_r"
    r_values: Tuple[int, ...] = (1, 2, 3, 4, 5)


# ───────────────────────────────────────────────────────────────
# Helpers (IDs, TEASER existence)
# ───────────────────────────────────────────────────────────────
def to_dashed_id(bid: str) -> str:
    """
    Converts 'ID_<32hex>' into 'ID_xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx'.
    Returns bid unchanged if it doesn't match.
    """
    m = re.match(r"^(ID_)([A-Fa-f0-9]{32})$", str(bid))
    if not m:
        return str(bid)
    h = m.group(2).upper()
    return f"{m.group(1)}{h[:8]}-{h[8:12]}-{h[12:16]}-{h[16:20]}-{h[20:]}"


def building_model_exists(building_id: str, sim_models_dir: str) -> bool:
    """
    Checks if TEASER/AixLib export exists for `building_id` inside `sim_models_dir`.
    Typical exports:
      - <sim_models_dir>/<building_id>/package.mo
      - <sim_models_dir>/<building_id>.mo (rare)
    """
    root = Path(sim_models_dir)
    return (root / f"{building_id}.mo").is_file() or (root / building_id / "package.mo").is_file()


# ───────────────────────────────────────────────────────────────
# Weather: parse .mos and update Modelica file reference
# ───────────────────────────────────────────────────────────────
def parse_weather_and_update_reference(
    mos_file_path: str,
    sim_models_dir: str,
    formatted_id: str,
    start_date_str: str = "2021-01-01 00:00:00",
):
    """
    - Reads (time_s, temp) from mos (assumes temp in °C; adjust upstream if needed)
    - Creates a dataframe with datetime (rounded hour) + temp (°C)
    - Updates <sim_models_dir>/<formatted_id>/<formatted_id>.mo:
        filNam="...path/to/mos..."
      using a robust regex replacement.

    Returns:
      df_temperature, outdoor_temperature_data (list of tuples)
    """
    mos_file_path = str(Path(mos_file_path).resolve())
    model_mo_path = Path(sim_models_dir) / str(formatted_id) / f"{formatted_id}.mo"

    # --- parse mos ---
    with open(mos_file_path, "r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()

    outdoor_temperature_data = []
    for line in lines:
        s = line.strip()
        if not s or s.startswith("#") or s.startswith("double"):
            continue
        parts = s.split()
        if len(parts) < 2:
            continue
        try:
            t_sec = int(float(parts[0]))
            temp = float(parts[1])  # assumed °C
            outdoor_temperature_data.append((t_sec, temp))
        except Exception:
            continue

    df_temperature = pd.DataFrame(outdoor_temperature_data, columns=["time_in_seconds", "temp_celsius"])
    start_date = pd.Timestamp(start_date_str)
    if not df_temperature.empty:
        df_temperature["datetime"] = (start_date + pd.to_timedelta(df_temperature["time_in_seconds"], unit="s")).dt.round("H")
        df_temperature = df_temperature[["datetime", "temp_celsius"]].rename(columns={"temp_celsius": "temp"})
    else:
        df_temperature = pd.DataFrame(columns=["datetime", "temp"])

    # --- patch Modelica mo reference ---
    if model_mo_path.is_file():
        # Modelica string: use POSIX path for portability
        mos_for_mo = Path(mos_file_path).as_posix()

        pattern = re.compile(r'filNam=(?:Modelica\.Utilities\.Files\.loadResource\([^)]+\)|"[^"]+")')
        with open(model_mo_path, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read()
        content = pattern.sub(f'filNam="{mos_for_mo}"', content)
        with open(model_mo_path, "w", encoding="utf-8") as f:
            f.write(content)

    return df_temperature, outdoor_temperature_data


# ───────────────────────────────────────────────────────────────
# LPG pool reading
# ───────────────────────────────────────────────────────────────
def _to8760(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float).flatten()
    if x.size >= 8760:
        return x[:8760]
    rep = int(np.ceil(8760 / max(x.size, 1)))
    return np.tile(x, rep)[:8760]


def _read_sumprofiles_csv(csv_path: str) -> np.ndarray:
    """
    Reads SumProfiles_3600s.*.csv (semicolon-separated) and returns the first "Sum [...]" column as float array.
    """
    df = pd.read_csv(csv_path, sep=";", engine="python")
    sum_cols = [c for c in df.columns if isinstance(c, str) and c.startswith("Sum [")]
    if not sum_cols:
        raise ValueError(f"No 'Sum [...]' column in {csv_path}")
    return df[sum_cols[0]].astype(float).to_numpy()


def _results_dir_for(tp: str, template: str, r: int) -> Path:
    """
    Tries a fast expected path, then falls back to glob search.
    """
    fast = LPG_POOL_ROOT / tp / template / f"r{r}" / f"Results_{template}_{tp.replace('_','')}_r{r}" / "Results"
    if fast.is_dir():
        return fast

    patt = str(LPG_POOL_ROOT / tp / template / f"r{r}" / "**" / "Results")
    hits = [Path(h) for h in glob.glob(patt, recursive=True) if Path(h).name.lower() == "results"]
    if not hits:
        raise FileNotFoundError(f"No Results folder found: tp={tp}, template={template}, r={r}")
    return hits[0]


def _find_profile_csv(tp: str, template: str, r: int, hh_key: str, label: str) -> Optional[Path]:
    """
    Expects: Results/SumProfiles_3600s.HH1.<label>.csv
    Uses a glob fallback if exact name not found.
    """
    res_dir = _results_dir_for(tp, template, r)
    p = res_dir / f"SumProfiles_3600s.{hh_key}.{label}.csv"
    if p.is_file():
        return p

    patt = str(res_dir / f"SumProfiles_3600s.{hh_key}.*{label}*.csv")
    hits = [Path(h) for h in glob.glob(patt)]
    return hits[0] if hits else None


def _find_outside_json(tp: str, template: str, r: int, hh_key: str = HH_FIXED) -> Optional[Path]:
    res_dir = _results_dir_for(tp, template, r)
    p = res_dir / f"{LABEL_BAL_OUTSIDE}.{hh_key}.json"
    return p if p.is_file() else None


def _outside_json_to_inside_hourly_8760(json_path: Path, persons_fallback: int) -> Tuple[np.ndarray, int]:
    with open(json_path, "r", encoding="utf-8") as f:
        d = json.load(f)

    outside_min = np.asarray(d.get("Values", []), dtype=float)
    if outside_min.size == 0:
        return np.zeros(8760, dtype=float), max(1, int(persons_fallback))

    mx = float(np.nanmax(outside_min))
    persons = max(1, int(np.ceil(mx))) if np.isfinite(mx) and mx > 0 else max(1, int(persons_fallback))

    inside_min = np.clip(float(persons) - outside_min, 0.0, float(persons))

    n_hours_avail = int(inside_min.size // 60)
    if n_hours_avail > 0:
        inside_h = inside_min[: n_hours_avail * 60].reshape(n_hours_avail, 60).mean(axis=1)
    else:
        inside_h = np.zeros(1, dtype=float)

    return _to8760(inside_h), persons


def _draw_template(rng: np.random.Generator, cfg: LPGSelectionConfig) -> str:
    sizes = np.array(list(cfg.size_probs.keys()), dtype=int)
    probs = np.array([cfg.size_probs[s] for s in sizes], dtype=float)
    probs = probs / probs.sum()

    size = int(rng.choice(sizes, p=probs))
    candidates = [t for t, p in cfg.template_to_persons.items() if int(p) == size]
    if not candidates:
        candidates = list(cfg.template_to_persons.keys())
    return str(rng.choice(candidates))


def _choose_tp(rng: np.random.Generator, cfg: LPGSelectionConfig) -> str:
    if str(cfg.tp_mode).lower() == "fixed":
        return str(cfg.tp_fixed)
    return str(rng.choice(TP_CODES))


def _load_one_hh_profile(tp: str, template: str, r: int, hh_key: str, persons_hint: int) -> Dict[str, np.ndarray]:
    elec_p = _find_profile_csv(tp, template, r, hh_key, LABEL_ELEC)
    ww_p = _find_profile_csv(tp, template, r, hh_key, LABEL_WW) or _find_profile_csv(tp, template, r, hh_key, LABEL_HW)
    ig_p = _find_profile_csv(tp, template, r, hh_key, LABEL_IG)
    cw_p = _find_profile_csv(tp, template, r, hh_key, LABEL_CW)

    if elec_p is None or ww_p is None or ig_p is None:
        raise FileNotFoundError(f"Missing profiles: tp={tp}, template={template}, r={r}, hh={hh_key}")

    elec = _to8760(_read_sumprofiles_csv(str(elec_p)))  # kWh/h (kW)
    ww = _to8760(_read_sumprofiles_csv(str(ww_p)))      # depends on export
    ig = _to8760(_read_sumprofiles_csv(str(ig_p)))      # kWh/h (kW)
    cw = np.zeros(8760, dtype=float) if cw_p is None else _to8760(_read_sumprofiles_csv(str(cw_p)))

    outside_json = _find_outside_json(tp, template, r, hh_key)
    if outside_json is None:
        occ_inside = np.zeros(8760, dtype=float)
        persons = max(1, int(persons_hint))
    else:
        occ_inside, persons = _outside_json_to_inside_hourly_8760(outside_json, persons_fallback=persons_hint)

    return {
        "electricity_demand": elec,
        "warm_water_demand": ww,
        "cold_water_demand": cw,
        "internal_gains": ig,
        "occupancy_inside": occ_inside,  # persons/h
        "persons": np.array([persons], dtype=float),
    }


def build_lpg_apartments_year(seed: int, cfg: LPGSelectionConfig) -> Dict[str, np.ndarray]:
    """
    Returns apartment-wise profiles:
      - occupancy_abs: (A, 8760) persons/h
      - machines:      (A, 8760) internal gains (kW by default; set MACHINES_SCALE=1000 if model expects W)
      - persons:       (A,)
      - occupancy_rel: (A, 8760) 0..1
    """
    rng = np.random.default_rng(int(seed))

    occ_list: List[np.ndarray] = []
    mach_list: List[np.ndarray] = []
    persons_list: List[int] = []

    for _ in range(int(cfg.n_apartments)):
        tp = _choose_tp(rng, cfg)
        template = _draw_template(rng, cfg)
        persons_hint = int(cfg.template_to_persons.get(template, 1))
        hh_key = HH_FIXED

        if str(cfg.lpg_seed_mode).lower() == "mean_r":
            acc_occ = np.zeros(8760, dtype=float)
            acc_mach = np.zeros(8760, dtype=float)
            persons_ap: Optional[int] = None
            count = 0

            for r in cfg.r_values:
                p = _load_one_hh_profile(tp, template, int(r), hh_key, persons_hint)
                acc_occ += np.asarray(p["occupancy_inside"], dtype=float)
                acc_mach += np.asarray(p["internal_gains"], dtype=float)
                if persons_ap is None:
                    persons_ap = int(p["persons"][0])
                count += 1

            if count <= 0:
                raise FileNotFoundError(f"No seeds loaded (mean_r): tp={tp}, template={template}")

            acc_occ /= float(count)
            acc_mach /= float(count)
            if persons_ap is None:
                persons_ap = persons_hint

            occ_list.append(acc_occ)
            mach_list.append(acc_mach * float(MACHINES_SCALE))
            persons_list.append(int(persons_ap))

        else:
            r = int(rng.choice(cfg.r_values))
            p = _load_one_hh_profile(tp, template, r, hh_key, persons_hint)
            occ_list.append(np.asarray(p["occupancy_inside"], dtype=float))
            mach_list.append(np.asarray(p["internal_gains"], dtype=float) * float(MACHINES_SCALE))
            persons_list.append(int(p["persons"][0]))

    occ = np.stack(occ_list, axis=0)                 # (A, 8760)
    mach = np.stack(mach_list, axis=0)               # (A, 8760)
    persons = np.asarray(persons_list, dtype=float)  # (A,)

    persons_safe = np.where(persons > 0, persons, 1.0)
    occ_rel = np.clip(occ / persons_safe[:, None], 0.0, 1.0)

    return {
        "occupancy_abs": occ,
        "machines": mach,
        "persons": persons,
        "occupancy_rel": occ_rel,
    }