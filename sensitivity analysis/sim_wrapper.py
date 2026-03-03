import os
import json
import time
import uuid
import shutil
import pathlib
import re
import csv
import multiprocessing
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List, Tuple

import numpy as np
from ebcpy import DymolaAPI, TimeSeriesData

# >>> utils import
from utils import (
    LPGSelectionConfig,
    building_model_exists,
    parse_weather_and_update_reference,
    to_dashed_id,
    build_lpg_apartments_year,
)

OUTPUT_INTERVAL_SEC = 60 * 60


# ───────────────────────────────────────────────────────────────
# SA Parameter
# ───────────────────────────────────────────────────────────────
@dataclass
class SAParams:
    # Setpoints (controlled via zone_control/payload; kept for interface compatibility)
    tset_mean_K: float = 293.15
    tset_spread_K: float = 1.0

    gains_scale: float = 1.0

    # optional (currently not used for LPG mapping but kept)
    zone_weights: Optional[List[float]] = None

    rng_seed: int = 1234
    lpg_cfg: Optional[LPGSelectionConfig] = None

    enable_cooling: bool = True
    th_zone_index: int = 0

    th_people_factor: float = 0.1
    th_lights_factor: float = 0.1
    th_occ_rel_factor: float = 0.1
    th_machines_factor: float = 0.0

    record_overrides_global: Dict[str, Any] = field(default_factory=dict)
    record_overrides_by_zone: Optional[List[Dict[str, Any]]] = None


# ───────────────────────────────────────────────────────────────
# Helpers: records + zones
# ───────────────────────────────────────────────────────────────
def _read_lines(path: str) -> List[str]:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.readlines()


def find_zone_record_files(sim_models_dir: str, formatted_id: str) -> list[tuple[int, str]]:
    """
    Returns list [(zone_number_in_filename, filepath), ...] sorted by zone_number_in_filename.
    Expects files like: <formatted_id>_Storey_1_Zone_2.mo
    """
    record_dir = os.path.join(sim_models_dir, formatted_id, f"{formatted_id}_DataBase")
    if not os.path.isdir(record_dir):
        return []

    patt = re.compile(rf"^{re.escape(formatted_id)}_.*_Zone_(\d+)\.mo$", re.IGNORECASE)
    out: list[tuple[int, str]] = []
    for fn in os.listdir(record_dir):
        if not fn.lower().endswith(".mo"):
            continue
        m = patt.match(fn)
        if not m:
            continue
        z = int(m.group(1))
        out.append((z, os.path.join(record_dir, fn)))

    out.sort(key=lambda t: t[0])
    return out


def _infer_n_zones_from_any_record(sim_models_dir: str, formatted_id: str) -> int:
    """
    Robustly infer n_zones by reading nNZs from any Zone_*.mo record.
    Fallback: len(zfiles) or 1.
    """
    zfiles = find_zone_record_files(sim_models_dir, formatted_id)
    if not zfiles:
        return 1

    _, fp = zfiles[0]
    lines = _read_lines(fp)

    for line in lines:
        s = line.strip()
        if s.startswith("nNZs"):
            try:
                return max(1, int(s.split("=")[1].strip().rstrip(",").rstrip(");")))
            except Exception:
                pass
    return max(1, len(zfiles))


def _zone_name_from_record_filename(formatted_id: str, record_fp: str) -> str:
    """
    Extracts zone name from record filename:
      <formatted_id>_<ZONE-NAME>.mo  -> returns <ZONE-NAME>
    """
    fn = os.path.basename(record_fp)
    if fn.lower().endswith(".mo"):
        fn = fn[:-3]
    prefix = f"{formatted_id}_"
    if fn.startswith(prefix):
        return fn[len(prefix):]
    return fn


# ───────────────────────────────────────────────────────────────
# Outputs: build list of Dymola variables to export
# ───────────────────────────────────────────────────────────────
def build_selected_parameters(n_zones: int) -> List[Tuple[str, str]]:
    """
    Returns list of (dymola_var, csv_column) depending on n_zones.
    Includes weather bus + zonal vectors [1..n_zones].
    """
    base: List[Tuple[str, str]] = [
        ("weaDat.weaBus.TDryBul", "temp_air"),
        ("weaDat.weaBus.TDewPoi", "temp_dew"),
        ("weaDat.weaBus.relHum",  "relHum"),
        ("weaDat.weaBus.HDirNor", "HDirNor"),
        ("weaDat.weaBus.HDifHor", "dhi"),
        ("weaDat.weaBus.HGloHor", "ghi"),
        ("weaDat.weaBus.winSpe",  "wind_speed"),
        ("weaDat.weaBus.pAtm",    "pressure"),
    ]

    for z in range(1, int(n_zones) + 1):
        base.append((f"multizone.TSetHeat[{z}]", f"TSetHeat_{z}"))
        base.append((f"multizone.TSetCool[{z}]", f"TSetCool_{z}"))
        base.append((f"multizone.TAir[{z}]",     f"TAir_{z}"))
        base.append((f"multizone.TRad[{z}]",     f"TRad_{z}"))
        base.append((f"multizone.PHeater[{z}]",  f"HeatDemand_{z}"))
        base.append((f"multizone.PCooler[{z}]",  f"CoolDemand_{z}"))
        base.append((f"multizone.QIntGains_flow[{z},1]", f"GainsLights_{z}"))
        base.append((f"multizone.QIntGains_flow[{z},2]", f"GainsMachines_{z}"))
        base.append((f"multizone.QIntGains_flow[{z},3]", f"GainsHumans_{z}"))

    # optional (only if model contains these variables)
    base.append(("integrator.y", "HeatDemand_SUM"))
    base.append(("integrator.u", "HeatDemand"))
    base.append(("tableInternalGains.y[1]", "occupancy_abs"))
    base.append(("tableInternalGains.y[4]", "occupancy_rel"))

    return base


# ───────────────────────────────────────────────────────────────
# Record patching: WWR scaling
# ───────────────────────────────────────────────────────────────
def _scale_array_assignment_in_mo(content: str, key: str, factor: float) -> str:
    """
    Scales array assignments like:
        AWin = {1,2,3}
    ->  AWin = {factor*1, factor*2, factor*3}
    """
    pattern = re.compile(rf"(\b{re.escape(key)}\b\s*=\s*)\{{([^}}]+)\}}", re.MULTILINE)

    def repl(m):
        prefix = m.group(1)
        inner = m.group(2).strip()
        parts = [p.strip() for p in inner.split(",") if p.strip()]
        vals: List[float] = []
        for p in parts:
            try:
                vals.append(float(p))
            except Exception:
                # if parsing fails, keep original text
                return m.group(0)
        scaled = [v * float(factor) for v in vals]
        inner_scaled = ", ".join(f"{v:.15g}" for v in scaled)
        return f"{prefix}{{{inner_scaled}}}"

    return pattern.sub(repl, content)


def apply_wwr_factor_to_zone_records(sim_models_dir: str, formatted_id: str, wwr_factor: float) -> int:
    """
    Scales AWin and ATransparent in all zone records.
    Returns number of patched files.
    """
    zfiles = find_zone_record_files(sim_models_dir, formatted_id)
    if not zfiles:
        return 0

    patched = 0
    for _znum, fp in zfiles:
        with open(fp, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read()

        new_content = content
        new_content = _scale_array_assignment_in_mo(new_content, "AWin", float(wwr_factor))
        new_content = _scale_array_assignment_in_mo(new_content, "ATransparent", float(wwr_factor))

        if new_content != content:
            with open(fp, "w", encoding="utf-8") as f:
                f.write(new_content)
            patched += 1

    return patched


# ───────────────────────────────────────────────────────────────
# Record patching: generic key=value overrides
# ───────────────────────────────────────────────────────────────
def apply_record_overrides_regex(record_file_path: str, overrides: Dict[str, Any]) -> None:
    """
    Replaces assignments in record files:
        key = <value>
    Keeps formatting fairly stable; only replaces the RHS value.
    """
    if not overrides:
        return

    with open(record_file_path, "r", encoding="utf-8", errors="ignore") as f:
        content = f.read()

    def fmt(v: Any) -> str:
        if isinstance(v, bool):
            return "true" if v else "false"
        if isinstance(v, (int, float, np.integer, np.floating)):
            return str(float(v)) if isinstance(v, (np.floating,)) else str(v)
        if isinstance(v, str):
            if v.startswith('"') and v.endswith('"'):
                return v
            return f'"{v}"'
        if isinstance(v, (list, tuple, np.ndarray)):
            arr = np.asarray(v).flatten()
            return "{" + ", ".join(str(float(x)) for x in arr) + "}"
        return str(v)

    for k, v in overrides.items():
        pattern = re.compile(rf"(\b{re.escape(k)}\b\s*=\s*)([^,\)\n]+)")
        content = pattern.sub(rf"\1{fmt(v)}", content)

    with open(record_file_path, "w", encoding="utf-8") as f:
        f.write(content)


def default_lpg_flags_for_zone(is_th: bool) -> Dict[str, Any]:
    if is_th:
        return {"use_lpg_people": False, "use_lpg_machines": False, "use_lpg_light": False}
    return {"use_lpg_people": True, "use_lpg_machines": True, "use_lpg_light": False}


def build_effective_zone_overrides(
    n_zones: int,
    th_zone_index: int,
    global_overrides: Dict[str, Any],
    by_zone: Optional[List[Dict[str, Any]]],
) -> List[Dict[str, Any]]:
    """
    Override precedence per zone:
      defaults(lpg flags) -> global_overrides -> by_zone[zi]
    """
    if n_zones <= 0:
        return []

    th = max(0, min(int(th_zone_index), n_zones - 1))

    out: List[Dict[str, Any]] = []
    for zi in range(n_zones):
        d: Dict[str, Any] = {}
        d.update(default_lpg_flags_for_zone(is_th=(zi == th)))
        d.update(global_overrides or {})
        if by_zone is not None and zi < len(by_zone) and by_zone[zi]:
            d.update(by_zone[zi])
        out.append(d)
    return out


def apply_zone_record_overrides(sim_models_dir: str, formatted_id: str, sa_params: SAParams) -> int:
    """
    Patches all Zone_*.mo records with overrides.
    Zone order matches sorted Zone_* file list.
    Returns number of zone records found (and processed).
    """
    zfiles = find_zone_record_files(sim_models_dir, formatted_id)
    if not zfiles:
        return 0

    n = len(zfiles)
    zone_ov = build_effective_zone_overrides(
        n_zones=n,
        th_zone_index=sa_params.th_zone_index,
        global_overrides=sa_params.record_overrides_global,
        by_zone=sa_params.record_overrides_by_zone,
    )

    for zi, (_zone_nr, fp) in enumerate(zfiles):
        apply_record_overrides_regex(fp, zone_ov[zi])

    return n


# ───────────────────────────────────────────────────────────────
# Setpoints tables (constant over 8760h)
# ───────────────────────────────────────────────────────────────
def write_setpoints_multizone(
    time_table_T_set: str,
    time_table_T_set_cool: str,
    full_year_hours: int,
    n_zones: int,
    enable_cooling: bool = True,
    heat_vals_K: Optional[np.ndarray] = None,
    cool_vals_K: Optional[np.ndarray] = None,
):
    """
    Writes constant setpoints (8760h) as Dymola text tables.
    If heat_vals_K/cool_vals_K are provided: uses them.
    """
    if n_zones <= 0:
        raise ValueError("n_zones must be >= 1")

    if heat_vals_K is None:
        heat_vals_K = np.full(n_zones, 293.15, dtype=float)
        if n_zones > 1:
            heat_vals_K[0] = 281.15  # TH frost protection

    if cool_vals_K is None:
        cool_vals_K = np.full(n_zones, 273.15 + 60.0, dtype=float)  # cooling off

    heat_vals_K = np.asarray(heat_vals_K, dtype=float).flatten()
    cool_vals_K = np.asarray(cool_vals_K, dtype=float).flatten()

    if heat_vals_K.size != n_zones:
        raise ValueError(f"heat_vals_K must have length {n_zones}, got {heat_vals_K.size}")
    if cool_vals_K.size != n_zones:
        raise ValueError(f"cool_vals_K must have length {n_zones}, got {cool_vals_K.size}")

    if not enable_cooling:
        cool_vals_K[:] = 273.15 + 60.0

    def _write(path: str, table_name: str, vals: np.ndarray):
        with open(path, "w", encoding="utf-8") as f:
            f.write(f"#1\ndouble {table_name}({full_year_hours}, {1+n_zones})\n")
            for h in range(full_year_hours):
                sec = h * 3600
                row = [f"{sec}"] + [f"{vals[z]:.2f}" for z in range(n_zones)]
                f.write("\t".join(row) + "\n")

    _write(time_table_T_set, "Tset", heat_vals_K)
    _write(time_table_T_set_cool, "Tset", cool_vals_K)


def _resolve_zone_control_value(zone_name: str, zone_control: Optional[dict]) -> dict:
    """
    Mirror of TEASER-side zone_control resolution (simplified):
    - exact match in zone_control["zones"] else zone_control["default"]
    """
    if not zone_control:
        return {"heated": True, "heat_setpoint_K": 294.15, "cooled": False, "cool_setpoint_K": 0.0}

    zones = zone_control.get("zones", {}) or {}
    if zone_name in zones:
        zc = dict(zones[zone_name])
    else:
        zc = dict(zone_control.get("default", {}) or {})

    heated = bool(zc.get("heated", True))
    cooled = bool(zc.get("cooled", False))

    heat_sp_K = float(zc.get("heat_setpoint_K", 294.15))
    cool_sp_K = float(zc.get("cool_setpoint_K", 299.15))

    return {"heated": heated, "heat_setpoint_K": heat_sp_K, "cooled": cooled, "cool_setpoint_K": cool_sp_K}


def _setpoints_from_zone_control(
    sim_models_dir: str,
    formatted_id: str,
    zone_control: Optional[dict],
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Builds (heat_vals_K, cool_vals_K) in the order of zone record files.
    Matches by record filename -> zone_name.
    """
    zfiles = find_zone_record_files(sim_models_dir, formatted_id)
    if not zfiles:
        return np.array([293.15], dtype=float), np.array([273.15 + 60.0], dtype=float)

    heat = np.zeros(len(zfiles), dtype=float)
    cool = np.zeros(len(zfiles), dtype=float)

    for i, (_znum, fp) in enumerate(zfiles):
        zname = _zone_name_from_record_filename(formatted_id, fp)
        zc = _resolve_zone_control_value(zname, zone_control)
        heat[i] = float(zc["heat_setpoint_K"]) if zc.get("heated", True) else 0.0
        cool[i] = float(zc.get("cool_setpoint_K", 299.15)) if zc.get("cooled", False) else (273.15 + 60.0)

    return heat, cool


# ───────────────────────────────────────────────────────────────
# TEASER build/export
# ───────────────────────────────────────────────────────────────
def ensure_teaser_model(
    sim_models_dir: str,
    formatted_id: str,
    building_payload: Dict[str, Any],
    mos_file_path: str,
    zone_control: Optional[dict],
    force_rebuild: bool = False,
) -> None:
    """
    Builds/overwrites the TEASER/AixLib export model for this formatted_id in sim_models_dir.
    - uses YoC via building_payload["building_data"]["sa_tabula_year_class"]
    - uses zone_control (sa_zone_control) for zonal HVAC setpoints
    """
    if (not force_rebuild) and building_model_exists(formatted_id, sim_models_dir):
        return

    bdir = os.path.join(sim_models_dir, formatted_id)
    if os.path.isdir(bdir):
        shutil.rmtree(bdir, ignore_errors=True)
    os.makedirs(sim_models_dir, exist_ok=True)

    from teaser.project import Project
    from teaser_export import create_teaser_project

    prj = Project(load_data=True)
    tabula_df = None

    if zone_control is None:
        zone_control = (building_payload.get("building_data", {}) or {}).get("sa_zone_control")

    # stabilize export folder/name: TEASER uses building_info["building_id"]
    payload_for_export = dict(building_payload)
    payload_for_export["building_id"] = str(formatted_id)

    create_teaser_project(
        building_info=payload_for_export,
        teaser_project=prj,
        weather_path=mos_file_path,
        tabula_df=tabula_df,
        construction=None,
        project_name=str(formatted_id),
        zone_control=zone_control,
        use_lpg_templates=False,
    )

    try:
        prj.export_aixlib(path=sim_models_dir)
    except TypeError:
        prj.export_aixlib(sim_models_dir)

    if not building_model_exists(formatted_id, sim_models_dir):
        raise RuntimeError(f"TEASER export failed: {formatted_id} not found in {sim_models_dir}")


# ───────────────────────────────────────────────────────────────
# Internal gains table (multizone) writing
# ───────────────────────────────────────────────────────────────
def write_internal_gains_multizone_table_from_zone_series(
    combi_time_table_path: str,
    full_year_hours: int,
    people_z: np.ndarray,     # (n_zones, 8760)
    machines_z: np.ndarray,   # (n_zones, 8760)
    lights_z: np.ndarray,     # (n_zones, 8760)
    occ_rel_z: np.ndarray,    # (n_zones, 8760)
):
    n_zones = int(people_z.shape[0])
    n_cols = 1 + 4 * n_zones

    with open(combi_time_table_path, "w", encoding="utf-8") as f:
        f.write(f"#1\ndouble Internals({full_year_hours}, {n_cols})\n")
        for h in range(full_year_hours):
            sec = h * 3600
            row = [f"{sec}"]
            for z in range(n_zones):
                row += [
                    f"{people_z[z, h]:.2f}",
                    f"{machines_z[z, h]:.2f}",
                    f"{lights_z[z, h]:.2f}",
                    f"{occ_rel_z[z, h]:.2f}",
                ]
            f.write("\t".join(row) + "\n")


# ───────────────────────────────────────────────────────────────
# Timeseries CSV export from TimeSeriesData
# ───────────────────────────────────────────────────────────────
def _tsd_time_vector(tsd: TimeSeriesData) -> np.ndarray:
    """
    Robust extraction of a time axis.
    """
    if hasattr(tsd, "time"):
        t = np.asarray(getattr(tsd, "time"))
        if t.size > 0:
            return t

    try:
        any_key = next(iter(tsd.keys()))
        t = np.asarray(tsd[any_key].index)
        return t
    except Exception:
        return np.array([], dtype=float)


def write_timeseries_csv(
    tsd: TimeSeriesData,
    selected: List[Tuple[str, str]],
    out_csv_path: str,
) -> Dict[str, Any]:
    """
    Writes CSV: time + all selected signals.
    Missing signals => NaN column.
    Returns metadata.
    """
    time_vec = _tsd_time_vector(tsd)

    if time_vec.size == 0:
        for var, _ in selected:
            try:
                s = tsd[var]
                time_vec = np.asarray(getattr(s, "index", np.arange(len(s.values))))
                break
            except Exception:
                continue

    n = int(time_vec.size)
    cols = ["time_s"] + [col for _, col in selected]

    data: Dict[str, np.ndarray] = {}
    missing: List[str] = []

    for var, col in selected:
        try:
            s = tsd[var]
            v = np.asarray(s.values).flatten()
            if v.size != n:
                if v.size > n:
                    v = v[:n]
                else:
                    vv = np.full(n, np.nan, dtype=float)
                    vv[:v.size] = v
                    v = vv
            data[col] = v
        except Exception:
            data[col] = np.full(n, np.nan, dtype=float)
            missing.append(var)

    os.makedirs(os.path.dirname(out_csv_path), exist_ok=True)
    with open(out_csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for i in range(n):
            row = [float(time_vec[i])]
            for _, col in selected:
                val = data[col][i]
                row.append("" if (val is None or (isinstance(val, float) and np.isnan(val))) else float(val))
            w.writerow(row)

    return {"csv_path": out_csv_path, "n_rows": n, "columns": cols, "missing_variables": missing}


# ───────────────────────────────────────────────────────────────
# Single run
# ───────────────────────────────────────────────────────────────
def simulate_one(
    out_dir: str,
    sim_models_dir: str,
    sim_model_pkg_name: str,
    aixlib_mo: str,
    building_id: str,
    mos_file_path: str,
    year: int,
    start_sim: int,
    end_sim: int,
    lpg_results_path: str,  # kept in interface
    sa_params: SAParams,
    internal_gains_mode: str = "multizone_table",
    # from head
    building_payload: Optional[Dict[str, Any]] = None,
    zone_control: Optional[dict] = None,
    task_meta: Optional[Dict[str, Any]] = None,
    # rebuild behavior
    force_teaser_rebuild: bool = True,
):
    json_id = to_dashed_id(building_id)
    formatted_id = json_id.replace("-", "")

    out_dir = pathlib.Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 0) TEASER model build (new per run by default)
    if building_payload is not None:
        ensure_teaser_model(
            sim_models_dir=sim_models_dir,
            formatted_id=formatted_id,
            building_payload=building_payload,
            mos_file_path=mos_file_path,
            zone_control=zone_control,
            force_rebuild=bool(force_teaser_rebuild),
        )

    if not building_model_exists(formatted_id, sim_models_dir):
        raise FileNotFoundError(f"No sim model for {formatted_id} in {sim_models_dir}")

    run_id = f"{int(time.time()*1000)}_{os.getpid()}_{uuid.uuid4().hex[:6]}"
    work_dir = out_dir / "work" / run_id
    work_dir.mkdir(parents=True, exist_ok=True)

    # Weather prepare (side-effects: updates record references etc.)
    parse_weather_and_update_reference(
        mos_file_path=mos_file_path,
        sim_models_dir=sim_models_dir,
        formatted_id=formatted_id,
    )

    combi_time_table_path = os.path.join(sim_models_dir, formatted_id, f"InternalGains_{formatted_id}.txt")
    time_table_T_set      = os.path.join(sim_models_dir, formatted_id, f"TsetHeat_{formatted_id}.txt")
    time_table_T_set_cool = os.path.join(sim_models_dir, formatted_id, f"TsetCool_{formatted_id}.txt")

    full_year_hours = 8760
    n_zones = _infer_n_zones_from_any_record(sim_models_dir, formatted_id)

    # 1) LPG profile generation
    if sa_params.lpg_cfg is None:
        raise ValueError("sa_params.lpg_cfg is None (LPGSelectionConfig required).")

    apt = build_lpg_apartments_year(seed=int(sa_params.rng_seed), cfg=sa_params.lpg_cfg)
    occ_ap = apt["occupancy_abs"]     # (A, 8760)
    mach_ap = apt["machines"]         # (A, 8760)
    occ_rel_ap = apt["occupancy_rel"] # (A, 8760)
    A = int(occ_ap.shape[0])

    lighting_profile = np.asarray(
        np.tile([0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0], 365),
        dtype=float
    )[:full_year_hours]

    # TH series
    th_people = np.full(full_year_hours, float(sa_params.th_people_factor))
    th_lights = float(sa_params.th_lights_factor) * lighting_profile
    th_machines = float(sa_params.th_machines_factor) * np.sum(mach_ap, axis=0)
    th_occ_rel = float(sa_params.th_occ_rel_factor) * np.clip(np.sum(occ_rel_ap, axis=0) / max(A, 1), 0, 1)

    # Apartment -> Zone mapping (best effort)
    if n_zones == 1:
        people_z = np.sum(occ_ap, axis=0, keepdims=True)
        machines_z = np.sum(mach_ap, axis=0, keepdims=True) * float(sa_params.gains_scale)
        lights_z = lighting_profile[None, :]
        occ_rel_z = np.clip(np.mean(occ_rel_ap, axis=0, keepdims=True), 0, 1)
    elif n_zones == 2:
        people_z = np.vstack([
            th_people[None, :],
            np.sum(occ_ap, axis=0, keepdims=True) * (1.0 - float(sa_params.th_people_factor)),
        ])
        machines_z = np.vstack([
            th_machines[None, :],
            np.sum(mach_ap, axis=0, keepdims=True) * float(sa_params.gains_scale),
        ])
        lights_z = np.vstack([
            th_lights[None, :],
            lighting_profile[None, :] * (1.0 - float(sa_params.th_lights_factor)),
        ])
        occ_rel_z = np.vstack([
            th_occ_rel[None, :],
            np.clip(np.mean(occ_rel_ap, axis=0, keepdims=True), 0, 1),
        ])
    else:
        n_res = n_zones - 1
        idx = np.arange(n_res) % max(A, 1)
        people_res = occ_ap[idx, :]
        machines_res = mach_ap[idx, :] * float(sa_params.gains_scale)
        occ_rel_res = occ_rel_ap[idx, :]
        lights_res = np.tile(lighting_profile[None, :], (n_res, 1))

        people_z = np.vstack([th_people[None, :], people_res])
        machines_z = np.vstack([th_machines[None, :], machines_res])
        lights_z = np.vstack([th_lights[None, :], lights_res])
        occ_rel_z = np.vstack([th_occ_rel[None, :], occ_rel_res])

    write_internal_gains_multizone_table_from_zone_series(
        combi_time_table_path=combi_time_table_path,
        full_year_hours=full_year_hours,
        people_z=people_z,
        machines_z=machines_z,
        lights_z=lights_z,
        occ_rel_z=occ_rel_z,
    )

    # 2) Setpoints from zone_control
    if zone_control is None and building_payload is not None:
        zone_control = (building_payload.get("building_data", {}) or {}).get("sa_zone_control")

    heat_vals_K, cool_vals_K = _setpoints_from_zone_control(sim_models_dir, formatted_id, zone_control)

    write_setpoints_multizone(
        time_table_T_set=time_table_T_set,
        time_table_T_set_cool=time_table_T_set_cool,
        full_year_hours=full_year_hours,
        n_zones=n_zones,
        enable_cooling=bool(sa_params.enable_cooling),
        heat_vals_K=heat_vals_K,
        cool_vals_K=cool_vals_K,
    )

    # 3) WWR patching via record_overrides_global["wwr_factor"]
    wwr_factor = float(sa_params.record_overrides_global.get("wwr_factor", 1.0))
    wwr_patched_files = apply_wwr_factor_to_zone_records(sim_models_dir, formatted_id, wwr_factor)

    # 4) Generic record overrides (also writes default LPG flags per zone)
    patched = apply_zone_record_overrides(sim_models_dir, formatted_id, sa_params)

    # Copy model export for later inspection (per run)
    model_src = os.path.join(sim_models_dir, formatted_id)
    model_dst = os.path.join(str(out_dir), "model_export")
    if os.path.isdir(model_dst):
        shutil.rmtree(model_dst, ignore_errors=True)
    shutil.copytree(model_src, model_dst)

    # 5) Dymola simulation
    teaser_mo = os.path.join(sim_models_dir, "package.mo")
    model_name = f"{sim_model_pkg_name}.{formatted_id}.{formatted_id}"

    dym_api = DymolaAPI(
        working_directory=str(work_dir),
        model_name=model_name,
        n_cpu=1,
        packages=[str(aixlib_mo), str(teaser_mo)],
        show_window=False,
        equidistant_output=False,
    )
    dym_api.set_sim_setup({
        "start_time": int(start_sim),
        "stop_time": int(end_sim),
        "output_interval": int(OUTPUT_INTERVAL_SEC),
    })

    result_path = dym_api.simulate(return_option="savepath")
    tsd = TimeSeriesData(result_path)

    # Sum loads across zones
    heat_sum_W = np.zeros_like(tsd["multizone.PHeater[1]"].values.flatten(), dtype=float)
    cool_sum_W = np.zeros_like(tsd["multizone.PCooler[1]"].values.flatten(), dtype=float)
    for z in range(1, n_zones + 1):
        heat_sum_W += tsd[f"multizone.PHeater[{z}]"].values.flatten()
        cool_sum_W += tsd[f"multizone.PCooler[{z}]"].values.flatten()

    # Save zone map (record filenames => zone labels)
    zfiles = find_zone_record_files(sim_models_dir, formatted_id)
    zone_labels = [_zone_name_from_record_filename(formatted_id, fp) for _, fp in zfiles]
    with open(out_dir / "zone_map.json", "w", encoding="utf-8") as f:
        json.dump({"n_zones": int(n_zones), "zones": zone_labels}, f, indent=2, ensure_ascii=False)

    # Save timeseries
    selected = build_selected_parameters(n_zones=n_zones)
    ts_csv_path = str(out_dir / "timeseries.csv")
    ts_meta = write_timeseries_csv(tsd, selected, ts_csv_path)

    # Aggregated KPIs
    heat_kWh = float(np.trapz(heat_sum_W, dx=OUTPUT_INTERVAL_SEC) / 3.6e6)
    cool_kWh = float(np.trapz(cool_sum_W, dx=OUTPUT_INTERVAL_SEC) / 3.6e6)
    peak_heat_kW = float(np.max(heat_sum_W) / 1000.0)
    peak_cool_kW = float(np.max(cool_sum_W) / 1000.0)

    out = {
        "building_id": str(json_id),
        "formatted_id": str(formatted_id),
        "n_zones": int(n_zones),

        "patched_zone_records": int(patched),
        "wwr_factor": float(wwr_factor),
        "wwr_patched_files": int(wwr_patched_files),

        "heat_demand_kWh": heat_kWh,
        "cool_demand_kWh": cool_kWh,
        "peak_heat_kW": peak_heat_kW,
        "peak_cool_kW": peak_cool_kW,

        "lpg_cfg": {
            "n_apartments": int(sa_params.lpg_cfg.n_apartments),
            "tp_mode": str(sa_params.lpg_cfg.tp_mode),
            "tp_fixed": str(sa_params.lpg_cfg.tp_fixed),
            "lpg_seed_mode": str(sa_params.lpg_cfg.lpg_seed_mode),
            "r_values": list(sa_params.lpg_cfg.r_values),
        },
        "sa_params": {
            "gains_scale": float(sa_params.gains_scale),
            "rng_seed": int(sa_params.rng_seed),
            "enable_cooling": bool(sa_params.enable_cooling),
            "th_people_factor": float(sa_params.th_people_factor),
            "th_lights_factor": float(sa_params.th_lights_factor),
            "th_occ_rel_factor": float(sa_params.th_occ_rel_factor),
            "th_machines_factor": float(sa_params.th_machines_factor),
            "record_overrides_global": dict(sa_params.record_overrides_global),
            "record_overrides_by_zone": sa_params.record_overrides_by_zone,
        },

        "zone_control_used": zone_control,
        "task_meta": task_meta,
        "mos_file": str(mos_file_path),
        "start_sim": int(start_sim),
        "end_sim": int(end_sim),
        "timeseries_csv": ts_meta,
        "model_export_dir": str(out_dir / "model_export"),
    }

    with open(out_dir / "overall.json", "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    dym_api.close()
    shutil.rmtree(work_dir, ignore_errors=True)

    return out


# ───────────────────────────────────────────────────────────────
# Batch runner
# ───────────────────────────────────────────────────────────────
def _worker(task: Dict[str, Any]):
    return simulate_one(**task)


def run_many(tasks: List[Dict[str, Any]], n_proc: int = 1):
    if n_proc == 1:
        return [_worker(t) for t in tasks]

    ctx = multiprocessing.get_context("spawn")
    with ctx.Pool(processes=n_proc, maxtasksperchild=1) as pool:
        return pool.map(_worker, tasks)