# sa_sim_runner.py
import os, json, time, uuid, shutil, pathlib, re
import numpy as np
import multiprocessing
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List, Tuple

from ebcpy import DymolaAPI, TimeSeriesData

# >>> utils import (Modulname ggf. anpassen)
from utils import (
    LPGSelectionConfig,
    build_lpg_building_year,
    building_model_exists,
    parse_weather_and_update_reference,
    to_dashed_id,
    build_lpg_apartments_year
)

OUTPUT_INTERVAL_SEC = 60 * 60


# ───────────────────────────────────────────────────────────────
# SA Parameter
# ───────────────────────────────────────────────────────────────
@dataclass
class SAParams:
    # Setpoints (Interface bleibt; konkrete Logik unten)
    tset_mean_K: float = 293.15
    tset_spread_K: float = 1.0

    # Internal gains scaling
    gains_scale: float = 1.0

    # Verteilung der WOHNZONEN (Zonen 1..n-1). Falls None => gleich.
    # Wenn du hier n_zones Werte gibst, wird Zone 0 ignoriert.
    zone_weights: Optional[List[float]] = None

    # Reproduzierbarkeit
    rng_seed: int = 1234

    # LPG Konfig (Pool-Auswahl)
    lpg_cfg: Optional[LPGSelectionConfig] = None

    # Cooling komplett abschaltbar
    enable_cooling: bool = True

    # TH-Index (im Modelica-Multizone immer Zone 0)
    th_zone_index: int = 0

    # TH Anteile (wie von dir gefordert)
    th_people_factor: float = 0.1
    th_lights_factor: float = 0.1
    th_occ_rel_factor: float = 0.1   # setz 0.0 wenn du es komplett aus willst
    th_machines_factor: float = 0.0

    # Record Overrides
    # global: wird auf alle Zonen gelegt, danach pro-zone überschrieben
    record_overrides_global: Dict[str, Any] = field(default_factory=dict)
    # per zone (Index 0..n-1): überschreibt global & defaults
    record_overrides_by_zone: Optional[List[Dict[str, Any]]] = None


# ───────────────────────────────────────────────────────────────
# Helpers
# ───────────────────────────────────────────────────────────────
def _normalize_weights(n: int, w: Optional[List[float]]) -> List[float]:
    """
    Erwartet i.d.R. Gewichte für alle Zonen.
    Für die Wohnzonen-Normalisierung wird später Zone 0 rausgenommen.
    """
    if n <= 0:
        return []
    if w is None or len(w) != n:
        return [1.0 / n] * n
    s = float(sum(w))
    if not np.isfinite(s) or s <= 0:
        return [1.0 / n] * n
    return [float(x) / s for x in w]


def _read_lines(path: str) -> List[str]:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.readlines()


def _infer_n_zones_from_any_record(sim_models_dir: str, formatted_id: str) -> int:
    """
    Robust: Nimmt irgendeinen Zone_*.mo Record und liest nNZs.
    Fallback: 1
    """
    zfiles = find_zone_record_files(sim_models_dir, formatted_id)
    if not zfiles:
        return 1

    # irgendeinen nehmen (z.B. den ersten)
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


def find_zone_record_files(sim_models_dir: str, formatted_id: str) -> list[tuple[int, str]]:
    """
    Liefert Liste [(zone_number_in_filename, filepath), ...] sortiert nach zone_number_in_filename.
    Erwartet Dateien wie: <formatted_id>_Storey_1_Zone_2.mo
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


def apply_record_overrides_regex(record_file_path: str, overrides: Dict[str, Any]) -> None:
    """
    Ersetzt Parameterzuweisungen im Record:
      key = <value>
    (Komma/Format bleibt stabil, wir ersetzen nur den Wert-Teil)
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
    # lights immer false
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
    Reihenfolge pro Zone:
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
    Patcht alle Zone_*.mo Records.
    Mapping: zi=0..n-1 entspricht SORTIERTER Reihenfolge der Zone_* Dateien.
    (Damit ist "erste Zone = TH" erfüllt.)
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
# Setpoints (multizone)
# ───────────────────────────────────────────────────────────────
def write_setpoints_multizone(
    time_table_T_set: str,
    time_table_T_set_cool: str,
    full_year_hours: int,
    n_zones: int,
    rng: np.random.Generator,
    enable_cooling: bool = True,
):
    """
    n_zones == 1:
      - kein TH: Heizen zufällig {18..22}, Kühlen {23..26} oder aus

    n_zones >= 2:
      - Zone 0 = TH: Heizen fix 8°C, Kühlen aus
      - Zonen 1..: Heizen zufällig {18..22}, Kühlen {23..26} oder aus
    """
    if n_zones <= 0:
        raise ValueError("n_zones muss >= 1 sein")

    heat_choices_C = np.array([18, 19, 20, 21, 22], dtype=int)
    cool_choices_C = np.array([23, 24, 25, 26], dtype=int)

    if n_zones == 1:
        heat_vals = np.array([273.15 + float(rng.choice(heat_choices_C))], dtype=float)
        if not enable_cooling:
            cool_vals = np.array([273.15 + 60.0], dtype=float)
        else:
            cool_vals = np.array([273.15 + float(rng.choice(cool_choices_C))], dtype=float)
    else:
        heat_vals = np.empty(n_zones, dtype=float)
        cool_vals = np.empty(n_zones, dtype=float)

        # TH
        heat_vals[0] = 273.15 + 8.0
        cool_vals[0] = 273.15 + 60.0  # nie kühlen

        # Wohnzonen
        drawn_heat_C = rng.choice(heat_choices_C, size=n_zones - 1, replace=True)
        heat_vals[1:] = 273.15 + drawn_heat_C.astype(float)

        if not enable_cooling:
            cool_vals[1:] = 273.15 + 60.0
        else:
            drawn_cool_C = rng.choice(cool_choices_C, size=n_zones - 1, replace=True)
            cool_vals[1:] = 273.15 + drawn_cool_C.astype(float)

    def _write(path, table_name, vals):
        with open(path, "w", encoding="utf-8") as f:
            f.write(f"#1\ndouble {table_name}({full_year_hours}, {1+n_zones})\n")
            for h in range(full_year_hours):
                sec = h * 3600
                row = [f"{sec}"] + [f"{vals[z]:.2f}" for z in range(n_zones)]
                f.write("\t".join(row) + "\n")

    _write(time_table_T_set, "Tset", heat_vals)
    _write(time_table_T_set_cool, "Tset", cool_vals)


# ───────────────────────────────────────────────────────────────
# Internal gains tables
# ───────────────────────────────────────────────────────────────
def write_internal_gains_single_table(
    combi_time_table_path: str,
    full_year_hours: int,
    occ_abs_year: np.ndarray,
    machines_year: np.ndarray,
    lights_year: np.ndarray,
    occ_rel_year: np.ndarray,
    gains_scale: float,
):
    occ_abs_year  = np.asarray(occ_abs_year, dtype=float).flatten()
    machines_year = np.asarray(machines_year, dtype=float).flatten() * float(gains_scale)
    lights_year   = np.asarray(lights_year, dtype=float).flatten() * float(gains_scale)
    occ_rel_year  = np.asarray(occ_rel_year, dtype=float).flatten()

    with open(combi_time_table_path, "w", encoding="utf-8") as f:
        f.write(f"#1\ndouble Internals({full_year_hours}, 5)\n")
        for h in range(full_year_hours):
            sec = h * 3600
            f.write(
                f"{sec}\t{occ_abs_year[h]:.2f}\t{machines_year[h]:.2f}\t{lights_year[h]:.2f}\t{occ_rel_year[h]:.2f}\n"
            )


def write_internal_gains_multizone_table(
    combi_time_table_path: str,
    full_year_hours: int,
    occ_abs_building: np.ndarray,     # building total
    machines_building: np.ndarray,    # building total
    lights_building: np.ndarray,      # building total
    occ_rel_building: np.ndarray,     # 0..1
    n_zones: int,
    zone_weights: List[float],
    gains_scale: float,
    # TH factors (wie von dir gefordert)
    th_people_factor: float = 0.1,
    th_lights_factor: float = 0.1,
    th_occ_rel_factor: float = 0.1,   # oder 0.0
    th_machines_factor: float = 0.0,
):
    """
    Semantik:
      n_zones == 1:
        - keine TH-Aufteilung: alles building total in Zone 0

      n_zones == 2:
        - Zone 0 = TH:
            people  = th_people_factor * occ_abs_building
            lights  = th_lights_factor * lights_building
            machines= th_machines_factor * machines_building (i.d.R. 0.0)
            occ_rel = th_occ_rel_factor * occ_rel_building
        - Zone 1 = Wohnungen:
            people  = (1-th_people_factor) * occ_abs_building
            lights  = (1-th_lights_factor) * lights_building
            machines= machines_building
            occ_rel = occ_rel_building

      n_zones > 2:
        - Zone 0 = TH wie oben
        - Zonen 1..n-1 = Wohnzonen:
            people  = (1-th_people_factor) * occ_abs_building  verteilt nach zone_weights (nur Wohnzonen)
            lights  = (1-th_lights_factor) * lights_building    verteilt nach zone_weights (nur Wohnzonen)
            machines= machines_building                         verteilt nach zone_weights (nur Wohnzonen)
            occ_rel = occ_rel_building                          (nicht gewichten)
    """
    if n_zones <= 0:
        raise ValueError("n_zones muss >= 1 sein")

    occ_abs_building  = np.asarray(occ_abs_building, dtype=float).flatten()
    machines_building = np.asarray(machines_building, dtype=float).flatten() * float(gains_scale)
    lights_building   = np.asarray(lights_building, dtype=float).flatten() * float(gains_scale)
    occ_rel_building  = np.asarray(occ_rel_building, dtype=float).flatten()

    # sicherstellen 8760
    def _fix_len(x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float).flatten()
        if x.size >= full_year_hours:
            return x[:full_year_hours]
        rep = int(np.ceil(full_year_hours / max(x.size, 1)))
        return np.tile(x, rep)[:full_year_hours]

    occ_abs_building  = _fix_len(occ_abs_building)
    machines_building = _fix_len(machines_building)
    lights_building   = _fix_len(lights_building)
    occ_rel_building  = _fix_len(occ_rel_building)

    if n_zones == 1:
        # single-zone: alles in Zone 0
        n_cols = 1 + 4 * 1
        with open(combi_time_table_path, "w", encoding="utf-8") as f:
            f.write(f"#1\ndouble Internals({full_year_hours}, {n_cols})\n")
            for h in range(full_year_hours):
                sec = h * 3600
                f.write(
                    f"{sec}\t{occ_abs_building[h]:.2f}\t{machines_building[h]:.2f}\t{lights_building[h]:.2f}\t{occ_rel_building[h]:.2f}\n"
                )
        return

    # --- Gewichte für Wohnzonen (1..n-1) ---
    n_res = n_zones - 1
    if n_res == 1:
        w_res = np.array([1.0], dtype=float)
    else:
        w = np.array(zone_weights, dtype=float) if zone_weights is not None else np.ones(n_zones, dtype=float)
        if w.size != n_zones:
            w = np.ones(n_zones, dtype=float)
        w_res = w[1:].copy()
        s = float(np.sum(w_res))
        if not np.isfinite(s) or s <= 0:
            w_res[:] = 1.0 / n_res
        else:
            w_res /= s

    # --- Split TH / Wohnungen ---
    th_p = float(th_people_factor)
    th_l = float(th_lights_factor)
    th_m = float(th_machines_factor)
    th_r = float(th_occ_rel_factor)

    th_p = float(np.clip(th_p, 0.0, 1.0))
    th_l = float(np.clip(th_l, 0.0, 1.0))
    # machines factor kann 0..1, aber i.d.R. 0.0
    th_m = float(np.clip(th_m, 0.0, 1.0))
    th_r = float(np.clip(th_r, 0.0, 1.0))

    # --- Tabellenarrays ---
    occ_abs_z  = np.zeros((full_year_hours, n_zones), dtype=float)
    machines_z = np.zeros((full_year_hours, n_zones), dtype=float)
    lights_z   = np.zeros((full_year_hours, n_zones), dtype=float)
    occ_rel_z  = np.zeros((full_year_hours, n_zones), dtype=float)

    # Zone 0 (TH): zeitvariable Reihen (Faktor * building)
    occ_abs_z[:, 0]  = th_p * occ_abs_building
    lights_z[:, 0]   = th_l * lights_building
    machines_z[:, 0] = th_m * machines_building
    occ_rel_z[:, 0]  = th_r * occ_rel_building

    # Wohnungen: "Rest" für people/lights, machines komplett in Wohnungen
    occ_abs_res  = (1.0 - th_p) * occ_abs_building
    lights_res   = (1.0 - th_l) * lights_building
    machines_res = machines_building

    if n_zones == 2:
        occ_abs_z[:, 1]  = occ_abs_res
        lights_z[:, 1]   = lights_res
        machines_z[:, 1] = machines_res
        occ_rel_z[:, 1]  = occ_rel_building
    else:
        for zi in range(1, n_zones):
            wi = float(w_res[zi - 1])
            occ_abs_z[:, zi]  = occ_abs_res * wi
            lights_z[:, zi]   = lights_res * wi
            machines_z[:, zi] = machines_res * wi
            occ_rel_z[:, zi]  = occ_rel_building

    # --- Schreiben ---
    n_cols = 1 + 4 * n_zones
    with open(combi_time_table_path, "w", encoding="utf-8") as f:
        f.write(f"#1\ndouble Internals({full_year_hours}, {n_cols})\n")
        for h in range(full_year_hours):
            sec = h * 3600
            row = [f"{sec}"]
            for z in range(n_zones):
                row += [
                    f"{occ_abs_z[h, z]:.2f}",
                    f"{machines_z[h, z]:.2f}",
                    f"{lights_z[h, z]:.2f}",
                    f"{occ_rel_z[h, z]:.2f}",
                ]
            f.write("\t".join(row) + "\n")


def write_internal_gains_multizone_table_from_zone_series(
    combi_time_table_path: str,
    full_year_hours: int,
    people_z: np.ndarray,     # shape (n_zones, 8760)
    machines_z: np.ndarray,   # shape (n_zones, 8760)
    lights_z: np.ndarray,     # shape (n_zones, 8760)
    occ_rel_z: np.ndarray,    # shape (n_zones, 8760)
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
    lpg_results_path: str,  # bleibt im Interface, wird aber nicht mehr genutzt
    sa_params: SAParams,
    internal_gains_mode: str = "multizone_table",  # "single_table" oder "multizone_table"
):
    json_id = to_dashed_id(building_id)
    formatted_id = json_id.replace("-", "")

    if not building_model_exists(formatted_id, sim_models_dir):
        raise FileNotFoundError(f"Kein Sim-Modell für {formatted_id} in {sim_models_dir}")

    # getrennte RNGs (Setpoints separat)
    rng_sp = np.random.default_rng(int(sa_params.rng_seed) + 10_000_003)

    out_dir = pathlib.Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    run_id = f"{int(time.time()*1000)}_{os.getpid()}_{uuid.uuid4().hex[:6]}"
    work_dir = out_dir / "work" / run_id
    work_dir.mkdir(parents=True, exist_ok=True)

    # Weather vorbereiten (Side-effect: Referenzen updaten)
    parse_weather_and_update_reference(
        mos_file_path=mos_file_path,
        sim_models_dir=sim_models_dir,
        formatted_id=formatted_id,
    )

    # Pfade für Input-Tabellen
    combi_time_table_path = os.path.join(sim_models_dir, formatted_id, f"InternalGains_{formatted_id}.txt")
    time_table_T_set      = os.path.join(sim_models_dir, formatted_id, f"TsetHeat_{formatted_id}.txt")
    time_table_T_set_cool = os.path.join(sim_models_dir, formatted_id, f"TsetCool_{formatted_id}.txt")

    full_year_hours = 8760

    # n_zones robust bestimmen über Zone_*.mo records
    n_zones = _infer_n_zones_from_any_record(sim_models_dir, formatted_id)

    # zone_weights (für >2 Wohnzonen)
    zone_weights = _normalize_weights(max(1, n_zones), sa_params.zone_weights)

    # ────────────────────────────────────────────────────────────
    # LPG: Gebäudeprofile aus Template-Pool bauen
    # ────────────────────────────────────────────────────────────
    if sa_params.lpg_cfg is None:
        raise ValueError("sa_params.lpg_cfg ist None – LPGSelectionConfig muss gesetzt sein.")

    apt = build_lpg_apartments_year(seed=int(sa_params.rng_seed), cfg=sa_params.lpg_cfg)

    occ_ap = apt["occupancy_abs"]  # (A, 8760)
    mach_ap = apt["machines"]  # (A, 8760)
    occ_rel_ap = apt["occupancy_rel"]  # (A, 8760)

    A = int(occ_ap.shape[0])

    lighting_profile = np.asarray(
        np.tile([0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0], 365),
        dtype=float
    )[:full_year_hours]

    # TH “vordefinierte Zeitreihe”
    th_people = np.full(full_year_hours, 0.1)  # oder konstant 0.1 überall, je nachdem was du meinst
    th_lights = 0.1 * lighting_profile
    th_machines = np.zeros(full_year_hours)
    th_occ_rel = 0.1 * np.clip(np.sum(occ_rel_ap, axis=0) / max(A, 1), 0, 1)  # oder 0.0

    if n_zones == 1:
        # V1: alles aggregiert in Zone 0
        people_z = np.sum(occ_ap, axis=0, keepdims=True)
        machines_z = np.sum(mach_ap, axis=0, keepdims=True) * sa_params.gains_scale
        lights_z = lighting_profile[None, :]  # unverändert
        # occ_rel: aggregiert z.B. über total persons (oder Mittel)
        occ_rel_z = np.clip(np.mean(occ_rel_ap, axis=0, keepdims=True), 0, 1)

    elif n_zones == 2:
        # V2: Zone0=TH, Zone1=alle Wohnungen aggregiert
        people_z = np.vstack([
            th_people[None, :],
            np.sum(occ_ap, axis=0, keepdims=True) * 0.9
        ])
        machines_z = np.vstack([
            th_machines[None, :],
            np.sum(mach_ap, axis=0, keepdims=True) * sa_params.gains_scale
        ])
        lights_z = np.vstack([
            th_lights[None, :],
            lighting_profile[None, :] * 0.9
        ])
        occ_rel_z = np.vstack([
            th_occ_rel[None, :],
            np.clip(np.mean(occ_rel_ap, axis=0, keepdims=True), 0, 1)
        ])

    else:
        # V3+: Zone0=TH, Zonen1.. = “direkt aus LPG”
        n_res = n_zones - 1

        # Falls A != n_res: wir mappen “best effort”
        # - wenn A >= n_res: n_res Apartments ziehen
        # - wenn A < n_res: Apartments zyklisch wiederholen
        idx = np.arange(n_res) % max(A, 1)

        people_res = occ_ap[idx, :]  # (n_res, 8760)
        machines_res = mach_ap[idx, :] * sa_params.gains_scale
        occ_rel_res = occ_rel_ap[idx, :]

        # lights: pro Wohnzone der gleiche lighting_profile (oder später zonal)
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

    # 1) Setpoints schreiben
    write_setpoints_multizone(
        time_table_T_set=time_table_T_set,
        time_table_T_set_cool=time_table_T_set_cool,
        full_year_hours=full_year_hours,
        n_zones=n_zones,
        rng=rng_sp,
        enable_cooling=bool(sa_params.enable_cooling),
    )

    # 3) Record overrides je Zone anwenden
    patched = apply_zone_record_overrides(sim_models_dir, formatted_id, sa_params)

    # 4) Dymola Run
    teaser_mo = os.path.join(sim_models_dir, "package.mo")
    model_name = f"{sim_model_pkg_name}.{formatted_id}.{formatted_id}"

    dym_api = DymolaAPI(
        working_directory=str(work_dir),
        model_name=model_name,
        n_cpu=1,
        packages=[aixlib_mo, teaser_mo],
        show_window=False,
        equidistant_output=False
    )
    dym_api.set_sim_setup({
        "start_time": int(start_sim),
        "stop_time": int(end_sim),
        "output_interval": int(OUTPUT_INTERVAL_SEC)
    })

    result_path = dym_api.simulate(return_option="savepath")
    tsd = TimeSeriesData(result_path)

    # 5) Minimal-Outputs
    heat_W = tsd["multizone.PHeater[1]"].values.flatten()
    cool_W = tsd["multizone.PCooler[1]"].values.flatten()

    heat_kWh = float(np.trapz(heat_W, dx=OUTPUT_INTERVAL_SEC) / 3.6e6)
    cool_kWh = float(np.trapz(cool_W, dx=OUTPUT_INTERVAL_SEC) / 3.6e6)
    peak_heat_kW = float(np.max(heat_W) / 1000.0)
    peak_cool_kW = float(np.max(cool_W) / 1000.0)

    out = {
        "building_id": str(json_id),
        "formatted_id": str(formatted_id),
        "n_zones": int(n_zones),
        "patched_zone_records": int(patched),
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
            "gains_scale": sa_params.gains_scale,
            "rng_seed": sa_params.rng_seed,
            "enable_cooling": bool(sa_params.enable_cooling),
            "th_people_factor": sa_params.th_people_factor,
            "th_lights_factor": sa_params.th_lights_factor,
            "th_occ_rel_factor": sa_params.th_occ_rel_factor,
            "th_machines_factor": sa_params.th_machines_factor,
            "record_overrides_global": sa_params.record_overrides_global,
            "record_overrides_by_zone": sa_params.record_overrides_by_zone,
        },
        "mos_file": mos_file_path,
        "start_sim": int(start_sim),
        "end_sim": int(end_sim),
    }

    with open(out_dir / "overall.json", "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    dym_api.close()
    dym_api = None
    shutil.rmtree(work_dir, ignore_errors=True)

    return out


# ───────────────────────────────────────────────────────────────
# Batch runner
# ───────────────────────────────────────────────────────────────
def _worker(task):
    return simulate_one(**task)

def run_many(tasks: List[Dict[str, Any]], n_proc: int = 1):
    if n_proc == 1:
        return [_worker(t) for t in tasks]
    ctx = multiprocessing.get_context("spawn")
    with ctx.Pool(processes=n_proc, maxtasksperchild=1) as pool:
        return pool.map(_worker, tasks)
