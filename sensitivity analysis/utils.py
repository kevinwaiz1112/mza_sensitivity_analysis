import glob, os, re
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import pathlib
import json

# Pool root
LPG_POOL_ROOT = r"C:\03_Repos\SimData\pylpg\HH_AllTemplates_5x\HH_AllTemplates_5x"

TP_CODES = ["TP_BER21", "TP_BER23", "TP_BER25", "TP_DEL25", "TP_FR"]

# Labels in euren LPG-Dateinamen (Beispiele)
LABEL_ELEC = "Electricity"
LABEL_WW   = "Warm Water"      # ggf. heißt es "Hot Water" in manchen Sets
LABEL_HW   = "Hot Water"
LABEL_CW   = "Cold Water"
LABEL_IG   = "Inner Device Heat Gains"
LABEL_BAL_OUTSIDE = "BodilyActivityLevel.Outside"
HH_FIXED = "HH1"


# --- Hilfsfunktion: stündliche SumProfiles_3600s.*.csv lesen ---
def _read_sumprofiles_csv(csv_path: str) -> np.ndarray:
    """
    Erwartet eine Datei wie: SumProfiles_3600s.Electricity.csv
    mit ';' getrennt und einer Spalte 'Sum [kWh]' oder ähnlich.
    Gibt Werte als numpy array zurück (Länge 8760 typischerweise).
    """
    df = pd.read_csv(csv_path, sep=";", engine="python")
    # passende Sum-Spalte suchen
    sum_cols = [c for c in df.columns if c.startswith("Sum [")]
    if not sum_cols:
        raise ValueError(f"Keine 'Sum [...]' Spalte in {csv_path}")
    col = sum_cols[0]
    vals = df[col].astype(float).to_numpy()
    return vals


def _results_dir_for(tp: str, template: str, r: int) -> str:
    res_dir = os.path.join(
        LPG_POOL_ROOT, tp, template, f"r{r}",
        f"Results_{template}_{tp.replace('_','')}_r{r}",
        "Results"
    )
    if os.path.isdir(res_dir):
        return res_dir

    patt = os.path.join(LPG_POOL_ROOT, tp, template, f"r{r}", "**", "Results")
    hits = [h for h in glob.glob(patt, recursive=True) if os.path.basename(h).lower() == "results"]
    if not hits:
        raise FileNotFoundError(f"Kein Results-Ordner gefunden: tp={tp}, template={template}, r={r}")
    return hits[0]

def _list_hh_keys(tp: str, template: str, r: int) -> list[str]:
    return [HH_FIXED]

def _find_outside_json(tp: str, template: str, r: int, hh_key: str = HH_FIXED) -> str | None:
    res_dir = _results_dir_for(tp, template, r)
    p = os.path.join(res_dir, f"{LABEL_BAL_OUTSIDE}.{hh_key}.json")
    return p if os.path.isfile(p) else None


def _outside_json_to_inside_hourly_8760(json_path: str, persons_fallback: int) -> tuple[np.ndarray, int]:
    with open(json_path, "r", encoding="utf-8") as f:
        d = json.load(f)

    outside_min = np.asarray(d.get("Values", []), dtype=float)
    if outside_min.size == 0:
        return np.zeros(8760, dtype=float), max(1, int(persons_fallback))

    # persons: wenn Outside in [0..persons] ist, max(outside) ist brauchbarer Proxy
    mx = float(np.nanmax(outside_min))
    persons = max(1, int(np.ceil(mx))) if np.isfinite(mx) and mx > 0 else max(1, int(persons_fallback))

    inside_min = np.clip(float(persons) - outside_min, 0.0, float(persons))

    n_hours_avail = int(inside_min.size // 60)
    inside_h = inside_min[:n_hours_avail * 60].reshape(n_hours_avail, 60).mean(axis=1) if n_hours_avail > 0 else np.zeros(1)

    # auf 8760 trim/tile
    if inside_h.size >= 8760:
        inside_8760 = inside_h[:8760].astype(float)
    else:
        rep = int(np.ceil(8760 / max(inside_h.size, 1)))
        inside_8760 = np.tile(inside_h, rep)[:8760].astype(float)

    return inside_8760, persons


def _find_profile_csv(tp: str, template: str, r: int, hh_key: str, label: str) -> str | None:
    """
    Erwartet:
    ...\Results\SumProfiles_3600s.HH1.Electricity.csv
    """
    res_dir = _results_dir_for(tp, template, r)
    p = os.path.join(res_dir, f"SumProfiles_3600s.{hh_key}.{label}.csv")
    if os.path.isfile(p):
        return p

    # Fallback: glob (robust gegen label-Varianten)
    patt = os.path.join(res_dir, f"SumProfiles_3600s.{hh_key}.*{label}*.csv")
    hits = glob.glob(patt)
    return hits[0] if hits else None


@dataclass
class LPGSelectionConfig:
    n_apartments: int
    # Häufigkeiten nach Personenanzahl, z.B. mehr 1-2P
    size_probs: Dict[int, float]  # {1:0.4,2:0.35,3:0.15,4:0.07,5:0.03,6:0.0}
    # Mapping Template->persons (aus household_sizes ableitbar)
    template_to_persons: Dict[str, int]
    # Wetter/TP: entweder fix oder zufällig
    tp_mode: str = "fixed"            # "fixed" oder "random"
    tp_fixed: str = "TP_BER21"
    # LPG seed handling
    lpg_seed_mode: str = "random_r"   # "random_r" oder "mean_r"
    r_values: Tuple[int, ...] = (1,2,3,4,5)

def _draw_template(rng: np.random.Generator, cfg: LPGSelectionConfig) -> str:
    # 1) Haushaltsgröße ziehen
    sizes = np.array(list(cfg.size_probs.keys()), dtype=int)
    probs = np.array([cfg.size_probs[s] for s in sizes], dtype=float)
    probs = probs / probs.sum()

    size = int(rng.choice(sizes, p=probs))

    # 2) Template aus dieser Größe ziehen
    candidates = [t for t, p in cfg.template_to_persons.items() if int(p) == size]
    if not candidates:
        # Fallback: wenn keine Kandidaten existieren, aus allen ziehen
        candidates = list(cfg.template_to_persons.keys())
    return str(rng.choice(candidates))

def _choose_tp(rng: np.random.Generator, cfg: LPGSelectionConfig) -> str:
    if cfg.tp_mode == "fixed":
        return cfg.tp_fixed
    return str(rng.choice(TP_CODES))

def _to8760(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float).flatten()
    if x.size >= 8760:
        return x[:8760]
    rep = int(np.ceil(8760 / max(x.size, 1)))
    return np.tile(x, rep)[:8760]

def _load_one_hh_profile(tp: str, template: str, r: int, hh_key: str, persons_hint: int) -> dict[str, np.ndarray]:
    elec_p = _find_profile_csv(tp, template, r, hh_key, LABEL_ELEC)
    ww_p   = _find_profile_csv(tp, template, r, hh_key, LABEL_WW) or _find_profile_csv(tp, template, r, hh_key, LABEL_HW)
    cw_p   = _find_profile_csv(tp, template, r, hh_key, LABEL_CW)
    ig_p   = _find_profile_csv(tp, template, r, hh_key, LABEL_IG)

    if elec_p is None or ww_p is None or ig_p is None:
        raise FileNotFoundError(f"Profil fehlt: tp={tp}, template={template}, r={r}, hh={hh_key}")

    elec = _to8760(_read_sumprofiles_csv(elec_p))  # kWh/h
    ww   = _to8760(_read_sumprofiles_csv(ww_p))    # L/h
    ig   = _to8760(_read_sumprofiles_csv(ig_p))    # kWh/h
    cw   = np.zeros(8760) if cw_p is None else _to8760(_read_sumprofiles_csv(cw_p))

    outside_json = _find_outside_json(tp, template, r, hh_key)
    if outside_json is None:
        occ_inside = np.zeros(8760)
        persons = max(1, int(persons_hint))
    else:
        occ_inside, persons = _outside_json_to_inside_hourly_8760(outside_json, persons_fallback=persons_hint)

    return {
        "electricity_demand": elec,
        "warm_water_demand":  ww,
        "cold_water_demand":  cw,
        "internal_gains":     ig,
        "occupancy_inside":   occ_inside,                 # persons/h
        "persons":            np.array([persons], float),
    }

def build_lpg_apartments_year(seed: int, cfg: LPGSelectionConfig) -> dict:
    """
    Liefert pro Apartment (cfg.n_apartments) ein eigenes Profil:
      - occupancy_abs: (n_apartments, 8760)
      - machines:      (n_apartments, 8760)
      - persons:       (n_apartments,)
      - occupancy_rel: (n_apartments, 8760)
    """
    rng = np.random.default_rng(seed)

    occ_list = []
    mach_list = []
    persons_list = []

    for _ in range(cfg.n_apartments):
        tp = _choose_tp(rng, cfg)
        template = _draw_template(rng, cfg)
        persons_hint = int(cfg.template_to_persons.get(template, 1))
        hh_key = HH_FIXED  # HH1 fix

        if cfg.lpg_seed_mode == "mean_r":
            acc_occ = np.zeros(8760, dtype=float)
            acc_mach = np.zeros(8760, dtype=float)
            persons_ap = None
            count = 0

            for r in cfg.r_values:
                p = _load_one_hh_profile(tp, template, int(r), hh_key, persons_hint)
                acc_occ += np.asarray(p["occupancy_inside"], float)
                acc_mach += np.asarray(p["internal_gains"], float)
                if persons_ap is None:
                    persons_ap = int(p["persons"][0])
                count += 1

            if count == 0:
                raise FileNotFoundError(f"Keine Seeds geladen (mean_r): tp={tp}, template={template}")

            acc_occ /= float(count)
            acc_mach /= float(count)
            if persons_ap is None:
                persons_ap = persons_hint

            occ_list.append(acc_occ)
            mach_list.append(acc_mach)
            persons_list.append(persons_ap)

        else:
            r = int(rng.choice(cfg.r_values))
            p = _load_one_hh_profile(tp, template, r, hh_key, persons_hint)
            occ_list.append(np.asarray(p["occupancy_inside"], float))
            mach_list.append(np.asarray(p["internal_gains"], float))
            persons_list.append(int(p["persons"][0]))

    occ = np.stack(occ_list, axis=0)              # (A, 8760)
    mach = np.stack(mach_list, axis=0)            # (A, 8760)
    persons = np.asarray(persons_list, float)     # (A,)

    persons_safe = np.where(persons > 0, persons, 1.0)
    occ_rel = np.clip(occ / persons_safe[:, None], 0.0, 1.0)

    return {
        "occupancy_abs": occ,
        "machines": mach,
        "persons": persons,
        "occupancy_rel": occ_rel,
    }



def build_lpg_building_year(seed: int, cfg: LPGSelectionConfig) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)

    elec_sum = np.zeros(8760)
    ww_sum   = np.zeros(8760)
    cw_sum   = np.zeros(8760)
    ig_sum   = np.zeros(8760)
    occ_abs  = np.zeros(8760)

    persons_total = 0

    for _ in range(cfg.n_apartments):
        tp = _choose_tp(rng, cfg)
        template = _draw_template(rng, cfg)
        persons_hint = int(cfg.template_to_persons.get(template, 1))

        if cfg.lpg_seed_mode == "mean_r":
            # hh_key einmal festlegen (aus erster r)
            r0 = int(cfg.r_values[0])
            hh_keys0 = _list_hh_keys(tp, template, r0)
            if not hh_keys0:
                raise FileNotFoundError(f"Keine HH-Profile gefunden: tp={tp}, template={template}, r={r0}")
            hh_key = str(rng.choice(hh_keys0))

            acc = None
            persons_ap = None
            count = 0
            for r in cfg.r_values:
                hh_keys_r = _list_hh_keys(tp, template, int(r))
                if not hh_keys_r:
                    continue
                hh_use = hh_key if hh_key in hh_keys_r else hh_keys_r[0]

                p = _load_one_hh_profile(tp, template, int(r), hh_use, persons_hint)
                count += 1
                if persons_ap is None:
                    persons_ap = int(p["persons"][0])

                if acc is None:
                    acc = {k: p[k].copy() for k in ["electricity_demand","warm_water_demand","cold_water_demand","internal_gains","occupancy_inside"]}
                else:
                    for k in acc:
                        acc[k] += p[k]

            if count == 0:
                raise FileNotFoundError(
                    f"Keine Seeds geladen (mean_r): tp={tp}, template={template}, r_values={cfg.r_values}")

            if acc is None:
                raise FileNotFoundError(f"Keine Daten beim mean_r: tp={tp}, template={template}")

            for k in acc:
                acc[k] /= float(count)

            elec_sum += acc["electricity_demand"]
            ww_sum   += acc["warm_water_demand"]
            cw_sum   += acc["cold_water_demand"]
            ig_sum   += acc["internal_gains"]
            occ_abs  += acc["occupancy_inside"]
            persons_total += persons_ap

        else:
            # random_r
            r = int(rng.choice(cfg.r_values))
            hh_keys = _list_hh_keys(tp, template, r)
            if not hh_keys:
                raise FileNotFoundError(f"Keine HH-Profile gefunden: tp={tp}, template={template}, r={r}")
            hh_key = str(rng.choice(hh_keys))

            p = _load_one_hh_profile(tp, template, r, hh_key, persons_hint)

            elec_sum += p["electricity_demand"]
            ww_sum   += p["warm_water_demand"]
            cw_sum   += p["cold_water_demand"]
            ig_sum   += p["internal_gains"]
            occ_abs  += p["occupancy_inside"]
            persons_total += int(p["persons"][0])

    occ_rel = np.zeros(8760)
    if persons_total > 0:
        occ_rel = np.clip(occ_abs / float(persons_total), 0.0, 1.0)

    return {
        "electricity_demand": elec_sum,
        "warm_water_demand":  ww_sum,
        "cold_water_demand":  cw_sum,
        "internal_gains":     ig_sum,
        "occupancy_abs":      occ_abs,                         # persons/h
        "occupancy_rel":      occ_rel,                         # 0..1
        "persons_total":      np.array([persons_total], float),
    }



def building_model_exists(building_id: str, sim_models_dir: str) -> bool:
    """
    Prüft, ob für ‹building_id› ein .mo-Modell im TEASER-Export liegt.

    Gesucht wird nach  …/<sim_models_dir>/<building_id>.mo
                    oder …/<sim_models_dir>/<building_id>/package.mo
    """
    root = pathlib.Path(sim_models_dir)
    return (
        (root / f"{building_id}.mo").is_file()
        or (root / building_id / "package.mo").is_file()
    )

def prepare_t_soil_inputs(
    use_variable_t_soil: bool,
    mos_file_path: str,
    outdoor_temperature_data: list[tuple[float, float]],
    T_soil_constant: float,
):
    if use_variable_t_soil == True:
        import os
        input_directory, input_filename = os.path.split(mos_file_path)

        base_dir = os.path.dirname(os.path.abspath(__file__))
        soil_mos_file_path = os.path.join(base_dir, "weather", "mos_data", "Variable_T_soil_temperature.mos")
        soil_mos_file_path = soil_mos_file_path.replace('\\', '\\\\')

        with open(soil_mos_file_path, 'w') as output_file:
            output_file.write('#1\n')
            output_file.write('double TGround({},{})\n'.format(len(outdoor_temperature_data), 2))
            for time, temperature in outdoor_temperature_data:
                output_file.write('{}\t{:.2f}\n'.format(time, temperature + 273.15))

        T_Soil = T_soil_constant
        T_SoilFile = soil_mos_file_path
        T_SoilDataSource = "AixLib.BoundaryConditions.GroundTemperature.GroundTemperatureDataSource.File"
    else:
        T_Soil = T_soil_constant
        T_SoilDataSource = "AixLib.BoundaryConditions.GroundTemperature.GroundTemperatureDataSource.Constant"
        T_SoilFile = "NoName"

    return T_Soil, T_SoilFile, T_SoilDataSource


def parse_weather_and_update_reference(
    mos_file_path: str,
    sim_models_dir: str,
    formatted_id: str,
    start_date_str: str = "2021-01-01 00:00:00",
):
    # --- Weather Data (aus .mos lesen) ---
    with open(mos_file_path, 'r') as input_file:
        lines = input_file.readlines()

    outdoor_temperature_data = []
    for line in lines:
        if line.strip() and not line.startswith('#') and not line.startswith('double'):
            parts = line.split()
            if len(parts) >= 2:
                try:
                    times = int(parts[0])          # Zeit in Sekunden
                    temperature = float(parts[1])  # Temperatur in °C
                    outdoor_temperature_data.append((times, temperature))
                except ValueError:
                    continue

    df_temperature = pd.DataFrame(outdoor_temperature_data, columns=['time_in_seconds', 'temp_celsius'])
    start_date = pd.Timestamp(start_date_str)
    df_temperature['datetime'] = pd.to_datetime(start_date) + pd.to_timedelta(df_temperature['time_in_seconds'], unit='s')
    df_temperature['datetime'] = df_temperature['datetime'].dt.round('H')
    df_temperature = df_temperature[['datetime', 'temp_celsius']].rename(columns={'temp_celsius': 'temp'})

    # --- Modelica .mo: filNam auf neue MOS-Datei setzen (update_modelica_mos_reference) ---
    base_modelica_file_path = os.path.join(sim_models_dir, str(formatted_id), f"{formatted_id}.mo")
    new_filnam = base_modelica_file_path.replace('\\', '\\\\')
    new_mos_file_path = mos_file_path.replace('\\', '\\\\')

    pattern = re.compile(r'filNam=(?:Modelica\.Utilities\.Files\.loadResource\([^)]+\)|"[^"]+")')
    with open(new_filnam, 'r', encoding='utf-8') as file:
        content = file.read()
    content = pattern.sub('filNam="' + new_mos_file_path + '"', content)
    with open(new_filnam, 'w', encoding='utf-8') as file:
        file.write(content)

    return df_temperature, outdoor_temperature_data


def to_dashed_id(bid: str) -> str:
    """
    Wandelt 'ID_<32hex>' in 'ID_xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx' um.
    Gibt bid unverändert zurück, wenn kein Match.
    """
    m = re.match(r'^(ID_)([A-Fa-f0-9]{32})$', str(bid))
    if not m:
        return str(bid)
    h = m.group(2).upper()
    return f"{m.group(1)}{h[:8]}-{h[8:12]}-{h[12:16]}-{h[16:20]}-{h[20:]}"

def tsd_scalar(tsd, key, caster=float, default=None):
    try:
        obj = tsd[key]  # kann Series/DF/numpy sein
        # DataFrame → ersten numerischen Wert
        if isinstance(obj, pd.DataFrame):
            vals = pd.to_numeric(obj.stack(), errors="coerce").dropna()
            if not vals.empty:
                val = vals.iloc[0]
                return caster(val.item() if hasattr(val, "item") else val)
        # Series → ersten numerischen Wert
        if isinstance(obj, pd.Series):
            vals = pd.to_numeric(obj, errors="coerce").dropna()
            if not vals.empty:
                val = vals.iloc[0]
                return caster(val.item() if hasattr(val, "item") else val)
        # numpy/Skalar
        if hasattr(obj, "item"):
            return caster(obj.item())
        return caster(obj)
    except Exception:
        return default