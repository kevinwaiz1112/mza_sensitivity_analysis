from teaser.project import Project
from teaser.logic.buildingobjects.building import Building
from teaser.logic.buildingobjects.thermalzone import ThermalZone
from teaser.logic.buildingobjects.useconditions import UseConditions
from teaser.logic.buildingobjects.buildingphysics.rooftop import Rooftop
from teaser.logic.buildingobjects.buildingphysics.outerwall import OuterWall
from teaser.logic.buildingobjects.buildingphysics.innerwall import InnerWall
from teaser.logic.buildingobjects.buildingphysics.ceiling import Ceiling
from teaser.logic.buildingobjects.buildingphysics.floor import Floor
from teaser.logic.buildingobjects.buildingphysics.groundfloor import GroundFloor
from teaser.logic.buildingobjects.buildingphysics.window import Window
from teaser.logic.buildingobjects.buildingphysics.interzonalceiling import InterzonalCeiling
from teaser.logic.buildingobjects.buildingphysics.interzonalfloor import InterzonalFloor
from teaser.logic.buildingobjects.buildingphysics.interzonalwall import InterzonalWall
from teaser.data.dataclass import DataClass
import teaser.logic.utilities as utilities
import numpy as np
import uuid
import json
import re
import math
import warnings
import hashlib
from pathlib import Path
import os
import pandas as pd

"""
The following code is adapted from the heatbrain repository and expanded for the use of interzonal elements.
This script is responsible for generating a TEASER project for a building based on its input data.
It processes walls, floors, ceilings, and thermal zones to calculate building elements.
"""

def _tile_8760(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    if x.size == 0:
        return np.zeros(8760, dtype=float)
    if x.size < 8760:
        reps = int(np.ceil(8760 / x.size))
        x = np.tile(x, reps)
    return x[:8760]

def as_int(val, default):
    try:
        s = str(val).strip()
        if not s:
            return default
        return int(float(s))
    except Exception:
        return default

def get_area_factors(year_of_construction, tabula_building_type, json_data):
    # Hole den Gebäudetyp (z.B. "AB", "SFH") aus der JSON-Daten
    if tabula_building_type not in json_data:
        raise ValueError(f"Gebäudetyp {tabula_building_type} nicht in den JSON-Daten gefunden.")

    building_type_data = json_data[tabula_building_type]

    # Iteriere durch die Baujahresgruppen und finde die passende Gruppe
    for year_range, factors in building_type_data.items():
        # Extrahiere das Jahr-Intervall (z.B. "1860-1918")
        start_year, end_year = map(int, year_range.split('-'))

        # Überprüfe, ob das Baujahr innerhalb des Intervalls liegt
        if start_year <= year_of_construction <= end_year:
            return factors

def read_lpg_template_year_elec_occ(template_results_dir: str, persons_in_household: int) -> tuple[np.ndarray, np.ndarray]:
    """
    template_results_dir: .../Results_ID_SH_<TEMPLATE>_2021/Results (Jahr fix egal)
    returns:
      electricity_kwh_h (8760)
      occupancy_pers_h  (8760)  = persons - outside(BAL)
    """
    d = Path(template_results_dir)

    # Electricity: SumProfiles_3600s.Electricity.csv
    el_path = d / "SumProfiles_3600s.Electricity.csv"
    el_df = pd.read_csv(el_path, sep=";")
    # typische Spalte ist "Sum [kWh]"
    sum_cols = [c for c in el_df.columns if isinstance(c, str) and c.startswith("Sum [")]
    vcol = "Sum [kWh]" if "Sum [kWh]" in el_df.columns else (sum_cols[0] if sum_cols else None)
    if vcol is None:
        raise ValueError(f"Keine 'Sum [...]'-Spalte in {el_path} gefunden.")
    electricity_kwh_h = _tile_8760(el_df[vcol].to_numpy(dtype=float))

    # Occupancy: BodilyActivityLevel.Outside.HH1.json (Minutenwerte)
    bal_path = d / "BodilyActivityLevel.Outside.HH1.json"
    bal = json.load(open(bal_path, encoding="utf-8"))
    outside_min = np.asarray(bal["Values"], dtype=float)

    hours = len(outside_min) // 60
    outside_h = outside_min[:hours * 60].reshape(hours, 60).mean(axis=1)
    inside_h = float(persons_in_household) - outside_h
    inside_h = np.clip(inside_h, 0.0, None)
    occupancy_pers_h = _tile_8760(inside_h)

    return electricity_kwh_h, occupancy_pers_h


def _resolve_zone_control(zone_name: str, zone_control: dict | None) -> dict:
    if not zone_control:
        return {"heated": True, "heat_setpoint_K": 294.15, "cooled": False, "cool_setpoint_K": 0.0}

    def _to_K(vC, vK, fallbackK):
        if vK is not None:
            return float(vK)
        if vC is not None:
            return float(vC) + 273.15
        return float(fallbackK)

    # 1) Exact
    zones = zone_control.get("zones", {}) or {}
    if zone_name in zones:
        zc = dict(zones[zone_name])
    else:
        # 2) Rules
        zc = None
        for rule in (zone_control.get("rules", []) or []):
            pat = rule.get("pattern")
            if pat and re.match(pat, zone_name):
                zc = dict(rule)
                break
        # 3) Default
        if zc is None:
            zc = dict(zone_control.get("default", {}) or {})

    # defaults
    heated = bool(zc.get("heated", True))
    cooled = bool(zc.get("cooled", False))

    heat_sp_K = _to_K(zc.get("heat_setpoint_C"), zc.get("heat_setpoint_K"), 294.15)
    cool_sp_K = _to_K(zc.get("cool_setpoint_C"), zc.get("cool_setpoint_K"), 299.15)  # default z.B. 26°C

    return {"heated": heated, "heat_setpoint_K": heat_sp_K,
            "cooled": cooled, "cool_setpoint_K": cool_sp_K}


def _apply_hvac_to_zone(tz, prj_data, heated: bool, heat_sp_K: float, cooled: bool, cool_sp_K: float):
    use_cond = UseConditions(parent=tz)
    use_cond.load_use_conditions("Living", prj_data)

    use_cond.with_ahu = False  # wie bisher

    # Heating
    use_cond.with_heating = bool(heated)
    use_cond.heating_profile = [float(heat_sp_K)] if heated else [0.0]

    # Cooling (nur setzen, wenn TEASER-UseConditions das Feld hat)
    if hasattr(use_cond, "with_cooling"):
        use_cond.with_cooling = bool(cooled)

    if hasattr(use_cond, "cooling_profile"):
        use_cond.cooling_profile = [float(cool_sp_K)] if cooled else [0.0]

    tz.use_conditions = use_cond


def _zone_index(zname: str) -> int | None:
    m = re.search(r"Zone_(\d+)$", zname)
    return int(m.group(1)) if m else None


def _is_core_zone(zname: str, n_cores: int) -> bool:
    idx = _zone_index(zname)
    return (idx is not None) and (1 <= idx <= n_cores)


def _apply_elec_occ_to_teaser_useconds(tz, electricity_kwh_h, occupancy_pers_h):
    """
    electricity_kwh_h: 8760 (kWh/h == kW)
    occupancy_pers_h:  8760 (Persons)
    -> setzt machines_profile/machines und persons_profile/persons
    """
    if not getattr(tz, "use_conditions", None):
        return
    uc = tz.use_conditions

    A = float(getattr(tz, "area", 0.0) or 0.0)
    if A <= 0:
        A = 1.0

    # Machines: kWh/h == kW -> W
    elec_W = np.asarray(electricity_kwh_h, dtype=float) * 1000.0
    peak_W = float(np.max(elec_W)) if elec_W.size else 0.0
    if peak_W > 0:
        machines_profile = np.clip(elec_W / peak_W, 0.0, 1.0)
        machines_W_m2 = peak_W / A
    else:
        machines_profile = np.zeros(8760, dtype=float)
        machines_W_m2 = 0.0

    if hasattr(uc, "machines_profile"):
        uc.machines_profile = machines_profile.tolist()
    if hasattr(uc, "machines"):
        uc.machines = float(machines_W_m2)

    # Persons
    occ = np.asarray(occupancy_pers_h, dtype=float)
    peak_p = float(np.max(occ)) if occ.size else 0.0
    if peak_p > 0:
        persons_profile = np.clip(occ / peak_p, 0.0, 1.0)
        persons_per_m2 = peak_p / A
    else:
        persons_profile = np.zeros(8760, dtype=float)
        persons_per_m2 = 0.0

    if hasattr(uc, "persons_profile"):
        uc.persons_profile = persons_profile.tolist()
    if hasattr(uc, "persons"):
        uc.persons = float(persons_per_m2)

    # optional: Licht fix aus
    if hasattr(uc, "lighting_profile"):
        uc.lighting_profile = [0.0] * 8760


def _pick_idx(building_id: str, zone_name: str, n: int) -> int:
    h = hashlib.sha256(f"{building_id}::{zone_name}".encode("utf-8")).hexdigest()
    return int(h[:8], 16) % max(n, 1)


def create_teaser_project(
    building_info, teaser_project, weather_path, tabula_df, construction=None,
    project_name='Generated_TEASER_Project', zone_control=None,

    # LPG templates
    use_lpg_templates: bool = False,
    templates_root: str | None = None,          # z.B. r"C:\03_Repos\SimData\LPG_SH\Results_SH_2021_1"
    zone_household_cfg: dict | None = None,     # Mapping + _available_templates
    default_persons_per_zone: int = 2,
):
    """
    Creates a TEASER project from the processed building data.

    Parameters
    ----------
    building_info: dict
        Building data dictionary generated by the main script.
    teaser_project: Project
        The TEASER project object to which this building is added.
    weather_path: string
        Path to .mos weather file
    tabula_df: string
        External data from heatbrain
    construction: string
        Type of construction "standard", "retrofit" or "adv_retrofit"
    project_name: str
        Name of the TEASER project to be created.

    Returns
    -------
    None
    """

    # Set building data source (tabula_de for Germany's TABULA data)
    used_statistic = 'tabula_de'
    if used_statistic is None:
        state_code = building_info['building_data'].get('addr:state_code')
        if state_code in ['BW', 'BY', 'HB', 'HH', 'HE', 'NI', 'NW', 'RP', 'SL',
                          'SH']:
            used_statistic = 'tabula_de'
        elif state_code in ['BB', 'MV', 'SN', 'ST', 'TH']:
            used_statistic = 'tabula_dd'
        elif state_code == 'BE':
            raise ValueError(f'please provide "used_statistic" explicitly for '
                             f'state_code {state_code}')
        else:
            raise ValueError('Unknown state code')
    prj = teaser_project
    prj.data = DataClass(used_statistic=used_statistic)

    # Set project name
    if project_name is None:
        prj.name = 'Mustergebäude'
    else:
        prj.name = project_name

    # Extract basic information about the building
    building_id = str(building_info['building_id']).strip()
    name = f"{building_id}"
    street_name = building_info['building_data'].get('addr:street', 'Musterstraße 1')
    city = building_info['building_data'].get('addr:suburb', 'Musterstadt')

    bd = building_info['building_data']

    # Set construction year based on TABULA classification
    tabula_year_class = as_int(
        bd.get("sa_tabula_year_class",
               bd.get("tabula_year_class", bd.get("bldg:tabula_year_class", 6))),
        6
    )

    year_of_construction = {
        1: 1850, 2: 1910, 3: 1930, 4: 1950, 5: 1960,
        6: 1970, 7: 1980, 8: 1990, 9: 2000, 10: 2005, 11: 2010, 12: 2020
    }.get(tabula_year_class, 1970)  # Default: 1970


    # Extract further building information such as floors, height, area
    number_of_floors = bd.get('bldg:number_of_storeys')
    height_of_floors = bd.get('bldg:storey_height')
    net_leased_area = bd.get('bldg:net_leased_area')

    # TABULA-Bautyp aus building_data
    building_type_hb = bd.get("tabula_building_type")
    tabula_building_type = (
        building_type_hb if building_type_hb == "TH"
        else bd.get("bldg:tabula_building_type", "SFH")
    )

    if tabula_building_type == 'SFH':
        tabula_usage = 'single_family_house'
        use_conds = "DINV18599_Living_SFH"
    elif tabula_building_type == 'MFH':
        tabula_usage = 'multi_family_house'
        use_conds = "DINV18599_Living_MFH"
    elif tabula_building_type == 'TH':
        tabula_usage = 'terraced_house'
        use_conds = "DINV18599_Living_SFH"
    elif tabula_building_type == 'AB':
        if 1860 <= year_of_construction <= 1978:
            tabula_usage = 'apartment_block'
        else:
            tabula_usage = 'multi_family_house'
        use_conds = "DINV18599_Living_MFH"
    else:
        tabula_usage = 'single_family_house'
        use_conds = "DINV18599_Living_SFH"

    # Retrofit-State ggf. aus building_data
    ret_state = str(
        bd.get("sa_retrofit_state", bd.get("bldg:retrofit_state", "standard"))
    ).strip()
    if not ret_state:
        ret_state = "standard"
    construction_type = f"tabula_{ret_state}"
    if construction is not None:
        construction_type = f"tabula_{construction}"

    setpoint_difference_for_exchange = 3
    # Check if the building has only one zone (single-zone building)
    if sum(len(storey['zones']) for storey in building_info['polygons']['storeys']) == 1:
        # If single zone, calculate number of floors and height
        number_of_floors = 1
        height_of_floors = 0 if np.isnan(building_info['building_data'].get('bldg:roof_edge_height', 0)) else \
        building_info[
            'building_data'].get('bldg:roof_edge_height', 0)

        # Add building to the TEASER project using residential method
        try:
            prj.add_residential(
                method=used_statistic,
                usage=tabula_usage,
                name=name,
                year_of_construction=year_of_construction,
                number_of_floors=number_of_floors,
                height_of_floors=height_of_floors,
                net_leased_area=net_leased_area,
                construction_type=construction_type,
                inner_wall_approximation_approach='teaser_default',
                internal_gains_mode=1
            )
        except RuntimeError as error:
            print(f'No building was added for {building_id} due to invalid TABULA usage')

        # Retrieve the last added building from the project
        building = prj.buildings[-1]

        # Initialize dictionaries for various building elements
        element_areas = dict()
        area_factors = building.facade_estimation_factors[building.building_age_group]
        building._outer_wall_names_1 = dict()
        building._outer_wall_names_2 = dict()
        building.window_names_1 = dict()
        building.window_names_2 = dict()
        building.interzonal_wall_names_1 = dict()
        building.interzonal_wall_names_2 = dict()
        building.roof_names_1 = dict()
        building.roof_names_2 = dict()
        building.ground_floor_names_1 = dict()
        building.ground_floor_names_2 = dict()

        def add_area(target_dict, key, area):
            """Addiert die Fläche, falls Key bereits existiert"""
            target_dict[key] = target_dict.get(key, 0) + area

        # Process each storey and zone to calculate walls, windows, and roof areas
        for storey in building_info['polygons']['storeys']:
            add_interzonal = False  # Flag for interzonal walls
            for zone in storey['zones']:
                total_wall_area = sum(
                    [wall[1][0] for wall in zone['walls']])

                adjacent_areas = zone.get('adjacent_areas', {})
                interzonal_adjacent_buildings = adjacent_areas.get('interzonal_adjacent_buildings', [])

                # Process all walls in the zone
                for wall in zone['walls']:
                    shared_area = 0
                    wall_area = wall[1][0]  # Total wall area
                    orientation = wall[1][2]

                    for adjacent_building in interzonal_adjacent_buildings:
                        if adjacent_building.get('wall')[0] == wall[0]:
                            shared_area = adjacent_building.get('area')
                            element_areas[f"AdjacentWall_{adjacent_building['target_building']}"] = shared_area

                    # Calculate open wall area (after subtracting shared area)
                    area_open = wall_area - shared_area

                    # Sum factors for outer walls and windows
                    sum_ow_win_factors = (
                                area_factors["ow1"] + area_factors["ow2"] + area_factors["win1"] + area_factors["win2"])

                    # Distribute wall areas across outer walls and windows
                    for open_type, name_dict, base in zip(
                            ["ow1", "ow2", "win1", "win2"],
                            [building._outer_wall_names_1,
                             building._outer_wall_names_2,
                             building.window_names_1,
                             building.window_names_2],
                            ["ExteriorFacade", "ExteriorFacade", "WindowFacade", "WindowFacade"]
                    ):
                        if area_open <= 0 and "win" in open_type:
                            continue

                        area = area_open * area_factors[open_type] / sum_ow_win_factors

                        if area <= 0:
                            continue

                        key = f"{base}_{orientation}_{open_type[-1]}"  # z. B. ExteriorFacade_90_1
                        name_dict[key] = [90.0, orientation]
                        add_area(element_areas, key, area)

                    # Process interzonal walls, if any shared area exists
                    if shared_area > 0:
                        for adj_idx, adjacent_building in enumerate(interzonal_adjacent_buildings):
                            if adjacent_building.get('wall')[0] == wall[0]:
                                for izw_type, name_dict, element_name in zip(
                                        ["ow1", "ow2"],
                                        [building.interzonal_wall_names_1,
                                         building.interzonal_wall_names_2],
                                        [f"Izw_{wall[1][2]}_{adj_idx}_1",
                                         f"Izw_{wall[1][2]}_{adj_idx}_2"]):
                                    # Set interzonal wall properties
                                    name_dict[element_name] = [90.0, wall[1][2], adj_idx, shared_area]

                for i, (_, ceiling) in enumerate(zone['ceilings']):
                    area, tilt, orientation = ceiling
                    roof_pitch_angle = 0.0 if abs(tilt - 180) < 0.1 else tilt

                    # Korrekte Formeln
                    denom = area_factors["rt1"] + area_factors["rt2"]
                    area_roof1 = area * area_factors["rt1"] / denom
                    area_roof2 = area * area_factors["rt2"] / denom  # rt2!

                    if area_roof1 > 0:
                        key = f"Rooftop_{orientation}_1"
                        building.roof_names_1[key] = [roof_pitch_angle, orientation]
                        add_area(element_areas, key, area_roof1)

                    if area_roof2 > 0:
                        key = f"Rooftop_{orientation}_2"
                        building.roof_names_2[key] = [roof_pitch_angle, orientation]
                        add_area(element_areas, key, area_roof2)

                # Ground floor area calculation for each zone
                for i, (_, floor) in enumerate(zone['floors']):
                    area, tilt, orientation = floor
                    denom = area_factors["gf1"] + area_factors["gf2"]
                    area_gf1 = area * area_factors["gf1"] / denom
                    area_gf2 = area * area_factors["gf2"] / denom

                    if area_gf1 > 0:
                        key = f"GroundFloor_{orientation}_1"
                        building.ground_floor_names_1[key] = [0.0, orientation]
                        add_area(element_areas, key, area_gf1)

                    if area_gf2 > 0:
                        key = f"GroundFloor_{orientation}_2"
                        building.ground_floor_names_2[key] = [0.0, orientation]
                        add_area(element_areas, key, area_gf2)

        # Remove doors from the building model (for simplicity)
        building.door_names = {}

        # Re-generate the building archetype with newly calculated elements
        building.generate_archetype()
        thermal_zone = building.thermal_zones[0]
        thermal_zone.name = "ThermalZone"

        # Set the areas of the building elements (walls, windows, roofs, floors)
        element_areas_teaser = dict()
        for key, value in element_areas.items():
            element_areas_teaser[teaser_name_converter(key)] = value
        for element in (thermal_zone.outer_walls
                        + thermal_zone.windows
                        + thermal_zone.rooftops
                        + thermal_zone.ground_floors):
            element.area = element_areas_teaser[element.name]

        # Set use conditions and configure heating based on building state
        heating_state = True
        if heating_state == 'False' or heating_state is False:
            use_cond = UseConditions(parent=thermal_zone)
            use_cond.load_use_conditions("Living", prj.data)
            use_cond.with_heating = False
            thermal_zone.use_conditions = use_cond
            thermal_zone.use_conditions.with_ahu = False
        else:
            use_cond = UseConditions(parent=thermal_zone)
            use_cond.load_use_conditions("Living", prj.data)
            use_cond.heating_profile = 293.15
            thermal_zone.use_conditions = use_cond


        # Optionally add interzonal walls
        if add_interzonal:
            for izw_dict, construction_type in zip(
                    [building.interzonal_wall_names_1, building.interzonal_wall_names_2],
                    [building._construction_type_1, building._construction_type_2]):
                for izw_name, izw_values in izw_dict.items():
                    interzonal_wall = InterzonalWall(thermal_zone, other_side=None)
                    interzonal_wall.load_type_element(
                        year=interzonal_wall.year_of_construction,
                        construction=construction_type,
                        data_class=building.parent.data
                    )
                    interzonal_wall.name = izw_name
                    interzonal_wall.tilt = izw_values[0]
                    interzonal_wall.orientation = izw_values[1]
                    interzonal_wall.area = izw_values[3]
                    interzonal_wall.interzonal_type_export = "outer_ordered"
                    interzonal_wall.interzonal_type_material = "outer_ordered"

        # Set minimum eaves height and roof type
        roof_height = building_info['building_data'].get('bldg:measuredHeight', 0)
        building.minimal_eaves_height_above_ground = building_info['building_data'].get('bldg:roof_edge_height', 0)
        building.roof_type_code = str(building_info['building_data'].get('bldg:roofType', 'default'))

        # Calculate the inner wall areas based on floor areas and zones
        innerWallFactor = 1.0
        thermal_zone.set_inner_wall_area()

        # Set the total volume of the thermal zone
        thermal_zone.total_volume = building_info['building_data'].get('bldg:volume', 0)
        V_e = building_info['building_data'].get('bldg:volume', 0)
        if building.number_of_floors <= 3:
            thermal_zone.volume = V_e * 0.76
        else:
            thermal_zone.volume = V_e * 0.8

        # Perform final calculations for building elements (AixLib library)
        building.calc_building_parameter(number_of_elements=4, used_library="AixLib")
        prj.weather_file_path = utilities.get_full_path(
            weather_path)


    ##################################################################################

    ############################   Multizoning approach   ############################

    ##################################################################################
    else:

        number_of_floors = number_of_floors
        height_of_floors = height_of_floors
        n_cores = int(building_info.get("building_data", {}).get("bldg:n_cores", 1))

        # Add building to the TEASER project using residential method
        try:
            building = Building(parent=prj)
            building.method = used_statistic
            building.usage = tabula_usage
            building.name = name
            building.year_of_construction = year_of_construction
            building.number_of_floors = number_of_floors
            building.height_of_floors = height_of_floors
            building.net_leased_area = net_leased_area
            construction_type = construction_type
            building.internal_gains_mode = 1
            # attic=1,
            # cellar=1
        except RuntimeError as error:
            print(f'No building was added for {building_id} due to invalid TABULA usage')

        # Retrieve the last added building from the project
        building = prj.buildings[-1]

        # Initialize dictionaries for various building elements
        element_areas = dict()
        # Pfad zur JSON-Datei
        # Relativer Pfad zur JSON-Datei
        json_file_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "..", "..", "data", "geodata", "tabula_area_factors.json"
        )
        with open(json_file_path, 'r') as json_file:
            json_data = json.load(json_file)
        try:
            area_factors = get_area_factors(year_of_construction, tabula_building_type, json_data)
            # area_factors = building.facade_estimation_factors[building.building_age_group]
            # print(f'Area factors for {year_of_construction} and {tabula_building_type}: {area_factors}')
        except ValueError as e:
            print(e)
            area_factors = None

        # Remove doors from the building model (for simplicity)
        building.door_names = {}
        building.inner_wall_approximation_approach = 'typical_minus_outer_extended'

        # Process each storey and zone to calculate walls, windows, and roof
        zone_objects = {}
        _lpg_cache = {}
        j = 0
        for storey in building_info["polygons"]["storeys"]:
            for zone in storey["zones"]:

                # statt nur Zone_1: alle Core-Zonen (Zone_1..Zone_n_cores) ab j!=0 skippen
                if _is_core_zone(zone["name"], n_cores) and j != 0:
                    continue

                tz = ThermalZone(parent=building)
                tz.name = zone["name"]
                tz.area = zone["floors"][0][1][0]
                tz.volume = zone["volume"]
                tz.infiltration_rate = 0.5

                # NEU: Heizung/Setpoint aus zone_control anwenden
                zc = _resolve_zone_control(tz.name, zone_control)  # zone_control muss als Parameter übergeben werden
                _apply_hvac_to_zone(
                    tz, prj.data,
                    zc["heated"], zc["heat_setpoint_K"],
                    zc["cooled"], zc["cool_setpoint_K"]
                )

                if use_lpg_templates:
                    if not templates_root:
                        raise ValueError("templates_root fehlt (Root mit Results_ID_SH_<TEMPLATE>_2021)")
                    zone_household_cfg = zone_household_cfg or {}

                    if _is_core_zone(tz.name, n_cores):
                        _apply_elec_occ_to_teaser_useconds(tz, np.zeros(8760), np.zeros(8760))
                    else:
                        bldg_id = str(building_info.get("building_id", "unknown"))
                        zname = tz.name

                        zcfg = zone_household_cfg.get(zname, {})
                        template_name = zcfg.get("template_name")

                        if not template_name:
                            available = zone_household_cfg.get("_available_templates", [])
                            if not available:
                                raise ValueError("... _available_templates ...")
                            template_name = available[_pick_idx(bldg_id, zname, len(available))]

                        hh_sizes = zone_household_cfg.get("_household_sizes", {})
                        persons = int(zcfg.get("persons", hh_sizes.get(template_name, default_persons_per_zone)))

                        template_code = str(template_name).split()[0]  # -> "CHR50"
                        template_dir = str(
                            Path(templates_root) / f"Results_ID_SH_{template_code}_2021" / "Results"
                        )

                        key = (template_dir, persons)
                        if key not in _lpg_cache:
                            elec, occ = read_lpg_template_year_elec_occ(template_dir, persons_in_household=persons)
                            _lpg_cache[key] = (elec, occ)

                        elec, occ = _lpg_cache[key]
                        _apply_elec_occ_to_teaser_useconds(tz, elec, occ)

                zone_objects[tz.name] = tz

                if j == 0:
                    # Ground floor area calculation for each zone if zone is in first floor
                    for i, (floor_polygon, floor_data) in enumerate(zone['floors']):
                        area_ground1 = (
                                floor_data[0] * area_factors["rt1"] / (area_factors["rt1"] + area_factors["rt2"]))
                        area_ground2 = (
                                floor_data[0] * area_factors["rt2"] / (area_factors["rt1"] + area_factors["rt2"]))
                        if area_ground1 > 0:
                            ground = GroundFloor(parent=tz)
                            ground.name = f"GroundFloor_{tz.name}_{i + 1}_1"
                            ground.load_type_element(year=building.year_of_construction,
                                                     construction=f'{construction_type}_1_{tabula_building_type}')
                            ground.area = area_ground1
                            ground.tilt = floor_data[1]
                            ground.orientation = floor_data[2]
                        if area_ground2 > 0:
                            ground = GroundFloor(parent=tz)
                            ground.name = f"GroundFloor_{tz.name}_{i + 1}_2"
                            ground.load_type_element(year=building.year_of_construction,
                                                     construction=f'{construction_type}_2_{tabula_building_type}')
                            ground.area = area_ground2
                            ground.tilt = floor_data[1]
                            ground.orientation = floor_data[2]

                elif j == len(building_info['polygons']['storeys']) - 1:
                    for i, (ceiling_polygon, ceiling_data) in enumerate(zone['ceilings']):
                        area_roof1 = (
                                    ceiling_data[0] * area_factors["rt1"] / (area_factors["rt1"] + area_factors["rt2"]))
                        area_roof2 = (
                                    ceiling_data[0] * area_factors["rt2"] / (area_factors["rt1"] + area_factors["rt2"]))
                        if area_roof1 > 0:
                            # Creating rooftop properties
                            rooftop = Rooftop(parent=tz)
                            rooftop.name = f"Rooftop_{tz.name}_{i + 1}_1"
                            rooftop.load_type_element(year=building.year_of_construction,
                                                      construction=f'{construction_type}_1_{tabula_building_type}')
                            rooftop.area = area_roof1
                            rooftop.tilt = ceiling_data[1]
                            rooftop.orientation = ceiling_data[2]

                        if area_roof2 > 0:
                            # Creating rooftop properties
                            rooftop = Rooftop(parent=tz)
                            rooftop.name = f"Rooftop_{tz.name}_{i + 1}_2"
                            rooftop.load_type_element(year=building.year_of_construction,
                                                      construction=f'{construction_type}_2_{tabula_building_type}')
                            rooftop.area = area_roof2
                            rooftop.tilt = ceiling_data[1]
                            rooftop.orientation = ceiling_data[2]

                # Walls and windows
                adjacent_areas = zone.get('adjacent_areas', {})
                interzonal_adjacent_buildings = adjacent_areas.get('interzonal_adjacent_buildings', [])

                interzonal_adjacent_walls = adjacent_areas.get('interzonal_walls', [])
                interzonal_walls_dict = {adjacent_wall['wall'][0]: adjacent_wall for adjacent_wall in
                                         interzonal_adjacent_walls}

                # Process all walls in the zone
                w = 1
                for wall in zone['walls']:
                    shared_area = 0
                    wall_area = wall[1][0]
                    wall_tilt = wall[1][1]
                    wall_orientation = wall[1][2]
                    wall_polygon = wall[0]

                    adjacent_wall = interzonal_walls_dict.get(wall_polygon)
                    if adjacent_wall:
                        continue

                    # OuterWalls
                    else:
                        for adjacent_building in interzonal_adjacent_buildings:
                            if adjacent_building.get('wall')[0] == wall[0]:
                                shared_area = adjacent_building.get('area')
                                element_areas[f"AdjacentWall_{adjacent_building['target_building']}"] = shared_area

                        # Calculate open wall area (after subtracting shared area)
                        area_open = wall_area - shared_area

                        # Sum factors for outer walls and windows
                        sum_ow_win_factors = (area_factors["ow1"] + area_factors["ow2"]
                                              + area_factors["win1"] + area_factors["win2"])

                        # Distribute wall areas across outer walls and windows
                        for open_type in ["ow1", "ow2", "win1", "win2"]:
                            if area_open > 0:
                                # Calculate respective area for outer walls and windows
                                area = (area_open * area_factors[open_type] / sum_ow_win_factors)
                            else:
                                if "win" in open_type:
                                    area = 0  # No window area if there is no open wall
                                else:
                                    area = (wall_area * area_factors[open_type] /
                                            area_factors["ow1"] + area_factors["ow2"])

                            if area > 0:
                                if "ow1" in open_type:
                                    outer_wall = OuterWall(parent=tz)
                                    outer_wall.name = f"OuterWall_{tz.name}_{w}_1"
                                    outer_wall.load_type_element(year=building.year_of_construction,
                                                                 construction=f'{construction_type}_1_{tabula_building_type}')
                                    outer_wall.area = area
                                    outer_wall.tilt = wall_tilt
                                    outer_wall.orientation = wall_orientation
                                if "ow2" in open_type:
                                    outer_wall = OuterWall(parent=tz)
                                    outer_wall.name = f"OuterWall_{tz.name}_{w}_2"
                                    outer_wall.load_type_element(year=building.year_of_construction,
                                                                 construction=f'{construction_type}_2_{tabula_building_type}')
                                    outer_wall.area = area
                                    outer_wall.tilt = wall_tilt
                                    outer_wall.orientation = wall_orientation
                                elif "win1" in open_type:
                                    win = Window(parent=tz)
                                    win.name = f"Window_{tz.name}_{w}_1"
                                    win.load_type_element(year=building.year_of_construction,
                                                          construction=f'{construction_type}_1_{tabula_building_type}')
                                    win.area = area
                                    win.tilt = wall_tilt
                                    win.orientation = wall_orientation
                                elif "win2" in open_type:
                                    win = Window(parent=tz)
                                    win.name = f"Window_{tz.name}_{w}_2"
                                    win.load_type_element(year=building.year_of_construction,
                                                          construction=f'{construction_type}_2_{tabula_building_type}')
                                    win.area = area
                                    win.tilt = wall_tilt
                                    win.orientation = wall_orientation

                        """
                        # Process interzonal walls, if any shared area exists - not considered yet
                        if shared_area > 0:
                            for adj_idx, adjacent_building in enumerate(interzonal_adjacent_buildings):
                                if adjacent_building.get('wall')[0] == wall[0]:
                                    for izw_type, name_dict, element_name in zip(
                                            ["ow1", "ow2"],
                                            [building.interzonal_wall_names_1,
                                             building.interzonal_wall_names_2],
                                            [f"Izw_{wall[1][2]}_{adj_idx}_1",
                                             f"Izw_{wall[1][2]}_{adj_idx}_2"]):
                                        # Set interzonal wall properties
                                        name_dict[element_name] = [90.0, wall[1][2], adj_idx, shared_area]

                        """
                    w += 1

            j += 1

        # Processing interzonal elements
        k = 0
        for storey_k in building_info['polygons']['storeys']:
            setpoint_difference_for_exchange = 3
            for zone in storey_k["zones"]:
                if _is_core_zone(zone["name"], n_cores) and k != 0:
                    continue
                tz = zone_objects[zone["name"]]
                adjacent_areas = zone.get('adjacent_areas', {})

                interzonal_adjacent_ceilings = adjacent_areas.get('interzonal_ceilings', [])
                interzonal_ceilings_dict = {adjacent_ceiling['ceiling'][0]: adjacent_ceiling for adjacent_ceiling in
                                            interzonal_adjacent_ceilings}

                interzonal_adjacent_walls = adjacent_areas.get('interzonal_walls', [])
                interzonal_walls_dict = {adjacent_wall['wall'][0]: adjacent_wall for adjacent_wall in
                                         interzonal_adjacent_walls}

                interzonal_adjacent_floors = adjacent_areas.get('interzonal_floors', [])
                interzonal_floors_dict = {adjacent_floor['floor'][0]: adjacent_floor for adjacent_floor in
                                          interzonal_adjacent_floors}

                for wall in zone['walls']:
                    shared_area = 0
                    wall_area = wall[1][0]
                    wall_tilt = wall[1][1]
                    wall_orientation = wall[1][2]
                    wall_polygon = wall[0]

                    adjacent_wall = interzonal_walls_dict.get(wall_polygon)
                    if adjacent_wall:
                        other_zone_name = adjacent_wall.get("target_zone", None)
                        if other_zone_name in zone_objects:
                            other_zone = zone_objects[other_zone_name]
                            if tz.use_conditions.heating_profile[0] and \
                                    other_zone.use_conditions.heating_profile[0]:
                                if (tz.use_conditions.heating_profile[0] -
                                        other_zone.use_conditions.heating_profile[0]
                                        >= setpoint_difference_for_exchange):
                                    type_export = "outer_ordered"
                                elif (tz.use_conditions.heating_profile[0] -
                                      other_zone.use_conditions.heating_profile[
                                          0]
                                      <= (-setpoint_difference_for_exchange)):
                                    type_export = "outer_reversed"
                                else:
                                    type_export = "inner"

                            else:
                                type_export = 'inner'
                                if ((other_zone.use_conditions.with_heating and tz.use_conditions.with_heating)
                                        or (not other_zone.use_conditions.with_heating
                                            and not tz.use_conditions.with_heating)):
                                    type_export = 'inner'
                                elif tz.use_conditions.with_heating is True:
                                    type_export = 'outer_ordered'
                                else:  # this_use.with_heating is False:
                                    type_export = 'outer_reversed'

                            if type_export == 'outer_ordered' or type_export == 'outer_reversed':
                                interzonal_wall = InterzonalWall(parent=tz, other_side=zone_objects[other_zone_name])
                                interzonal_wall.name = f"InterzonalWall_{tz.name}_to_{other_zone_name}"
                                iz_constr_1 = f"{construction_type}_1_{tabula_building_type}"
                                interzonal_wall.load_type_element(
                                    year=building.year_of_construction,
                                    construction=iz_constr_1
                                )
                                interzonal_wall.area = wall_area
                                interzonal_wall.tilt = wall_tilt
                                interzonal_wall.orientation = wall_orientation
                                interzonal_wall.interzonal_type_export = type_export
                                interzonal_wall.interzonal_type_material = type_export

                            else:
                                inner_wall = InnerWall(parent=tz)
                                inner_wall.name = f"InnerWall_{tz.name}_to_{other_zone_name}"
                                inner_wall.load_type_element(year=building.year_of_construction,
                                                             construction=f"tabula_standard")
                                inner_wall.area = wall_area
                                inner_wall.tilt = wall_tilt
                                inner_wall.orientation = wall_orientation

                if k != 0:
                    for floor in zone['floors']:
                        shared_area = 0
                        floor_area = floor[1][0]
                        floor_tilt = floor[1][1]
                        floor_orientation = floor[1][2]
                        floor_polygon = floor[0]

                        adjacent_ceiling = interzonal_floors_dict.get(floor_polygon)
                        if adjacent_ceiling:
                            other_zone_name = adjacent_ceiling.get("target_zone", None)
                            if other_zone_name in zone_objects:
                                other_zone = zone_objects[other_zone_name]
                                if tz.use_conditions.heating_profile[0] and \
                                        other_zone.use_conditions.heating_profile[0]:
                                    if (tz.use_conditions.heating_profile[0] -
                                            other_zone.use_conditions.heating_profile[0]
                                            >= setpoint_difference_for_exchange):
                                        type_export = "outer_ordered"
                                    elif (tz.use_conditions.heating_profile[0] -
                                          other_zone.use_conditions.heating_profile[0]
                                          <= (-setpoint_difference_for_exchange)):
                                        type_export = "outer_reversed"
                                    else:
                                        type_export = "inner"

                                else:
                                    type_export = 'inner'
                                    if ((other_zone.use_conditions.with_heating and tz.use_conditions.with_heating)
                                            or (not other_zone.use_conditions.with_heating
                                                and not tz.use_conditions.with_heating)):
                                        type_export = 'inner'
                                    elif tz.use_conditions.with_heating is True:
                                        type_export = 'outer_ordered'
                                    else:  # this_use.with_heating is False:
                                        type_export = 'outer_reversed'

                                if type_export == 'outer_ordered' or type_export == 'outer_reversed':
                                    interzonal_floor = InterzonalFloor(parent=tz,
                                                                       other_side=zone_objects[other_zone_name])
                                    interzonal_floor.name = f"InterzonalFloor_{tz.name}_to_{other_zone_name}"
                                    iz_constr_1 = f"{construction_type}_1_{tabula_building_type}"
                                    interzonal_floor.load_type_element(
                                        year=building.year_of_construction,
                                        construction=iz_constr_1
                                    )

                                    interzonal_floor.area = floor_area
                                    interzonal_floor.tilt = floor_tilt
                                    interzonal_floor.orientation = floor_orientation
                                    interzonal_floor.interzonal_type_export = type_export
                                    interzonal_floor.interzonal_type_material = type_export

                                else:
                                    inner_floor = Floor(parent=tz)
                                    inner_floor.name = f"Floor_{tz.name}_to_{other_zone_name}"
                                    inner_floor.load_type_element(year=building.year_of_construction,
                                                                  construction=f"tabula_standard")
                                    inner_floor.area = floor_area
                                    inner_floor.tilt = floor_tilt
                                    inner_floor.orientation = floor_orientation

                        else:
                            inner_floor = Floor(parent=tz)
                            inner_floor.name = f"Floor_{tz.name}_to_{uuid.uuid4()}"
                            inner_floor.load_type_element(year=building.year_of_construction,
                                                          construction=f"tabula_standard")
                            inner_floor.area = floor_area
                            inner_floor.tilt = floor_tilt
                            inner_floor.orientation = floor_orientation

                if k != len(building_info['polygons']['storeys']) - 1:
                    for ceiling in zone['ceilings']:
                        shared_area = 0
                        ceiling_area = ceiling[1][0]
                        ceiling_tilt = ceiling[1][1]
                        ceiling_orientation = ceiling[1][2]
                        ceiling_polygon = ceiling[0]

                        adjacent_floor = interzonal_ceilings_dict.get(ceiling_polygon)
                        if adjacent_floor:
                            other_zone_name = adjacent_floor.get("target_zone", None)
                            if other_zone_name in zone_objects:
                                other_zone = zone_objects[other_zone_name]
                                if tz.use_conditions.heating_profile[0] and \
                                        other_zone.use_conditions.heating_profile[0]:
                                    if (tz.use_conditions.heating_profile[0] -
                                            other_zone.use_conditions.heating_profile[0]
                                            >= setpoint_difference_for_exchange):
                                        type_export = "outer_ordered"
                                    elif (tz.use_conditions.heating_profile[0] -
                                          other_zone.use_conditions.heating_profile[0]
                                          <= (-setpoint_difference_for_exchange)):
                                        type_export = "outer_reversed"
                                    else:
                                        type_export = "inner"

                                else:
                                    type_export = 'inner'
                                    if ((other_zone.use_conditions.with_heating and tz.use_conditions.with_heating)
                                            or (not other_zone.use_conditions.with_heating
                                                and not tz.use_conditions.with_heating)):
                                        type_export = 'inner'
                                    elif tz.use_conditions.with_heating is True:
                                        type_export = 'outer_ordered'
                                    else:  # this_use.with_heating is False:
                                        type_export = 'outer_reversed'

                                if type_export == 'outer_ordered' or type_export == 'outer_reversed':
                                    interzonal_ceiling = InterzonalCeiling(parent=tz,
                                                                           other_side=zone_objects[other_zone_name])
                                    interzonal_ceiling.name = f"InterzonalCeiling_{tz.name}_to_{other_zone_name}"
                                    iz_constr_1 = f"{construction_type}_1_{tabula_building_type}"
                                    interzonal_ceiling.load_type_element(
                                        year=building.year_of_construction,
                                        construction=iz_constr_1
                                    )

                                    interzonal_ceiling.area = ceiling_area
                                    interzonal_ceiling.tilt = ceiling_tilt
                                    interzonal_ceiling.orientation = ceiling_orientation
                                    interzonal_ceiling.interzonal_type_export = type_export
                                    interzonal_ceiling.interzonal_type_material = type_export

                                else:
                                    inner_ceiling = Ceiling(parent=tz)
                                    inner_ceiling.name = f"Ceiling_{tz.name}_to_{other_zone_name}"
                                    inner_ceiling.load_type_element(year=building.year_of_construction,
                                                                    construction=f"tabula_standard")
                                    inner_ceiling.area = ceiling_area
                                    inner_ceiling.tilt = ceiling_tilt
                                    inner_ceiling.orientation = ceiling_orientation

                        else:
                            inner_ceiling = Ceiling(parent=tz)
                            inner_ceiling.name = f"Ceiling_{tz.name}_{uuid.uuid4()}"
                            inner_ceiling.load_type_element(year=building.year_of_construction,
                                                            construction=f"tabula_standard")
                            inner_ceiling.area = ceiling_area
                            inner_ceiling.tilt = ceiling_tilt
                            inner_ceiling.orientation = ceiling_orientation

                # Adding additional interior Walls according to set_inner_walls method for archetypes
                calculate_additional_inner_walls(tz, building)

                # Calculate the inner wall areas based on floor areas and zones
                # innerWallFactor = 1.0
                # tz.number_of_floors = 1
                # tz.set_inner_wall_area()

                # Set the total volume of the thermal zone (for later adjustments)
                # V_e = zone["volume"]
                # if building.number_of_floors <= 3:
                #   tz.volume = V_e * 0.76
                # else:
                #   tz.volume = V_e * 0.8
            k += 1

        # Set minimum eaves height and roof type
        roof_height = building_info['building_data'].get('bldg:measuredHeight', 0)
        building.minimal_eaves_height_above_ground = building_info['building_data'].get('bldg:roof_edge_height', 0)
        building.roof_type_code = str(building_info['building_data'].get('bldg:roofType', 'default'))

        # Perform final calculations for building elements (AixLib library)
        building.calc_building_parameter(number_of_elements=5, used_library="AixLib")

        # prj.t_soil_file_path = utilities.get_full_path(
        #   "data/input/inputdata/weatherdata/t_soil_example_e10.txt")

        # 2. the weather is set to the actual weather at the time
        prj.weather_file_path = utilities.get_full_path(
            weather_path)

    return prj


def teaser_name_converter(element_name):
    """
    Converts a string to TEASER-compatible naming conventions (alphanumeric only).

    TEASER names can only contain alphanumeric characters due to Modelica restrictions.
    This function mirrors TEASER behavior by removing all other signs from a string.

    Parameters
    ----------
    element_name: str
        The name to be used for an object in TEASER.

    Returns
    -------
    element_name_teaser: str
        A string representation of element_name with alphanumerical signs only.
    """
    if isinstance(element_name, str):
        regex = re.compile("[^a-zA-z0-9]")
        element_name_teaser = regex.sub("", element_name)
    else:
        try:
            value = str(element_name)
            regex = re.compile("[^a-zA-z0-9]")
            element_name_teaser = regex.sub("", value)
        except ValueError:
            print("Can't convert name to string")

    if element_name_teaser[0].isdigit():
        element_name_teaser = "B" + element_name_teaser
    return element_name_teaser


def calculate_additional_inner_walls(zone, building):
    """
    Berechnet und fügt zusätzliche innere Wände für die Zone hinzu,
    falls die Zonengröße mindestens 30 m² beträgt.
    """
    if zone.area < 30:
        return

    typical_area = zone.use_conditions.typical_length * zone.use_conditions.typical_width
    avg_room_nr = zone.area / typical_area

    approximation_approach = building.inner_wall_approximation_approach
    if approximation_approach not in ('teaser_default', 'typical_minus_outer', 'typical_minus_outer_extended'):
        warnings.warn(
            f'Inner wall approximation approach {approximation_approach} unknown. Falling back to teaser_default.')
        approximation_approach = 'teaser_default'

    if approximation_approach == 'typical_minus_outer':
        wall_area = ((int(avg_room_nr) + math.sqrt(avg_room_nr - int(avg_room_nr))) *
                     (2 * zone.use_conditions.typical_length * zone.height_of_floors +
                      2 * zone.use_conditions.typical_width * zone.height_of_floors))
        for other_verticals in zone.outer_walls + zone.interzonal_walls + zone.windows + zone.doors:
            wall_area -= other_verticals.area
        wall_area = max(0.01, wall_area)

    elif approximation_approach == 'typical_minus_outer_extended':
        wall_area = ((int(avg_room_nr) + math.sqrt(avg_room_nr - int(avg_room_nr))) *
                     (2 * zone.use_conditions.typical_length * zone.height_of_floors +
                      2 * zone.use_conditions.typical_width * zone.height_of_floors))
        for other_verticals in zone.outer_walls + zone.interzonal_walls + zone.doors:
            wall_area -= other_verticals.area
        for pot_vert_be in zone.rooftops + zone.windows:
            wall_area -= pot_vert_be.area * math.sin(pot_vert_be.tilt * math.pi / 180)
        wall_area -= max(0.01, sum(gf.area for gf in zone.ground_floors) - zone.area)
        wall_area = max(0.01, wall_area)
        wall_area = wall_area/2

    else:
        wall_area = avg_room_nr * (zone.use_conditions.typical_length * zone.height_of_floors +
                                   2 * zone.use_conditions.typical_width * zone.height_of_floors)

    additional_wall = InnerWall(parent=zone)
    additional_wall.name = f"AdditionalInnerWall_{zone.name}"
    additional_wall.load_type_element(year=building.year_of_construction, construction="tabula_standard")
    additional_wall.area = wall_area
    additional_wall.tilt = 90
    additional_wall.orientation = 0

    zone.inner_walls.append(additional_wall)
