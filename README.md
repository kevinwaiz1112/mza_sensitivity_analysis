"""
README — Sensitivity Analysis Step 2 (Main Block)

This project runs the Step 2 “main block” of the sensitivity analysis:
- compares 8 model variants (V1–V8),
- propagates input uncertainties (YoC, setpoints, WWR, weather, gains),
- quantifies LPG usage stochasticity via multiple seeds,
- exports and simulates a fresh TEASER/AixLib model per run,
- saves time series + full run configuration per run.

──────────────────────────────────────────────────────────────────────────────
FILES (what they do)

1) sa_head.py
   - Defines variants V1–V8 (zones/apartments assumptions).
   - Loads building payloads (PKL preferred).
   - Draws uncertainty samples:
     * YoC (TABULA year class): 83% correct, 17% neighbor class
     * Setpoints: T_mean ~ N(T0, sigma), sigma in [1..2] K; T_spread in [0.5..2] K
       -> per zone setpoints around T_mean
     * WWR factor: multiplicative (default 0.8–1.2)
     * Weather: categorical (TRY_A/TRY_B/WeekCold/WeekHot)
     * Gains scale: multiplicative (default 0.8–1.2)
   - Creates tasks for the runner (sim_wrapper.py):
     Variant × Building × Sample × Seed
   - DRY_RUN mode: only builds and validates tasks (no TEASER/Dymola).

2) sim_wrapper.py (runner)
   For each task:
   A) Build TEASER model (every run)
      - Uses building_payload["building_data"]["sa_tabula_year_class"] (YoC) to set construction year.
      - Uses building_payload["building_data"]["sa_zone_control"] to set zonal heating setpoints in TEASER.
      - Exports AixLib model into: sim_models_dir/<formatted_id>/...

   B) Patch the exported Zone Records (Database)
      - WWR: scales in every zone record:
          AWin = {...}         -> scaled by wwr_factor
          ATransparent = {...} -> scaled by wwr_factor
      - LPG flags per zone record:
          TH zone (zone index 0): use_lpg_people/machines/light = false
          residential zones:      people/machines true, light false
      - Optional: further record overrides via sa_params.record_overrides_global / by_zone

   C) Write input tables (8760h)
      - Internal gains table: people/machines/lights/occ_rel per zone (from LPG templates, seed-driven)
      - Setpoint tables: TsetHeat_*.txt / TsetCool_*.txt (constant), derived from zone_control

   D) Weather patch
      - Updates the Modelica file reference to the chosen .mos

   E) Simulate (Dymola via ebcpy)
      - Runs the model and reads results from the result file

   F) Save outputs per run
      - model_export/        snapshot of the *exact model used* (TEASER export + all patches)
      - timeseries.csv       zoned outputs; columns depend on n_zones
      - zone_map.json        mapping of zone indices to zone record names
      - overall.json         KPIs + full run configuration (task_meta, sa_params, lpg_cfg, zone_control, paths)

3) teaser_export.py
   - Implements create_teaser_project(...):
     builds TEASER thermal zones and elements and applies zone_control setpoints.
   - Uses sa_tabula_year_class if present; otherwise falls back to original tabula_year_class.

4) utils.py
   - LPG selection + template utilities
   - ID helpers (to_dashed_id)
   - Weather reference patching (parse_weather_and_update_reference)
   - building_model_exists checks

──────────────────────────────────────────────────────────────────────────────
OUTPUT STRUCTURE (per run)

OUT_BASE/
  <VariantKey>/
    <BuildingID>/
      <WeatherKey>/
        sample_0000/
          seed_1/
            overall.json
            timeseries.csv
            zone_map.json
            model_export/   (TEASER/AixLib export used in this run)

overall.json includes (examples):
- building_id, formatted_id, n_zones
- heat_demand_kWh, cool_demand_kWh, peak_heat_kW, peak_cool_kW
- wwr_factor, wwr_patched_files
- sa_params (gains_scale, rng_seed, overrides, TH factors)
- lpg_cfg (n_apartments, template mapping, r_values, tp mode)
- task_meta (variant, sample_id, weather_key, YoC class, setpoint stats)
- timeseries_csv meta (columns, missing variables)

──────────────────────────────────────────────────────────────────────────────
HOW TO USE (quick)

1) Configure paths in sa_head.py (TODOs)
   - TEASER_BASE: folder containing Var_1 ... Var_8
   - OUT_BASE: results folder
   - AIXLIB_MO: path to AixLib/package.mo
   - MOS_FILES: paths to TRY_A/TRY_B and optional WeekCold/WeekHot .mos files
   - BUILDING_DATA_PKL must exist (preferred); CSV is fallback but for TEASER build you want PKL payloads

2) Dry run (recommended first)
   - In sa_head.py:
       DRY_RUN = True
   - Run:
       python sa_head.py
   - Output:
       OUT_BASE/dryrun_preview.json
       OUT_BASE/dryrun_task_*.json (optional previews)

3) Full simulation runs
   - In your current sa_head.py, full simulation is intentionally disabled when DRY_RUN=False.
   - Typical setup:
       * create a separate run script (e.g. sa_run.py) that imports build_tasks(...) and calls:
           run_many(tasks, n_proc=N_PROC)
     or
       * remove the RuntimeError and call run_many(...) in the else branch.

──────────────────────────────────────────────────────────────────────────────
NOTES / EXPECTATIONS

- TEASER model is rebuilt for every run in sim_wrapper.py (force_teaser_rebuild=True),
  so YoC changes (sa_tabula_year_class) are reflected in the exported model.
- WWR is applied by scaling AWin and ATransparent in the zone record .mo files.
- Seeds affect LPG profile assignment; uncertainty samples affect YoC/setpoints/weather/WWR/gains.
- timeseries.csv is “zonendynamic”: number of zone columns depends on the variant.

"""
