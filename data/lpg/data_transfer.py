import json
import shutil
from pathlib import Path
import glob

# -----------------------------
# CONFIG
# -----------------------------
SRC_ROOT = Path(r"C:\03_Repos\SimData\pylpg\HH_AllTemplates_5x\HH_AllTemplates_5x")
DST_ROOT = Path(r"C:\03_Repos\mza_sensitivity_analysis\data\lpg\HH_AllTemplates_5x\HH_AllTemplates_5x")

SELECTED_TEMPLATES = [
    "CHR07", "CHR09", "CHR10", "CHR13", "CHR23", "CHR24", "CHR30", "OR01",
    "CHR01", "CHR02", "CHR16", "CHR17",
    "CHR03", "CHR52",
    "CHR27",
    "CHR41",
]

TP_CODES = ["TP_BER21", "TP_BER23", "TP_BER25", "TP_DEL25", "TP_FR"]
R_VALUES = [1, 2, 3, 4, 5]

# True: overwrite existing dst Results folder for that run
OVERWRITE = False

HH_KEY = "HH1"

LABEL_ELEC = "Electricity"
LABEL_WW = "Warm Water"
LABEL_HW = "Hot Water"
LABEL_CW = "Cold Water"
LABEL_IG = "Inner Device Heat Gains"
LABEL_BAL_OUTSIDE = "BodilyActivityLevel.Outside"


# -----------------------------
# Helpers
# -----------------------------
def find_results_dir(src_r_dir: Path) -> Path | None:
    """
    Finds the folder named 'Results' somewhere under rX.
    This matches your sim layout robustly.
    """
    hits = [Path(p) for p in glob.glob(str(src_r_dir / "**" / "Results"), recursive=True)]
    hits = [p for p in hits if p.is_dir() and p.name.lower() == "results"]
    return hits[0] if hits else None


def pick_one(results_dir: Path, pattern: str) -> Path | None:
    hits = [Path(p) for p in glob.glob(str(results_dir / pattern))]
    return hits[0] if hits else None


def ensure_clean_dir(dst: Path):
    if dst.exists():
        if not OVERWRITE:
            return False
        shutil.rmtree(dst, ignore_errors=True)
    dst.mkdir(parents=True, exist_ok=True)
    return True


def copy_file(src: Path, dst_dir: Path) -> str:
    dst_dir.mkdir(parents=True, exist_ok=True)
    dst = dst_dir / src.name
    shutil.copy2(src, dst)
    return str(dst)


def expected_dst_results_dir(tp: str, template: str, r: int) -> Path:
    # matches utils.py fast path:
    # .../r<r>/Results_<template>_<tpWithoutUnderscores>_r<r>/Results
    tp_compact = tp.replace("_", "")
    return DST_ROOT / tp / template / f"r{r}" / f"Results_{template}_{tp_compact}_r{r}" / "Results"


def main():
    if not SRC_ROOT.is_dir():
        raise FileNotFoundError(f"SRC_ROOT not found: {SRC_ROOT}")

    DST_ROOT.mkdir(parents=True, exist_ok=True)

    manifest = {
        "src_root": str(SRC_ROOT),
        "dst_root": str(DST_ROOT),
        "selected_templates": SELECTED_TEMPLATES,
        "tp_codes": TP_CODES,
        "r_values": R_VALUES,
        "overwrite": OVERWRITE,
        "copied_files": [],
        "skipped_exists": [],
        "missing_runs": [],
        "missing_required_files": [],
        "errors": [],
    }

    for template in SELECTED_TEMPLATES:
        for tp in TP_CODES:
            for r in R_VALUES:
                src_r_dir = SRC_ROOT / tp / template / f"r{r}"
                if not src_r_dir.is_dir():
                    manifest["missing_runs"].append(str(src_r_dir))
                    continue

                src_results = find_results_dir(src_r_dir)
                if src_results is None:
                    manifest["missing_required_files"].append({
                        "run": str(src_r_dir),
                        "missing": ["**/Results (folder not found)"],
                    })
                    continue

                dst_results = expected_dst_results_dir(tp, template, r)

                # If dst exists and not overwriting: skip whole run
                if dst_results.exists() and not OVERWRITE:
                    manifest["skipped_exists"].append(str(dst_results))
                    continue

                try:
                    ensure_clean_dir(dst_results)

                    # required
                    elec = src_results / f"SumProfiles_3600s.{HH_KEY}.{LABEL_ELEC}.csv"
                    ig   = src_results / f"SumProfiles_3600s.{HH_KEY}.{LABEL_IG}.csv"
                    bal  = src_results / f"{LABEL_BAL_OUTSIDE}.{HH_KEY}.json"

                    missing = []
                    if not elec.is_file(): missing.append(str(elec.name))
                    if not ig.is_file():   missing.append(str(ig.name))
                    if not bal.is_file():  missing.append(str(bal.name))

                    # warm water: prefer exact Warm Water; fallback to any Hot Water
                    ww = src_results / f"SumProfiles_3600s.{HH_KEY}.{LABEL_WW}.csv"
                    if not ww.is_file():
                        ww = pick_one(src_results, f"SumProfiles_3600s.{HH_KEY}.*{LABEL_WW}*.csv") or ww

                    hw = src_results / f"SumProfiles_3600s.{HH_KEY}.{LABEL_HW}.csv"
                    if not ww.is_file():
                        # fallback to Hot Water (exact or fuzzy)
                        if hw.is_file():
                            ww = hw
                        else:
                            hw2 = pick_one(src_results, f"SumProfiles_3600s.{HH_KEY}.*{LABEL_HW}*.csv")
                            if hw2 is not None:
                                ww = hw2

                    if not ww.is_file():
                        missing.append(f"SumProfiles_3600s.{HH_KEY}.({LABEL_WW} or {LABEL_HW}).csv")

                    if missing:
                        manifest["missing_required_files"].append({
                            "run": str(src_r_dir),
                            "results_dir": str(src_results),
                            "missing": missing,
                        })
                        # nichts kopieren, damit dst nicht halbfertig bleibt
                        shutil.rmtree(dst_results.parent.parent, ignore_errors=True)  # removes rX/*
                        continue

                    # optional cold water
                    cw = src_results / f"SumProfiles_3600s.{HH_KEY}.{LABEL_CW}.csv"
                    if not cw.is_file():
                        cw = pick_one(src_results, f"SumProfiles_3600s.{HH_KEY}.*{LABEL_CW}*.csv") or cw

                    # copy required + optional
                    manifest["copied_files"].append(copy_file(elec, dst_results))
                    manifest["copied_files"].append(copy_file(ww, dst_results))
                    manifest["copied_files"].append(copy_file(ig, dst_results))
                    manifest["copied_files"].append(copy_file(bal, dst_results))

                    if cw.is_file():
                        manifest["copied_files"].append(copy_file(cw, dst_results))

                except Exception as e:
                    manifest["errors"].append({
                        "run": str(src_r_dir),
                        "err": str(e),
                    })

    out_manifest = DST_ROOT.parent / "copy_manifest_minimal.json"
    with open(out_manifest, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    print("Done.")
    print("Manifest:", out_manifest)
    print("Copied files:", len(manifest["copied_files"]))
    print("Skipped (exists):", len(manifest["skipped_exists"]))
    print("Missing runs:", len(manifest["missing_runs"]))
    print("Missing required files:", len(manifest["missing_required_files"]))
    print("Errors:", len(manifest["errors"]))


if __name__ == "__main__":
    main()