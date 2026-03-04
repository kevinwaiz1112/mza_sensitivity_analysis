import pathlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from ebcpy import DymolaAPI, TimeSeriesData


def main(
    base_dir=None,
    aixlib_mo=None,
    model_name="AixLib.Fluid.HeatPumps.ModularReversible.Examples.AirToWater2D_OneRoomRadiator",
    with_plot=True,
    n_cpu=1,
):
    """
    Minimal DymolaAPI example:
    - loads AixLib
    - runs one simulation
    - saves result .mat into ./results
    - loads result via TimeSeriesData
    - writes csv + optional quick plot

    Parameters
    ----------
    base_dir : str | Path | None
        Base directory for results. If None -> directory of this script.
    aixlib_mo : str | Path | None
        Path to AixLib/package.mo. If None -> expects submodule at ./external/AixLib/AixLib/package.mo
    model_name : str
        Modelica model to simulate.
    with_plot : bool
        Plot one simple variable if available.
    n_cpu : int
        Number of processes for DymolaAPI.
    """

    script_dir = pathlib.Path(__file__).resolve().parent

    if base_dir is None:
        base_dir = script_dir
    else:
        base_dir = pathlib.Path(base_dir)

    results_dir = base_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    if aixlib_mo is None:
        aixlib_mo = script_dir / "external" / "AixLib" / "AixLib" / "package.mo"
    aixlib_mo = pathlib.Path(aixlib_mo)

    dym_api = DymolaAPI(
        model_name=model_name,
        working_directory=results_dir,   # store Dymola output here
        n_cpu=n_cpu,
        packages=[str(aixlib_mo)],
        show_window=False,
        equidistant_output=False,
        n_restart=-1,
    )

    dym_api.set_sim_setup({
        "start_time": 0,
        "stop_time": 3600,
        "output_interval": 100,
    })

    # Run simulation and store result mat
    result_mat = dym_api.simulate(return_option="savepath")
    print("Result file:", result_mat)

    # Load results
    tsd = TimeSeriesData(result_mat)

    # Save to CSV
    df = pd.DataFrame(tsd)
    csv_path = results_dir / "result.csv"
    df.to_csv(csv_path, index=True)
    print("CSV saved to:", csv_path)

    # Optional simple plot: try a few common candidates
    if with_plot:
        candidates = [
            "heaPum.sigBus.PEleMea",
            "heaPum.PEleMea",
            "PEle",
            "PEleMea",
        ]
        col = None
        for c in candidates:
            if c in df.columns:
                col = c
                break
        if col is not None:
            plt.figure()
            plt.plot(df[col])
            plt.title(col)
            plt.xlabel("time")
            plt.ylabel(col)
            plt.show()
        else:
            print("No known power variable found to plot. Available columns example:", list(df.columns)[:20])

    dym_api.close()


if __name__ == "__main__":
    # If your AixLib is included as submodule in this repo, you can leave aixlib_mo=None
    main(
        base_dir=None,
        aixlib_mo=None,
        with_plot=False,
        n_cpu=1,
    )