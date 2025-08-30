import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import pandas as pd

from src.impact_models import LinearOWModel, NonlinearAFSModel
from src.utils import load_data
from src.plotting import (
    plot_exploratory_data_analysis,
    plot_model_comparison,
    plot_detailed_analysis,
    plot_3d_impact_surfaces,
    plot_volume_bucket_analysis,
)


def fit_price_impact_models(data: pd.DataFrame):
    linear_ow_model = LinearOWModel()
    nonlinear_afs_model = NonlinearAFSModel()

    linear_ow_model.fit(data)
    nonlinear_afs_model.fit(data)


if __name__ == "__main__":
    # Resolve project root and data path robustly
    _this_dir = os.path.dirname(os.path.abspath(__file__))
    _project_root = os.path.abspath(os.path.join(_this_dir, ".."))
    _data_path = os.path.join(_project_root, "data", "merged_data.csv")
    data = load_data(_data_path)
    data["ts_event"] = pd.to_datetime(data["ts_event"])

    linear_ow_model = LinearOWModel()
    nonlinear_afs_model = NonlinearAFSModel(delta=0.5)

    linear_ow_model.fit(data)
    nonlinear_afs_model.fit(data)

    print(linear_ow_model)
    print(nonlinear_afs_model)

    # Prepare analysis data
    analysis_data = data.copy()
    analysis_data["price_change"] = analysis_data["mid_price"].diff()
    analysis_data = analysis_data.dropna()

    # Ensure output directory exists
    output_dir = os.path.abspath(os.path.join(_project_root, "report", "figures"))
    os.makedirs(output_dir, exist_ok=True)

    # Generate and save plots
    plot_exploratory_data_analysis(analysis_data, output_dir)
    plot_model_comparison(
        linear_ow_model, nonlinear_afs_model, analysis_data, output_dir
    )
    plot_detailed_analysis(
        linear_ow_model, nonlinear_afs_model, analysis_data, output_dir
    )
    plot_3d_impact_surfaces(
        linear_ow_model, nonlinear_afs_model, analysis_data, output_dir
    )
    plot_volume_bucket_analysis(
        linear_ow_model, nonlinear_afs_model, analysis_data, output_dir
    )
