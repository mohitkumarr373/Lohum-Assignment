
import numpy as np
import pandas as pd
from typing import Tuple
from data_loader import load_oem_data

def compute_rated_values(df: pd.DataFrame) -> Tuple[float, float]:
    rated_capacity_mAh = float(df["discharge_capacity_mAh"].max())
    rated_power_W = float(df["power_W"].max())
    return rated_capacity_mAh, rated_power_W


def identify_main_discharge_step(cell_df: pd.DataFrame) -> int:
    step_cap = cell_df.groupby("step_number")["discharge_capacity_mAh"].max()
    return int(step_cap.idxmax())


def calculate_targets(
    df_oem: pd.DataFrame,
    rated_capacity_mAh: float | None = None,
    rated_power_W: float | None = None,
) -> pd.DataFrame:
    if rated_capacity_mAh is None or rated_power_W is None:
        rated_capacity_mAh, rated_power_W = compute_rated_values(df_oem)

    records = []
    for cell_id, group in df_oem.groupby("cell_number"):
        main_step = identify_main_discharge_step(group)
        step_df = group[group["step_number"] == main_step]

        max_capacity_mAh = float(step_df["discharge_capacity_mAh"].max())
        max_power_W = float(step_df["power_W"].max())

        soh = max_capacity_mAh / rated_capacity_mAh if rated_capacity_mAh > 0 else np.nan
        sop = max_power_W / rated_power_W if rated_power_W > 0 else np.nan

        records.append(
            {
                "cell_number": cell_id,
                "main_step": main_step,
                "max_capacity_mAh": max_capacity_mAh,
                "max_power_W": max_power_W,
                "SOH": soh,
                "SOP": sop,
            }
        )

    return pd.DataFrame(records)


def extract_predictive_features(df_oem: pd.DataFrame, frac: float = 0.1) -> pd.DataFrame:
    records = []

    for cell_id, group in df_oem.groupby("cell_number"):
        main_step = identify_main_discharge_step(group)
        step_df = group[group["step_number"] == main_step].sort_values("time_min")

        n = len(step_df)
        if n == 0:
            continue

        k = max(1, int(frac * n))
        early = step_df.iloc[:k]

        mean_voltage = float(early["voltage_V"].mean())
        std_voltage = float(early["voltage_V"].std(ddof=0))
        max_current = float(early["current_A"].max())
        mean_power = float(early["power_W"].mean())

        q = early["discharge_capacity_mAh"].to_numpy()
        v = early["voltage_V"].to_numpy()
        if len(q) >= 2:
            dq = np.diff(q)
            dv = np.diff(v)
            with np.errstate(divide="ignore", invalid="ignore"):
                dv_dq = np.where(dq != 0, dv / dq, 0.0)
            mean_abs_dv_dq = float(np.mean(np.abs(dv_dq)))
        else:
            mean_abs_dv_dq = 0.0

        records.append(
            {
                "cell_number": cell_id,
                "mean_voltage_early": mean_voltage,
                "std_voltage_early": std_voltage,
                "max_current_early_A": max_current,
                "mean_power_early_W": mean_power,
                "mean_abs_dv_dq_early": mean_abs_dv_dq,
            }
        )

    return pd.DataFrame(records)
