
import pandas as pd
from pathlib import Path
from typing import List

COL_NAMES = [
    "id1",
    "cell_number",
    "step_number",
    "id2",
    "row_count",
    "timestamp",
    "time_min",
    "voltage_mV",
    "current_mA",
    "discharge_capacity_mAh",
    "feature_10",
    "power_W",
]


def _load_txt_files(file_paths: List[Path], oem_id: int) -> pd.DataFrame:
    frames = []
    for path in file_paths:
        df = pd.read_csv(
            path,
            sep=r"\s+",
            header=None,
            names=COL_NAMES,
            engine="python",
        )
        df["oem_id"] = oem_id
        df["source_file"] = path.name

        df["time_min"] = pd.to_numeric(df["time_min"], errors="coerce")
        for col in ["voltage_mV", "current_mA", "discharge_capacity_mAh", "feature_10", "power_W"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        df["voltage_V"] = df["voltage_mV"] / 1000.0
        df["current_A"] = df["current_mA"] / 1000.0
        df["capacity_Ah"] = df["discharge_capacity_mAh"] / 1000.0

        df["feature_10_mWh"] = df["feature_10"]
        frames.append(df)

    if not frames:
        raise FileNotFoundError("No files loaded for OEM {oem_id}".format(oem_id=oem_id))

    return pd.concat(frames, ignore_index=True)


def load_oem_data(oem_id: int, base_path: str = "data") -> pd.DataFrame:
    base = Path(base_path)

    if oem_id == 1:
        pattern = "OEM1/*.txt"
    elif oem_id == 2:
        pattern = "OEM2/*.txt"
    elif oem_id == 3:
        pattern = "OEM3/*.txt"
    else:
        raise ValueError("oem_id must be 1, 2, or 3")

    files = sorted(base.glob(pattern))
    if not files:
        raise FileNotFoundError(
            f"No .txt files found for OEM {oem_id} under {base}/{pattern}. "
            "Place the 10 text files for each OEM inside data/OEM1, data/OEM2, data/OEM3."
        )

    return _load_txt_files(files, oem_id)


def load_all_oems(base_path: str = "data") -> pd.DataFrame:
    frames = []
    for oem_id in (1, 2, 3):
        try:
            frames.append(load_oem_data(oem_id, base_path=base_path))
        except FileNotFoundError:
            continue

    if not frames:
        raise FileNotFoundError("No OEM data found in the provided base_path.")

    return pd.concat(frames, ignore_index=True)
