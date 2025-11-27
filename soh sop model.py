
import os
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import KFold
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    mean_absolute_percentage_error,
)
from sklearn.ensemble import RandomForestRegressor

from data_loader import load_oem_data
from feature_engineering import calculate_targets, extract_predictive_features, compute_rated_values


def evaluate_model(X: pd.DataFrame, y: pd.Series, target_name: str) -> dict:
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    model = RandomForestRegressor(
        n_estimators=300,
        max_depth=None,
        random_state=42,
        n_jobs=-1,
    )

    rmse_list, mae_list, mape_list = [], [], []

    for train_idx, test_idx in kf.split(X):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        rmse_list.append(np.sqrt(mean_squared_error(y_test, y_pred)))
        mae_list.append(mean_absolute_error(y_test, y_pred))
        mape_list.append(mean_absolute_percentage_error(y_test, y_pred))

    rmse = float(np.mean(rmse_list))
    mae = float(np.mean(mae_list))
    mape = float(np.mean(mape_list))

    print(f"{target_name} - RMSE: {rmse:.4f}, MAE: {mae:.4f}, MAPE: {mape:.2%}")

    final_model = RandomForestRegressor(
        n_estimators=300,
        max_depth=None,
        random_state=42,
        n_jobs=-1,
    )
    final_model.fit(X, y)

    os.makedirs("models", exist_ok=True)
    joblib.dump(final_model, f"models/{target_name}_model.pkl")

    return {"RMSE": rmse, "MAE": mae, "MAPE": mape}


def main(oem_id: int = 2, base_path: str = "data") -> None:
    df_oem = load_oem_data(oem_id, base_path=base_path)
    rated_capacity_mAh, rated_power_W = compute_rated_values(df_oem)

    targets_df = calculate_targets(df_oem, rated_capacity_mAh, rated_power_W)
    features_df = extract_predictive_features(df_oem)

    final_df = pd.merge(features_df, targets_df, on="cell_number", how="inner").dropna()

    feature_cols = [
        col
        for col in final_df.columns
        if col
        not in [
            "cell_number",
            "main_step",
            "max_capacity_mAh",
            "max_power_W",
            "SOH",
            "SOP",
        ]
    ]

    X = final_df[feature_cols]
    y_soh = final_df["SOH"]
    y_sop = final_df["SOP"]

    soh_metrics = evaluate_model(X, y_soh, "SOH")
    sop_metrics = evaluate_model(X, y_sop, "SOP")

    metrics_df = pd.DataFrame(
        {
            "Metric": ["RMSE", "MAE", "MAPE"],
            "SOH": [
                soh_metrics["RMSE"],
                soh_metrics["MAE"],
                soh_metrics["MAPE"] * 100,
            ],
            "SOP": [
                sop_metrics["RMSE"],
                sop_metrics["MAE"],
                sop_metrics["MAPE"] * 100,
            ],
        }
    )

    metrics_df.to_csv("metrics.csv", index=False)
    print("Saved metrics to metrics.csv")


if __name__ == "__main__":
    main()
