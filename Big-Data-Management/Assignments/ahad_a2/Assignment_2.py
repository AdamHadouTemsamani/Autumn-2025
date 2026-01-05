
# Imports
import os
import re
import mlflow
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from mlflow.models import infer_signature
from urllib.parse import urlparse
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor 
import mlflow, mlflow.sklearn

# Helper functions 
def read_csv_with_time_index(path):
    """Helper to read CSVs with a datetime index."""
    df = pd.read_csv(path, parse_dates=["time"], index_col="time")
    df.index = pd.to_datetime(df.index)
    df.sort_index(inplace=True)
    return df

def create_eda_plot_dataset(experiment_name, X_test, y_test, y_pred):
    # Prepare directory and file name based on experiment
    output_dir = 'plots'
    os.makedirs(output_dir, exist_ok=True)
    file_name = os.path.join(output_dir, f"{experiment_name}.png")

    # Create 3 plots
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Time series (predictions vs actuals)
    ax_ts = axes[0]
    ax_ts.plot(y_test.index, y_test.values, label='Actual')
    ax_ts.plot(y_test.index, y_pred, label='Predicted', linestyle='--')
    ax_ts.set_title('Actual vs. Predicted (Time Series)')
    ax_ts.set_xlabel('Time')
    ax_ts.set_ylabel('Power')
    ax_ts.legend()

    # Scatter plot (predictions vs actuals)
    ax_scatter = axes[1]
    ax_scatter.scatter(y_test, y_pred, alpha=0.7)
    minval = min(y_test.min(), y_pred.min())
    maxval = max(y_test.max(), y_pred.max())
    ax_scatter.plot([minval, maxval], [minval, maxval], 'r--', linewidth=1)
    ax_scatter.set_title('Predicted vs. Actual')
    ax_scatter.set_xlabel('Actual Power')
    ax_scatter.set_ylabel('Predicted Power')

    # Depending on experiment: histogram vs quiver plot
    ax_third = axes[2]
    if 'SinCos' in experiment_name and 'Sin' in X_test and 'Cos' in X_test:
        # Polar histogram of wind directions
        sin_vals = X_test['Sin'].values
        cos_vals = X_test['Cos'].values
        angles = np.arctan2(sin_vals, cos_vals)
        angles = np.mod(angles, 2*np.pi)  # 0..2π
        ax_third = plt.subplot(1, 3, 3, projection='polar')
        ax_third.hist(angles, bins=16, color='teal', alpha=0.7)
        ax_third.set_title('Wind Direction Frequency')
    elif 'WindVector' in experiment_name and 'u' in X_test and 'v' in X_test:
        u = X_test['u'].values
        v = X_test['v'].values
        colors = y_test.values.squeeze()

        # hexbin showing mean power in each (u,v) bin
        gridsize = 40  # tweak to make bins finer/coarser
        hb = ax_third.hexbin(u, v, C=colors, reduce_C_function=np.mean,
                             gridsize=gridsize, cmap='viridis', mincnt=1)

        cbar = plt.colorbar(hb, ax=ax_third, pad=0.02)
        cbar.set_label('Mean Actual Power')

        # symmetric axis limits and equal aspect so directions are not distorted
        lim = max(np.max(np.abs(u)) if len(u) else 0, np.max(np.abs(v)) if len(v) else 0) * 1.05
        if lim == 0:
            lim = 0.05
        ax_third.set_xlim(-lim, lim)
        ax_third.set_ylim(-lim, lim)
        ax_third.set_aspect('equal')

        ax_third.set_title('Wind density (u,v) — color = mean Power')
        ax_third.set_xlabel('u-component')
        ax_third.set_ylabel('v-component')
    else:
        ax_third.axis('off')
        ax_third.set_title('No data for third plot')
    
    plt.tight_layout()
    fig.savefig(file_name)
    plt.close(fig)
    return file_name

def add_sin_cos(df):
    df = df.copy()
    df['Radians'] = np.deg2rad(df['Degree'])
    df['Sin'] = np.sin(df['Radians'])
    df['Cos'] = np.cos(df['Radians'])
    df = df.drop(columns=['Radians', 'Direction', 'Degree'], errors='ignore')
    return df

def wind_vector(df):
    df = add_sin_cos(df.copy())
    df['u'] = df['Speed'] * df['Sin']
    df['v'] = df['Speed'] * df['Cos']
    df = df.drop(columns=['Sin', 'Cos', 'Radians', 'Direction', 'Degree'], errors='ignore')
    return df

def candidate_models():
    models = {
        'linear': LinearRegression(),
        'rf': RandomForestRegressor(n_estimators=100, max_depth=10),
        'gbr': GradientBoostingRegressor(n_estimators=100, max_depth=10),
        'xgb': XGBRegressor(tree_method='hist', n_estimators=100, max_depth=10),
        'polynomial_lr_deg3_with_ridge': Pipeline([
            ('poly', PolynomialFeatures(degree=3, include_bias=False)),
            ('scaler', StandardScaler()),
            ('regressor', Ridge(alpha=1.0))   # recommended to help regularize polynomial features
        ]),
        'polynomial_lr_deg3_without_ridge': Pipeline([
            ('poly', PolynomialFeatures(degree=3, include_bias=False)),
            ('scaler', StandardScaler()),
            ('regressor', LinearRegression())
        ]),
        'polynomial_lr_deg2_with_ridge': Pipeline([
            ('poly', PolynomialFeatures(degree=2, include_bias=False)),
            ('scaler', StandardScaler()),
            ('regressor', Ridge(alpha=1.0))   # recommended to help regularize polynomial features
        ]),
        'polynomial_lr_deg2_without_ridge': Pipeline([
            ('poly', PolynomialFeatures(degree=2, include_bias=False)),
            ('scaler', StandardScaler()),
            ('regressor', LinearRegression())
        ])
    }
    return models

def train_model(df, dataset_name, models, features, label, all_model_info):

    # Drop rows missing any required column
    needed_cols = features + label
    df_clean = df.dropna(subset=needed_cols).copy()
    
    X = df_clean[features]
    y = df_clean[label].squeeze()

    # Chronological Split for train and test sets: 80% to 20% split, can also use TimeSeriesSplit
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    # Create pipeline
    models =  candidate_models()
    for model_name, model in models.items():  
    # If the candidate model is already a Pipeline, use it as-is.
        if isinstance(model, Pipeline):
            pipeline = model
        else:
            pipeline = Pipeline([
                ("scaler", StandardScaler()), # Transformations
                ("regressor", model) # Estimator
            ])

        # Mlflow 
        with mlflow.start_run(run_name=f"{dataset_name}_{model_name}") as run:
            mlflow.set_tag("feature_type", "SinCos" if "SinCos" in dataset_name else "WindVector")
            run_id = run.info.run_id
            experiment_id = run.info.experiment_id
            mlflow.log_param("dataset", dataset_name)
            mlflow.log_param("model", model_name)

            # Fit model on data
            pipeline.fit(X_train, y_train)
            y_pred = pipeline.predict(X_test)

            test_mae = mean_absolute_error(y_test, y_pred)
            test_mse = mean_squared_error(y_test, y_pred)
            test_rmse = np.sqrt(test_mse)
            test_r2 = r2_score(y_test, y_pred)

            # Plot predictions vs actuals
            file_tag = f"{dataset_name}_{model_name}"
            path = create_eda_plot_dataset(experiment_name=file_tag, X_test=X_test, y_test=y_test, y_pred=y_pred)

            if hasattr(model, "n_estimators"):
                mlflow.log_param("n_estimators", model.n_estimators)
            if hasattr(model, "max_depth"):
                mlflow.log_param("max_depth", model.max_depth)
            if hasattr(model, "tree_method"):
                mlflow.log_param("tree_method", model.tree_method)
            # Log metrics
            mlflow.log_metric("test_mae", test_mae)
            mlflow.log_metric("test_mse", test_mse)
            mlflow.log_metric("test_r2", test_r2)
            mlflow.log_metric("test_rmse", test_rmse)
            
            # Log model 
            signature = infer_signature(X_train, pipeline.predict(X_train))
            mlflow.sklearn.log_model(pipeline, f"{dataset_name}_{model_name}_model", signature=signature)
            
            all_model_info[f"{dataset_name}_{model_name}"] = {
                "run_id": run_id,
                "experiment_id": experiment_id,
                "dataset": dataset_name,
                "model": model_name,
                "test_mae": test_mae,
                "test_mse": test_mse,
                "test_r2": test_r2,
                "test_rmse": test_rmse
            }

            # Log EDA plot as artifact
            if os.path.exists(path):
                mlflow.log_artifact(path, artifact_path="eda_plots")

# --- Save model locally and load for prediction ---

def register_model(model_info, model_name):
    """Save the model to a local path."""
    path = f"runs:/{model_info['run_id']}/{model_info['dataset']}_{model_info['model']}_model"
    mlflow.register_model(path, model_name)

def main():
    # Enable autologging for scikit-learn
    mlflow.sklearn.autolog()

    mlflow.set_tracking_uri("http://127.0.0.1:5000") # We set the MLFlow UI to display in our local host.

    # Load data 
    power_df = read_csv_with_time_index("data/power.csv")
    wind_df = read_csv_with_time_index("data/weather.csv")

    # Filter: Lead_hours, Source_time, AMN, Non-AMN
    cols_to_drop_power = [c for c in ['ANM','Non-ANM'] if c in power_df.columns]
    cols_to_drop_wind = [c for c in ['Lead_hours','Source_time'] if c in wind_df.columns]
    power_df.drop(columns=cols_to_drop_power, inplace=True, errors='ignore')
    wind_df.drop(columns=cols_to_drop_wind, inplace=True, errors='ignore')

    # Naive join 
    power_weather_naive_join = power_df.join(wind_df, how="inner")

    # Upsampling, Downsampling, Merge_asof
    wind_upsample_dfs = wind_df.resample('1min').ffill()
    wind_upsample_join = wind_upsample_dfs.join(power_df, how="inner")

    power_downsample_dfs = power_df.resample('3h').mean()
    power_downsample_join = power_downsample_dfs.join(wind_df, how="inner")

    merged_dfs = pd.merge_asof(power_df, wind_df, on='time', direction='nearest', tolerance=pd.Timedelta('90min'))
    merged_dfs = merged_dfs.set_index('time')

    # Encoding the weather direction
    compass_directions = ['N','NNE','NE','ENE','E','ESE','SE','SSE',
                    'S','SSW','SW','WSW','W','WNW','NW','NNW']
    degree_map = {k: i * 22.5 for i, k in enumerate(compass_directions)}
    for df in [power_weather_naive_join, wind_upsample_join, power_downsample_join, merged_dfs]:

        df['Degree'] = df['Direction'].map(degree_map)

    # Cos-Sin 
    sincos_naive = add_sin_cos(power_weather_naive_join.copy())
    sincos_upsampled = add_sin_cos(wind_upsample_join.copy())
    sincos_downsampled = add_sin_cos(power_downsample_join.copy())
    sincos_merged = add_sin_cos(merged_dfs.copy())

    # Wind Vector 
    wind_vector_naive = wind_vector(power_weather_naive_join.copy())
    wind_vector_upsampled = wind_vector(wind_upsample_join.copy())
    wind_vector_downsampled = wind_vector(power_downsample_join.copy())
    wind_vector_merged = wind_vector(merged_dfs.copy())

    # Datasets
    datasets = {
    "Inner_Join_SinCos": sincos_naive,
    "Upsampled_SinCos": sincos_upsampled,
    "Downsampled_SinCos": sincos_downsampled,
    "Merged_SinCos": sincos_merged,
    "Inner_Join_WindVector": wind_vector_naive,
    "Upsampled_WindVector": wind_vector_upsampled,
    "Downsampled_WindVector": wind_vector_downsampled,
    "Merged_WindVector": wind_vector_merged,
    }   

    # Features and Label
    sincos_X = ['Cos', 'Sin', 'Speed'] # SinCos Features
    windvector_X = ['u', 'v', 'Speed'] # WindVector Features
    y = ['Total'] 


    mlflow.sklearn.autolog()
    mlflow.set_tracking_uri("http://127.0.0.1:5000")

    mlflow.set_experiment("Assignment 2 - WindPower Prediction")

    all_model_runs_sincos = {}
    all_model_runs_windvector = {}

    for name, df in datasets.items():
        if "SinCos" in name:
            train_model(df, name, candidate_models(), sincos_X, y, all_model_info=all_model_runs_sincos)
        if "WindVector" in name:
            train_model(df, name, candidate_models(), windvector_X, y, all_model_info=all_model_runs_windvector)
    
    # Initialize MLflow client and get registered models
    mlflow_client = mlflow.tracking.MlflowClient()
    registered_models = mlflow_client.search_registered_models()

    # Combine runs and find the model with lowest MAE
    combined_runs = {**all_model_runs_sincos, **all_model_runs_windvector}
    best_name, best_info = min(combined_runs.items(), key=lambda x: x[1]['test_mae'])
    best_mae = float(best_info['test_mae'])

    # Register best model if no model exists or if it's better
    if not registered_models or best_mae < registered_models[0].latest_versions[0].metrics['test_mae']:
        register_model(best_info, model_name="Best_Overall_MAE_Model")
        print(f"Registered Best Overall Model based on MAE: {best_name}")
    else:
        print("No new model registered as existing registered model is better or equal based on MAE.")

    # FInd out what the five best models are based on MAE
    combined_model_runs = {**all_model_runs_sincos, **all_model_runs_windvector}
    sorted_models_by_mae = sorted(combined_model_runs.items(), key=lambda x: x[1]['test_mae'])
    print("Top 5 models based on MAE:")
    for model_name, model_info in sorted_models_by_mae[:5]:
        print(f"Model: {model_name}, MAE: {model_info['test_mae']}, MSE: {model_info['test_mse']}, R2: {model_info['test_r2']}, RMSE: {model_info['test_rmse']}")

    # Find out what the five best models are based on RMSE
    sorted_models_by_rmse = sorted(combined_model_runs.items(), key=lambda x: x[1]['test_rmse'])
    print("Top 5 models based on RMSE:")
    for model_name, model_info in sorted_models_by_rmse[:5]:
        print(f"Model: {model_name}, MAE: {model_info['test_mae']}, MSE: {model_info['test_mse']}, R2: {model_info['test_r2']}, RMSE: {model_info['test_rmse']}")

if __name__ == "__main__":
    main()