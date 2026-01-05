########################################################################################################################
# IMPORTS
# You absolutely need these
# from influxdb import InfluxDBClient   # REMOVED — no longer needed
import mlflow
import os

# You will probably need these
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline

# This are for example purposes. You may discard them if you don't use them.
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split, TimeSeriesSplit, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, root_mean_squared_error
from mlflow.models import infer_signature
from urllib.parse import urlparse
from sklearn.model_selection import GridSearchCV

########################################################################################################################

# Configurations

POWER_CSV = "data/power.csv"
WEATHER_CSV = "data/weather.csv"
FUTURE_CSV = "data/future.csv"
ML_FLOW_TRACKING_URI = "http://127.0.0.1:5000"
EXPERIMENT_SIN_COS = "SinCos_Experiment"
EXPERIMENT_WIND_VECTOR = "WindVector_Experiment"
RANDOM_SEED = 42
CV_SPLITS = 5
TEST_SIZE = 0.2


########################################################################################################################

## Step 1: The Data (from CSVs)

def read_csv_with_time_index(path):
    """Helper to read CSVs with a datetime index."""
    df = pd.read_csv(path, parse_dates=["time"], index_col="time")
    df.index = pd.to_datetime(df.index)
    df.sort_index(inplace=True)
    return df

def eda_sincos(df, save_path):
    fig, ax = plt.subplots(1,2, figsize=(25,4))

    # Speed vs Total (Power Curve nature)
    ax[0].scatter(df["Speed"], df["Total"])
    power_curve = df.groupby("Speed").median(numeric_only=True)["Total"]
    ax[0].plot(power_curve.index, power_curve.values, "k:", label=" Power Curve")
    ax[0].legend()
    ax[0].set_title("Windspeed vs Power")
    ax[0].set_ylabel("Power [MW]")
    ax[0].set_xlabel("Windspeed [m/s]")

    
    # Cos and Sin heatmap distribution
    h = ax[1].hist2d(
        df["Dir_x"],
        df["Dir_y"],
        bins=80,              
        cmap="viridis"         
    )

    ax[1].set_title("Direction Sin-Cos Heatmap")
    ax[1].set_xlabel("Dir_x (Cosine)")
    ax[1].set_ylabel("Dir_y (Sine)")
    ax[1].set_aspect("equal")

    # Add colorbar
    plt.colorbar(h[3], ax=ax[1])

    plt.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)

    
    return save_path

def eda_windvector(df, save_path):
    fig, ax = plt.subplots(1,2, figsize=(15,4))
    
    # Wind Vector distribution
    ax[0].scatter(df["Wind_x"], df["Wind_y"], s=5)
    ax[0].set_title("Wind Vector Distribution")
    ax[0].set_xlabel("Wind_x")
    ax[0].set_ylabel("Wind_y")
    ax[0].set_aspect("equal")

    plt.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)
    return save_path

########################################################################################################################
## Preproccessing - Merging datasets

## Preprocessing - Feature Encoding
direction_to_deg = { # compass directions to degrees
    'N': 0, 'NNE': 22.5, 'NE': 45, 'ENE': 67.5,
    'E': 90, 'ESE': 112.5, 'SE': 135, 'SSE': 157.5,
    'S': 180, 'SSW': 202.5, 'SW': 225, 'WSW': 247.5,
    'W': 270, 'WNW': 292.5, 'NW': 315, 'NNW': 337.5
}

def preprocess_sin_cos_encoding(df):
   """
   Convert Direction to Sin and Cos to keep circular relationship
   """
   df = df.copy()
   df['deg'] = df['Direction'].map(direction_to_deg) # map text to compass degrees
   df['rad'] = np.deg2rad(df['deg'])
   df['Dir_x'] = np.cos(df['rad'])  # Cosine component
   df['Dir_y'] = np.sin(df['rad'])  # Sine component
   df.drop(columns=['Direction', 'deg', 'rad'], inplace=True)
   return df

def preprocess_wind_vector_encoding(df):
   """
   Combine Speed + Direction to wind vector (Wind_x, Wind_y)
   """
   df = df.copy()
   df['deg'] = df['Direction'].map(direction_to_deg)
   df['rad'] = np.deg2rad(df['deg'])
   df['Wind_x'] = df['Speed'] * np.cos(df['rad'])
   df['Wind_y'] = df['Speed'] * np.sin(df['rad'])
   df = df.drop(columns=['Direction', 'Speed', 'deg', 'rad'])
   return df

########################################################################################################################
# Pipeline 

def build_pipeline(model):
   if isinstance(model, Pipeline):
      return model                      # Polynomial pipeline already includes scaler
   else:
      return Pipeline([
         ("Scaler", StandardScaler()),  # Transformations
         ("regressor", model)           # Estimator
      ])

def get_models():
    return {
        "LinearRegression": LinearRegression(),
        "Polynomial_2deg": Pipeline([
            ("poly", PolynomialFeatures(degree=2, include_bias=False)),
            ("scaler", StandardScaler()),
            ("regressor", LinearRegression())
        ]),
        "Polynomial_3deg": Pipeline([
            ("poly", PolynomialFeatures(degree=3, include_bias=False)),
            ("scaler", StandardScaler()),
            ("regressor", LinearRegression())
        ]),
        "RandomForest": RandomForestRegressor(n_estimators=100, max_depth=5),
        "GradientBoosting": GradientBoostingRegressor(n_estimators=100, max_depth=5)
   }


# def get_future_forecasts():
#     """Load future forecasts from CSV (mimicking Influx future query)."""
#     try:
#         forecasts = read_csv_with_time_index(FUTURE_CSV)
#     except FileNotFoundError:
#         print("No future.csv found, skipping forecast step.")
#         return None
    
#     # mimic selection of most recent source time
#     if "Source_time" in forecasts.columns:
#         newest_forecasts = forecasts.loc[forecasts["Source_time"] == forecasts["Source_time"].max()].copy()
#         return newest_forecasts
#     return forecasts



#########################################################################################################################
# Model training
def train_and_log(df, dataset_name, feature_cols, models_dict, base_experiment_name, all_runs):
   
   X = df [feature_cols]
   y = df ['Total']

   # Chronological Split for train and test sets
   X_train, X_test, y_train, y_test = train_test_split(
      X, 
      y, 
      test_size=0.2, # 80% for training, 20% for testing
      shuffle=False # Important for time series data
   )

   for model_name, model in models_dict.items():
        pipeline = build_pipeline(model)
        
        experiment_name = f"{base_experiment_name}_{model_name}"
        mlflow.set_experiment(experiment_name)

        # Fit model on training data
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        test_mae = mean_absolute_error(y_test, y_pred)
        test_mse = np.sqrt(mean_squared_error(y_test, y_pred))
        test_r2 = r2_score(y_test, y_pred)
        test_rmse = root_mean_squared_error(y_test, y_pred)

        if base_experiment_name == EXPERIMENT_SIN_COS:
            plot_path = eda_sincos(df, save_path=f"plots/eda_sincos_{dataset_name}_{model_name}.png")
        else:
            plot_path = eda_windvector(df, save_path=f"plots/eda_windvector_{dataset_name}_{model_name}.png")
    

        # Log run for MLflow
        with mlflow.start_run(run_name=f"{dataset_name}_{model_name}") as run:
            run_id = run.info.run_id
            mlflow.log_param("dataset", dataset_name)
            mlflow.log_param("model", model_name)
            mlflow.log_param("features", feature_cols)
            if base_experiment_name == EXPERIMENT_SIN_COS:
                mlflow.log_param("encoding", "Sin-Cos")
            elif base_experiment_name == EXPERIMENT_WIND_VECTOR:
                mlflow.log_param("encoding", "Wind Vector")

            mlflow.log_metric("MAE", test_mae)
            mlflow.log_metric("MSE", test_mse)
            mlflow.log_metric("R2", test_r2)
            mlflow.log_metric("RMSE", test_rmse)

            # Log model artifact and signature
            try: 
                signature = infer_signature(X_train, pipeline.predict(X_train))
                mlflow.sklearn.log_model(
                    pipeline, 
                    artifact_path=f"{dataset_name}_{model_name}_model", 
                    signature=signature
                )
            except Exception:
                mlflow.sklearn.log_model(
                    pipeline, 
                    artifact_path=f"{dataset_name}_{model_name}_pipeline"
                )
            
            # Log EDA plot
            mlflow.log_artifact(plot_path)

            all_runs[f"{dataset_name}_{model_name}"] = {
                "run_id": run_id,
                "dataset": dataset_name,
                "model": model_name,
                "MAE": test_mae,
                "pipeline": pipeline
            }
        

########################################################################################################################

# Save best model locally

def register_best_model(best_model_info, registered_model_name):
    """Register the best model in MLflow Model Registry"""
    model_uri = f"runs:/{best_model_info['run_id']}/{best_model_info['dataset']}_{best_model_info['model']}_model"
    mlflow.register_model(model_uri, registered_model_name)

def save_pipeline_locally(pipeline, path="model"):
    """Save the trained pipeline locally"""
    if os.path.exists(path):
        import shutil
        shutil.rmtree(path)  # Remove existing directory
    mlflow.sklearn.save_model(pipeline, path)
    print(f"Model saved locally at {path}")

def load_and_predict_model(model, new_data):
    """Load saved model and make predictions on new data"""
    model = mlflow.sklearn.load_model(model)
    return model.predict(new_data)        

########################################################################################################################
# Main execution
def main():
    mlflow.sklearn.autolog()
    mlflow.set_tracking_uri(ML_FLOW_TRACKING_URI)

    # Load CSVs
    power_df = read_csv_with_time_index(POWER_CSV)
    wind_df = read_csv_with_time_index(WEATHER_CSV)

    # Filter dataframes to only have relevant columns
    power_df = power_df[["Total"]]
    wind_df = wind_df[["Speed", "Direction"]]

    # Merge datasets

    joined_dfs = power_df.join(wind_df, how="inner")

    weather_upsampled = wind_df.resample('1min').ffill()  # forward fill to propagate last known value
    upsampled_merged_dfs = power_df.join(weather_upsampled, how="inner")

    merged_dfs = pd.merge_asof(
    power_df.sort_index(),
    wind_df.sort_index(),
    left_index=True,
    right_index=True,
    direction='nearest', # choose the nearest timestamp (past or future timestamp)
    tolerance=pd.Timedelta('90min')  # only match within +/- 90 minutes  
    )

    merged_dfs = merged_dfs.dropna(subset=['Direction', 'Speed']) # Remove NaN values that could not be matched within tolerance

    # Prepare datasets for different encodings
    sin_cos_joined = preprocess_sin_cos_encoding(joined_dfs)
    sin_cos_upsampled = preprocess_sin_cos_encoding(upsampled_merged_dfs)
    sin_cos_merged = preprocess_sin_cos_encoding(merged_dfs)

    wind_vector_joined = preprocess_wind_vector_encoding(joined_dfs)
    wind_vector_upsampled = preprocess_wind_vector_encoding(upsampled_merged_dfs)
    wind_vector_merged = preprocess_wind_vector_encoding(merged_dfs)

    # Datasets dictionary
    sincos_datasets = {
        "SinCos_Joined": sin_cos_joined,
        "SinCos_Upsampled": sin_cos_upsampled,
        "SinCos_Merged": sin_cos_merged
    }

    wind_vector_datasets = {
        "WindVector_Joined": wind_vector_joined,
        "WindVector_Upsampled": wind_vector_upsampled,
        "WindVector_Merged": wind_vector_merged
    }

    all_sincos_runs = {}
    all_windvector_runs = {}

    # Run Experiment A: Sin-Cos Encoding
    print("Starting Sin-Cos Encoding Experiment...")
    models_sincos = get_models()
    for name, df in sincos_datasets.items():
        feature_cols = ['Speed', 'Dir_x', 'Dir_y']
        train_and_log(
            df, 
            name,   
            feature_cols, 
            models_sincos, 
            base_experiment_name=EXPERIMENT_SIN_COS,
            all_runs=all_sincos_runs
        )

    # Run Experiment B: Wind Vector Encoding
    print("Starting Wind Vector Encoding Experiment...")
    models_windvector = get_models()
    for name, df in wind_vector_datasets.items():
        feature_cols = ['Wind_x', 'Wind_y']
        train_and_log(
            df, 
            name, 
            feature_cols, 
            models_windvector, 
            base_experiment_name=EXPERIMENT_WIND_VECTOR,
            all_runs=all_windvector_runs
        )

    # Pick best model across both experiments
    if len(all_sincos_runs) == 0:
        print("No SinCos runs were recorded.")
    else:
        best_sincos_run_key = min(all_sincos_runs, key=lambda k: all_sincos_runs[k]['MAE'])
        best_sincos_run = all_sincos_runs[best_sincos_run_key]
        register_best_model(best_sincos_run, "Best_SinCos_Model")
        print(f"Best SinCos model: {best_sincos_run_key} on dataset {best_sincos_run['dataset']} with MAE: {best_sincos_run['MAE']}")

    if len(all_windvector_runs) == 0:
        print("No WindVector runs were recorded.")
    else:
        best_windvector_run_key = min(all_windvector_runs, key=lambda k: all_windvector_runs[k]['MAE'])
        best_windvector_run = all_windvector_runs[best_windvector_run_key]
        register_best_model(best_windvector_run, "Best_WindPower_Model")
        print(f"Best WindVector model: {best_windvector_run_key} on dataset {best_windvector_run['dataset']} with MAE: {best_windvector_run['MAE']}")

    print("Done")
    # # Simulate future forecasts and make predictions with the best model
    # future_forecasts = get_future_forecasts()
    # if future_forecasts is not None:
    #     print("Load future forecasts data")

    #     # Load best model from MLflow Model Registry
    #     best_model_sincos = mlflow.pyfunc.load_model("models:/Best_SinCos_Model/Production") # Change name
    #     best_model_windvector = mlflow.pyfunc.load_model("models:/Best_WindPower_Model/Production")

    #     # Copy future forecasts to avoid modifying original
    #     future_forecasts_sincos = future_forecasts.copy()
    #     future_forecasts_windvector = future_forecasts.copy()
    #     # Preprocess future forecasts for both encodings
    #     future_forecasts_sincos = preprocess_sin_cos_encoding(future_forecasts_sincos)
    #     future_forecasts_windvector = preprocess_wind_vector_encoding(future_forecasts_windvector)

    #     preds_sincos = best_model_sincos.predict(future_forecasts_sincos)
    #     preds_windvector = best_model_windvector.predict(future_forecasts_windvector)

    #     future_forecasts_sincos['Predicted_Total_SinCos'] = preds_sincos
    #     future_forecasts_windvector['Predicted_Total_WindVector'] = preds_windvector

    #     print("Predictions with Sin-Cos Encoding Model:")
    #     print(future_forecasts_sincos[['Predicted_Total_SinCos']])

    #     print("Predictions with Wind Vector Encoding Model:")
    #     print(future_forecasts_windvector[['Predicted_Total_WindVector']])

if __name__ == "__main__":
    main()


# # Enable autologging for scikit-learn
# mlflow.sklearn.autolog()

# mlflow.set_tracking_uri("http://127.0.0.1:5000") # We set the MLFlow UI to display in our local host.

# # Start a run
# with mlflow.start_run(run_name="LinearRegression"):


#     # --- Load from CSVs (instead of InfluxDB) ---
#     power_df = read_csv_with_time_index("data/power.csv")
#     wind_df = read_csv_with_time_index("data/weather.csv")

#     # --- Join datasets (as before) ---
#     joined_dfs = power_df.join(wind_df, how="inner").dropna(subset=["Total", "Speed"])

#     # --- Create and save EDA plots ---
#     eda_fig = create_eda_plots(joined_dfs)
#     eda_fig.savefig("eda_plots.png")
#     mlflow.log_artifact("eda_plots.png")
#     plt.close(eda_fig)

#     # --- Model section (same as before) ---
#     def load_and_predict_model(model_name, model_version, new_data):
#         model = mlflow.pyfunc.load_model(model_uri=f"models:/{model_name}/{model_version}")
#         return model.predict(new_data)

#     # --- Simulate "future" data ---
#     def get_future_forecasts():
#         """Load future forecasts from CSV (mimicking Influx future query)."""
#         try:
#             forecasts = read_csv_with_time_index("data/future.csv")
#         except FileNotFoundError:
#             print("No future.csv found, skipping forecast step.")
#             return None
#         # mimic selection of most recent source time
#         if "Source_time" in forecasts.columns:
#             newest_forecasts = forecasts.loc[forecasts["Source_time"] == forecasts["Source_time"].max()].copy()
#             return newest_forecasts
#         return forecasts

#     X = joined_dfs[["Speed"]]
#     y = joined_dfs["Total"]

#     number_of_splits = 5
#     tscv = TimeSeriesSplit(number_of_splits)

#     pipeline = Pipeline([
#         ('scaler', StandardScaler()),
#         ('regressor', LinearRegression())
#     ])

#     # Train and evaluate model using cross-validation
#     for i, (train, test) in enumerate(tscv.split(X, y)):
#         pipeline.fit(X.iloc[train], y.iloc[train])
#         predictions = pipeline.predict(X.iloc[test])
#         truth = y.iloc[test]

#         # Plot predictions
#         plt.figure()
#         plt.plot(truth.index, truth.values, label="Truth")
#         plt.plot(truth.index, predictions, label="Predictions")
#         plt.legend()
#         plt.savefig(f"predictions_{i}.png")
#         plt.close()
#         mlflow.log_artifact(f"predictions_{i}.png")
    
#     # Safe model
#     mlflow.sklearn.save_model(pipeline, "model")

#     # No need to manually log metrics - autologging handles:
#     # - Parameters
#     # - Metrics (R², MSE, MAE)
#     # - Model artifacts
#     # - Model signature
#     # - Feature importance (for supported models)


    

########################################################################################################################
