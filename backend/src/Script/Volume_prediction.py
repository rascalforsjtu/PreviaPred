import os
os.environ['SCIPY_ARRAY_API'] = '1'
import matplotlib
matplotlib.use('Agg')

import warnings
import joblib
import numpy as np
import pandas as pd
import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import xgboost as xgb
from scipy.stats import shapiro

warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")
warnings.filterwarnings("ignore", category=RuntimeWarning)

from sklearn.base import clone
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LassoCV
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from tabpfn import TabPFNRegressor

FEATURE_FILE_RAD = "backend/src/data/Data_post_extracted/Radiomics_Features.csv"
FEATURE_FILE_CLI = "backend/src/data/Data_post_extracted/Clinical_Feature.csv"
LABEL_FILE = "backend/src/data/prospective cases regression/True_Volume/hemorrhage_volume_log_transformed.csv"                            
MODEL_DIR = "backend/src/data/Processing"
OUTPUT_DIR = "backend/src/data/prospective cases regression"
FINAL_TEST_SIZE = 0.15
RANDOM_STATE = 439
CLASSIFICATION_MODEL_PATH = "backend/src/data/Processing/final_attention_fusion_model.joblib"  
INDEPENDENT_TEST_FILE = "backend/src/data/Processing/dataset_split.txt"

PREDICT_CLI_FILE = "backend/src/data/prospective cases regression/Clinical/clinical_data_7_cases.csv"  
PREDICT_RAD_FILE = "backend/src/data/prospective cases regression/Radiomics/Placenta_radiomics.csv/Placenta_Features.csv"
PREDICT_LABEL_FILE = "backend/src/data/prospective cases regression/True_Volume/label_6_cases_blood.csv"

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "predictions"), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "data_distribution"), exist_ok=True)

class AttentionFusionDataset(Dataset):
    def __init__(self, X_cli, X_rad):
        self.X_cli = torch.tensor(X_cli, dtype=torch.float32)
        self.X_rad = torch.tensor(X_rad, dtype=torch.float32)
    def __len__(self): return len(self.X_cli)
    def __getitem__(self, idx): return self.X_cli[idx], self.X_rad[idx]

class AttentionFusionModel(nn.Module):
    def __init__(self, num_cli_features, num_rad_features, hidden_dim=32, dropout_rate=0.3):
        super().__init__()
        self.cli_branch = nn.Sequential(
            nn.Linear(num_cli_features, hidden_dim*2),
            nn.BatchNorm1d(hidden_dim*2), nn.ReLU(), nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim*2, hidden_dim),
            nn.BatchNorm1d(hidden_dim), nn.ReLU(), nn.Dropout(dropout_rate/2),
        )
        self.rad_branch = nn.Sequential(
            nn.Linear(num_rad_features, hidden_dim*2),
            nn.BatchNorm1d(hidden_dim*2), nn.ReLU(), nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim*2, hidden_dim),
            nn.BatchNorm1d(hidden_dim), nn.ReLU(), nn.Dropout(dropout_rate),
        )
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim*2, hidden_dim), 
            nn.ReLU(),
            nn.Linear(hidden_dim, 2), 
            nn.Softmax(dim=1)
        )
        self.final = nn.Sequential(
            nn.Linear(hidden_dim*2, hidden_dim), 
            nn.BatchNorm1d(hidden_dim), 
            nn.ReLU(), 
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, 1), 
            nn.Sigmoid()
        )
        
    def forward(self, x_cli, x_rad):
        cli_out = self.cli_branch(x_cli)
        rad_out = self.rad_branch(x_rad)
        concat = torch.cat([cli_out, rad_out], dim=1)
        att = self.attention(concat)
        cli_att = cli_out * att[:,0:1]
        rad_att = rad_out * att[:,1:2]
        fused = torch.cat([cli_att, rad_att], dim=1)
        return self.final(fused), att

class AttentionFusionClassifier:
    def __init__(self, num_cli_features, num_rad_features, hidden_dim=32,
                 dropout_rate=0.3, learning_rate=1e-3, batch_size=16,
                 device=None):
        self.num_cli_features = num_cli_features
        self.num_rad_features = num_rad_features
        self.hidden_dim = hidden_dim
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.scaler_cli = StandardScaler()
        self.scaler_rad = StandardScaler()
        self.model = None
        
    def load_model(self, model_path):
        self.model = AttentionFusionModel(
            self.num_cli_features,
            self.num_rad_features,
            self.hidden_dim,
            self.dropout_rate
        ).to(self.device)
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()
    
    def predict_proba(self, X_cli, X_rad):
        Xc = self.scaler_cli.transform(X_cli)
        Xr = self.scaler_rad.transform(X_rad)
        ds = AttentionFusionDataset(Xc, Xr)
        loader = DataLoader(ds, batch_size=self.batch_size)
        probs = []
        with torch.no_grad():
            for xc, xr in loader:
                xc, xr = xc.to(self.device), xr.to(self.device)
                out, _ = self.model(xc, xr)
                probs.append(out.cpu().numpy())
        return np.vstack(probs).ravel()

MODELS = {
    "TabPFN": TabPFNRegressor(
        device="cuda" if torch.cuda.is_available() else "cpu",
        model_path='backend/src/data/Data_post_extracted/tabpfn-v2-regressor-2noar4o2.ckpt'
    ),
    "XGBoost": Pipeline([
        ('scaler', StandardScaler()),
        ('model', xgb.XGBRegressor(
            objective='reg:squarederror',
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=RANDOM_STATE,
            n_jobs=-1
        ))
    ]),
    "Lasso": Pipeline([
        ('scaler', StandardScaler()),
        ('model', LassoCV(cv=5, random_state=RANDOM_STATE, max_iter=1000000))
    ]),
    "RandomForest": Pipeline([
        ('model', RandomForestRegressor(
            n_estimators=100, 
            random_state=RANDOM_STATE, 
            n_jobs=-1,
            max_depth=8,
            min_samples_split=5))
    ]),
    "SVM": Pipeline([
        ('scaler', StandardScaler()),
        ('model', SVR(kernel='rbf', C=1.0, gamma='scale'))
    ])
}

def compute_regression_metrics(y_true_log, y_pred_log):
    metrics_log = {
        'mse_log': mean_squared_error(y_true_log, y_pred_log),
        'rmse_log': np.sqrt(mean_squared_error(y_true_log, y_pred_log)),
        'mae_log': mean_absolute_error(y_true_log, y_pred_log),
        'r2_log': r2_score(y_true_log, y_pred_log),
    }
    
    if np.any(y_true_log < 0) or np.any(y_pred_log < 0):
        warnings.warn("Negative values present in log labels/predictions, cannot restore original volume, skipping original scale metric calculation")
        return metrics_log
    
    y_true_original = np.exp(y_true_log)
    y_pred_original = np.exp(y_pred_log)
    metrics_original = {
        'mse_original': mean_squared_error(y_true_original, y_pred_original),
        'rmse_original': np.sqrt(mean_squared_error(y_true_original, y_pred_original)),
        'mae_original': mean_absolute_error(y_true_original, y_pred_original),
        'r2_original': r2_score(y_true_original, y_pred_original),
    }
    
    return {**metrics_log, **metrics_original}

def save_regression_metrics_to_csv(results, output_dir):
    metrics_data = []
    models_order = ["TabPFN", "XGBoost", "Lasso", "RandomForest", "SVM"]
    
    for feature_type, models in results.items():
        for model_name in models_order:
            if model_name not in models:
                continue
                
            train_metrics = models[model_name]['train_metrics']
            train_entry = {
                'feature_type': feature_type,
                'model_name': model_name,
                'data_set': 'train',
                'mse_log': train_metrics.get('mse_log', np.nan),
                'rmse_log': train_metrics.get('rmse_log', np.nan),
                'mae_log': train_metrics.get('mae_log', np.nan),
                'r2_log': train_metrics.get('r2_log', np.nan),
                'mse_original': train_metrics.get('mse_original', np.nan),
                'rmse_original': train_metrics.get('rmse_original', np.nan),
                'mae_original': train_metrics.get('mae_original', np.nan),
                'r2_original': train_metrics.get('r2_original', np.nan)
            }
            metrics_data.append(train_entry)
            
            if 'test_metrics' in models[model_name]:
                test_metrics = models[model_name]['test_metrics']
                test_entry = {
                    'feature_type': feature_type,
                    'model_name': model_name,
                    'data_set': 'test',
                    'mse_log': test_metrics.get('mse_log', np.nan),
                    'rmse_log': test_metrics.get('rmse_log', np.nan),
                    'mae_log': test_metrics.get('mae_log', np.nan),
                    'r2_log': test_metrics.get('r2_log', np.nan),
                    'mse_original': test_metrics.get('mse_original', np.nan),
                    'rmse_original': test_metrics.get('rmse_original', np.nan),
                    'mae_original': test_metrics.get('mae_original', np.nan),
                    'r2_original': test_metrics.get('r2_original', np.nan)
                }
                metrics_data.append(test_entry)
    
    df = pd.DataFrame(metrics_data)
    csv_path = os.path.join(output_dir, 'regression_metrics_summary.csv')
    df.to_csv(csv_path, index=False)

def load_raw_data(feature_file, label_file):
    df_feat = pd.read_csv(feature_file)
    df_label = pd.read_csv(label_file, header=None, names=['hemorrhage_volume','filename'])
    
    for df in (df_feat, df_label):
        df['filename'] = df['filename'].astype(str).str.split('.').str[0].str.strip().str.upper()
    
    return pd.merge(df_feat, df_label, on='filename')

def load_independent_test_filenames():
    with open(INDEPENDENT_TEST_FILE, 'r') as f:
        filenames = [line.strip().upper() for line in f.readlines()]
    return filenames

def validate_log_transformation(df_combined, output_dir):
    y_log = df_combined['hemorrhage_volume'].values
    y_original = np.exp(y_log)
    
    sample_size = min(5000, len(y_original))
    y_original_sample = np.random.choice(y_original, sample_size, replace=False)
    y_log_sample = np.random.choice(y_log, sample_size, replace=False)
    
    stat_original, p_original = shapiro(y_original_sample)
    stat_log, p_log = shapiro(y_log_sample)
    
    test_result = {
        "shapiro_test": {
            "sample_size": sample_size,
            "original_volume": {"statistic": stat_original, "p_value": p_original},
            "log_volume": {"statistic": stat_log, "p_value": p_log},
            "conclusion": "Log transformation significantly improves normality" if p_log > p_original else "Log transformation does not significantly improve normality"
        }
    }
    with open(os.path.join(output_dir, 'data_distribution', 'log_transformation_test.json'), 'w') as f:
        json.dump(test_result, f, indent=4)

def load_data():
    independent_test_filenames = load_independent_test_filenames()
    df_cli = load_raw_data(FEATURE_FILE_CLI, LABEL_FILE)
    df_rad = load_raw_data(FEATURE_FILE_RAD, LABEL_FILE)
    
    df_cli = df_cli.sort_values('filename')
    df_rad = df_rad.sort_values('filename')
    
    common_filenames = set(df_cli['filename']).intersection(set(df_rad['filename']))
    df_cli = df_cli[df_cli['filename'].isin(common_filenames)]
    df_rad = df_rad[df_rad['filename'].isin(common_filenames)]
    
    if 'hemorrhage_volume' in df_rad.columns:
        df_rad = df_rad.drop(columns=['hemorrhage_volume'])
    
    df_combined = pd.merge(df_cli[['filename', 'hemorrhage_volume']], df_rad, on='filename')
    df_combined = pd.merge(df_combined, df_cli.drop(columns=['hemorrhage_volume']), on='filename')
    
    validate_log_transformation(df_combined, OUTPUT_DIR)
    
    df_independent_test = df_combined[df_combined['filename'].isin(independent_test_filenames)]
    df_temp = df_combined[~df_combined['filename'].isin(independent_test_filenames)]
    
    clinical_features = [col for col in df_cli.columns if col not in ['filename', 'hemorrhage_volume']]
    rad_only_features = [col for col in df_rad.columns if col not in ['filename', 'hemorrhage_volume']]
    
    cli_features = [col for col in df_temp.columns if col in clinical_features]
    rad_features = [col for col in df_temp.columns if col not in ['filename', 'hemorrhage_volume'] and col not in clinical_features]
    
    X_temp_cli = df_temp[cli_features].values
    X_temp_rad = df_temp[rad_features].values
    y_temp = df_temp["hemorrhage_volume"].values
    
    X_ind_cli = df_independent_test[cli_features].values
    X_ind_rad = df_independent_test[rad_features].values
    y_ind = df_independent_test["hemorrhage_volume"].values
    
    temp_filenames = df_temp['filename'].values
    ind_filenames = df_independent_test['filename'].values
    
    return (X_temp_cli, X_temp_rad, y_temp, temp_filenames,
            X_ind_cli, X_ind_rad, y_ind, ind_filenames,
            cli_features, rad_features)

def prepare_datasets_with_pretrained_features(X_temp_cli, X_temp_rad, y_temp, temp_filenames,
                                             X_ind_cli, X_ind_rad, y_ind, ind_filenames,
                                             cli_features, rad_features):
    model_data = joblib.load(CLASSIFICATION_MODEL_PATH)
    cli_preprocessor = model_data["cli_preprocessor"]
    rad_preprocessor = model_data["rad_preprocessor"]
    cli_feature_names = model_data["cli_feature_names"]
    rad_feature_names = model_data["rad_feature_names"]
    
    df_temp_cli = pd.DataFrame(X_temp_cli, columns=cli_features)
    df_temp_cli = df_temp_cli[cli_feature_names]

    df_ind_cli = pd.DataFrame(X_ind_cli, columns=cli_features)
    df_ind_cli = df_ind_cli[cli_feature_names]  
    
    df_temp_rad = pd.DataFrame(X_temp_rad, columns=rad_features)
    df_temp_rad = df_temp_rad[rad_feature_names]
    
    df_ind_rad = pd.DataFrame(X_ind_rad, columns=rad_features)
    df_ind_rad = df_ind_rad[rad_feature_names]
    
    X_temp_concat = np.hstack([df_temp_cli, df_temp_rad])
    X_ind_concat = np.hstack([df_ind_cli, df_ind_rad])
    concat_feature_names = cli_feature_names + rad_feature_names
    
    return {
        "clinical": (df_temp_cli, y_temp, df_ind_cli, y_ind, cli_feature_names, temp_filenames, ind_filenames),
        "radiomics": (df_temp_rad, y_temp, df_ind_rad, y_ind, rad_feature_names, temp_filenames, ind_filenames),
        "concatenated": (X_temp_concat, y_temp, X_ind_concat, y_ind, concat_feature_names, temp_filenames, ind_filenames)
    }

def train_and_evaluate_model(model, model_name, X_train, y_train_log, X_test, y_test_log, 
                            feature_names, feature_type, output_dir):
    if model_name == "TabPFN":
        model.fit(X_train, y_train_log)
    else:
        if hasattr(model, 'fit'):
            model.fit(X_train, y_train_log)
    
    y_pred_train_log = model.predict(X_train)
    y_pred_test_log = model.predict(X_test)
    
    train_metrics = compute_regression_metrics(y_train_log, y_pred_train_log)
    test_metrics = compute_regression_metrics(y_test_log, y_pred_test_log)
    
    y_train_original = np.exp(y_train_log) if 'mse_original' in train_metrics else None
    y_pred_train_original = np.exp(y_pred_train_log) if 'mse_original' in train_metrics else None
    y_test_original = np.exp(y_test_log) if 'mse_original' in test_metrics else None
    y_pred_test_original = np.exp(y_pred_test_log) if 'mse_original' in test_metrics else None
    
    return {
        "model": model,
        "train_metrics": train_metrics,
        "test_metrics": test_metrics,
        "y_train_log": y_train_log,
        "y_pred_train_log": y_pred_train_log,
        "y_train_original": y_train_original,
        "y_pred_train_original": y_pred_train_original,
        "y_test_log": y_test_log,
        "y_pred_test_log": y_pred_test_log,
        "y_test_original": y_test_original,
        "y_pred_test_original": y_pred_test_original,
        "feature_names": feature_names
    }

def save_results(results, output_dir):
    json_safe = {}
    for feature_type, models in results.items():
        json_safe[feature_type] = {}
        for model_name, model_data in models.items():
            json_safe[feature_type][model_name] = {
                "train_metrics": model_data["train_metrics"],
                "test_metrics": model_data["test_metrics"]
            }
    
    with open(os.path.join(output_dir, 'evaluation_results.json'), 'w') as f:
        json.dump(json_safe, f, indent=4)

    for feature_type, models in results.items():
        for model_name, model_data in models.items():
            if model_name == "TabPFN":
                continue
                
            model_path = os.path.join(output_dir, f'{feature_type}_{model_name}_model.joblib')
            joblib.dump(model_data['model'], model_path)

def predict_and_save(results, datasets, output_dir):
    pred_dir = os.path.join(OUTPUT_DIR, "predictions")
    os.makedirs(pred_dir, exist_ok=True)
    models_order = ["TabPFN", "XGBoost", "Lasso", "RandomForest", "SVM"]

    for feature_type in datasets.keys():
        X_train, y_train_log, X_test, y_test_log, _, train_filenames, test_filenames = datasets[feature_type]
        
        y_train_original = np.exp(y_train_log) if 'mse_original' in results[feature_type][models_order[0]]['train_metrics'] else None
        y_test_original = np.exp(y_test_log) if 'mse_original' in results[feature_type][models_order[0]]['test_metrics'] else None
        
        df_train = pd.DataFrame({
            "filename": train_filenames,
            "true_hemorrhage_volume_log": y_train_log,
            "true_hemorrhage_volume_original": y_train_original
        })
        for model_name in models_order:
            if model_name in results[feature_type]:
                df_train[f"{model_name}_pred_log"] = results[feature_type][model_name]["y_pred_train_log"]
                if y_train_original is not None:
                    df_train[f"{model_name}_pred_original"] = results[feature_type][model_name]["y_pred_train_original"]
        train_save_path = os.path.join(pred_dir, f"train_predictions_{feature_type}.csv")
        df_train.to_csv(train_save_path, index=False)
        
        df_test = pd.DataFrame({
            "filename": test_filenames,
            "true_hemorrhage_volume_log": y_test_log,
            "true_hemorrhage_volume_original": y_test_original
        })
        for model_name in models_order:
            if model_name in results[feature_type]:
                df_test[f"{model_name}_pred_log"] = results[feature_type][model_name]["y_pred_test_log"]
                if y_test_original is not None:
                    df_test[f"{model_name}_pred_original"] = results[feature_type][model_name]["y_pred_test_original"]
        test_save_path = os.path.join(pred_dir, f"test_predictions_{feature_type}.csv")
        df_test.to_csv(test_save_path, index=False)

    if PREDICT_CLI_FILE and PREDICT_RAD_FILE:
        model_data = joblib.load(CLASSIFICATION_MODEL_PATH)
        cli_feature_names = model_data["cli_feature_names"]
        rad_feature_names = model_data["rad_feature_names"]
        
        df_cli_6 = pd.read_csv(PREDICT_CLI_FILE)
        df_rad_6 = pd.read_csv(PREDICT_RAD_FILE)
        
        for df in [df_cli_6, df_rad_6]:
            df["filename"] = df["filename"].astype(str).str.split(".").str[0].str.strip().str.upper()
        
        df_6_combined = pd.merge(
            df_cli_6, df_rad_6, on="filename", suffixes=("_cli", "_rad")
        ).drop_duplicates("filename")
        
        has_original_label = False
        if PREDICT_LABEL_FILE:
            df_label_6 = pd.read_csv(PREDICT_LABEL_FILE, header=None, names=["hemorrhage_volume_original", "filename"])
            df_label_6["filename"] = df_label_6["filename"].astype(str).str.split(".").str[0].str.strip().str.upper()
            df_6_combined = pd.merge(df_6_combined, df_label_6, on="filename", how="left")
            df_6_combined["hemorrhage_volume_log"] = np.log(df_6_combined["hemorrhage_volume_original"])
            has_original_label = True
        
        missing_cli = [f for f in cli_feature_names if f not in df_6_combined.columns]
        if missing_cli:
            raise ValueError(f"6 cases are missing clinical features: {missing_cli}")
        X_6_cli = df_6_combined[cli_feature_names]
        
        missing_rad = [f for f in rad_feature_names if f not in df_6_combined.columns]
        if missing_rad:
            raise ValueError(f"6 cases are missing radiomics features: {missing_rad}")
        X_6_rad = df_6_combined[rad_feature_names]
        
        X_6_concat = np.hstack([X_6_cli, X_6_rad])
        
        for feat_type, X_6 in [
            ("clinical", X_6_cli),
            ("radiomics", X_6_rad),
            ("concatenated", X_6_concat)
        ]:
            df_6_pred = pd.DataFrame({
                "filename": df_6_combined["filename"].values
            })
            if has_original_label:
                df_6_pred["true_hemorrhage_volume_original"] = df_6_combined["hemorrhage_volume_original"].values
                df_6_pred["true_hemorrhage_volume_log"] = df_6_combined["hemorrhage_volume_log"].values
            
            for model_name in models_order:
                if model_name == "TabPFN":
                    if feat_type in results and model_name in results[feat_type]:
                        tabpfn_model = results[feat_type][model_name]["model"]
                        y_pred_log_6 = tabpfn_model.predict(X_6)
                        df_6_pred[f"{model_name}_pred_log"] = y_pred_log_6
                        df_6_pred[f"{model_name}_pred_original"] = np.exp(y_pred_log_6)
                    continue
                
                model_path = os.path.join(OUTPUT_DIR, f"{feat_type}_{model_name}_model.joblib")
                if not os.path.exists(model_path):
                    continue
                
                model = joblib.load(model_path)
                y_pred_log_6 = model.predict(X_6)
                df_6_pred[f"{model_name}_pred_log"] = y_pred_log_6
                df_6_pred[f"{model_name}_pred_original"] = np.exp(y_pred_log_6)
            
            save_path_6 = os.path.join(pred_dir, f"6_cases_predictions_{feat_type}.csv")
            df_6_pred.to_csv(save_path_6, index=False)

def print_summary(results):
    print("\nModel Evaluation Results Summary:")
    print("=" * 80)
    
    for feature_type, models in results.items():
        print(f"\nFeature Type: {feature_type}")
        print("-" * 60)
        
        models_order = ["TabPFN", "XGBoost", "Lasso", "RandomForest", "SVM"]
        for model_name in models_order:
            if model_name not in models:
                continue
                
            train = models[model_name]['train_metrics']
            test = models[model_name]['test_metrics']
            
            print(f"\n Model: {model_name}")
            print(f"  Training Set (Log): MSE={train['mse_log']:.4f}, RMSE={train['rmse_log']:.4f}, "
                  f"MAE={train['mae_log']:.4f}, R²={train['r2_log']:.4f}")
            print(f"  Test Set (Log): MSE={test['mse_log']:.4f}, RMSE={test['rmse_log']:.4f}, "
                  f"MAE={test['mae_log']:.4f}, R²={test['r2_log']:.4f}")
            if 'mse_original' in train and 'mse_original' in test:
                print(f"  Training Set (Original): MSE={train['mse_original']:.4f}, RMSE={train['rmse_original']:.4f}, "
                      f"MAE={train['mae_original']:.4f}, R²={train['r2_original']:.4f}")
                print(f"  Test Set (Original): MSE={test['mse_original']:.4f}, RMSE={test['rmse_original']:.4f}, "
                      f"MAE={test['mae_original']:.4f}, R²={test['r2_original']:.4f}")

def main():
    print("=" * 80)
    print("Starting regression analysis process")
    print(f"Current time: {pd.Timestamp.now()}")
    print(f"Output directory: {OUTPUT_DIR}")
    print("=" * 80)

    try:
        print("\n----- 1. Loading data -----")
        (X_temp_cli, X_temp_rad, y_temp, temp_filenames,
         X_ind_cli, X_ind_rad, y_ind, ind_filenames,
         cli_features, rad_features) = load_data()

        print(f"\n----- 2. Preparing feature datasets -----")
        datasets = prepare_datasets_with_pretrained_features(
            X_temp_cli, X_temp_rad, y_temp, temp_filenames,
            X_ind_cli, X_ind_rad, y_ind, ind_filenames,
            cli_features, rad_features
        )

        print("\n----- 3. Training all models -----")
        results = {}
        models_order = ["TabPFN", "XGBoost", "Lasso", "RandomForest", "SVM"]
        for feature_type, (X_train, y_train_log, X_test, y_test_log, feature_names, _, _) in datasets.items():
            print(f"\n===== Training {feature_type} feature models =====")
            results[feature_type] = {}
            
            for model_name in models_order:
                model = MODELS[model_name]
                try:
                    model_clone = clone(model)
                except:
                    model_clone = model
                
                print(f"\n----- Training {model_name} -----")
                model_results = train_and_evaluate_model(
                    model_clone, model_name, 
                    X_train, y_train_log, X_test, y_test_log,
                    feature_names, feature_type, OUTPUT_DIR
                )
                results[feature_type][model_name] = model_results
                r2_log = model_results['test_metrics']['r2_log']
                r2_original = model_results['test_metrics'].get('r2_original', 'N/A')
                print(f"{model_name} training completed (Test Set R²-Log: {r2_log:.4f}, Original: {r2_original})")

        print("\n----- 4. Saving models and evaluation metrics -----")
        save_results(results, OUTPUT_DIR)
        save_regression_metrics_to_csv(results, OUTPUT_DIR)
        print("\n----- 5. Generating prediction result CSVs -----")
        predict_and_save(results, datasets, OUTPUT_DIR)

        print_summary(results)
        print(f"\n----- All processes completed -----")

    except Exception as e:
        print(f"Process execution error: {str(e)}")
        raise

if __name__ == "__main__":
    main()
