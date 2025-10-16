import os
import warnings
os.environ['SCIPY_ARRAY_API'] = '1'
import json
import joblib
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.base import clone, BaseEstimator, ClassifierMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import mutual_info_classif, RFE, VarianceThreshold
from sklearn.linear_model import LassoCV, LogisticRegression
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    roc_auc_score,
    accuracy_score,
    recall_score,
    precision_score
)
from sklearn.model_selection import (
    RepeatedStratifiedKFold,
    StratifiedKFold,
    cross_val_score,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from joblib import Parallel, delayed
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from tabpfn import TabPFNClassifier
from tabpfn_extensions.post_hoc_ensembles.sklearn_interface import AutoTabPFNClassifier
import traceback
from xgboost import XGBClassifier
from sklearn.svm import SVC

warnings.filterwarnings("ignore")


class AttentionFusionDataset(Dataset):
    def __init__(self, X_cli, X_rad, y):
        self.X_cli = torch.tensor(X_cli, dtype=torch.float32)
        self.X_rad = torch.tensor(X_rad, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32).view(-1, 1)
    
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        return self.X_cli[idx], self.X_rad[idx], self.y[idx]


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
        cli_att = cli_out * att[:, 0:1]
        rad_att = rad_out * att[:, 1:2]
        fused = torch.cat([cli_att, rad_att], dim=1)
        return self.final(fused), att


class AttentionFusionClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, num_cli_features, num_rad_features, hidden_dim=32,
                 dropout_rate=0.3, learning_rate=1e-3, batch_size=16,
                 num_epochs=100, device=None):
        self.num_cli_features = num_cli_features
        self.num_rad_features = num_rad_features
        self.hidden_dim = hidden_dim
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.scaler_cli = StandardScaler()
        self.scaler_rad = StandardScaler()
        self.model = None
        self.optimizer = None
        self.criterion = nn.BCELoss()
    
    def fit(self, X_cli, X_rad, y):
        Xc = self.scaler_cli.fit_transform(X_cli)
        Xr = self.scaler_rad.fit_transform(X_rad)
        ds = AttentionFusionDataset(Xc, Xr, y)
        loader = DataLoader(ds, batch_size=self.batch_size, shuffle=True, 
                           num_workers=min(4, os.cpu_count()),
                           pin_memory=True)
        
        self.model = AttentionFusionModel(self.num_cli_features,
                                         self.num_rad_features,
                                         self.hidden_dim,
                                         self.dropout_rate).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        
        best_loss = float('inf')
        wait = 0
        patience = 15
        
        for epoch in range(self.num_epochs):
            self.model.train()
            total_loss = 0
            
            for xc, xr, yy in loader:
                xc, xr, yy = xc.to(self.device, non_blocking=True), xr.to(self.device, non_blocking=True), yy.to(self.device, non_blocking=True)
                self.optimizer.zero_grad()
                out, _ = self.model(xc, xr)
                loss = self.criterion(out, yy)
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item() * xc.size(0)
            
            epoch_loss = total_loss / len(ds)
            
            if epoch_loss < best_loss:
                best_loss = epoch_loss
                wait = 0
                torch.save(self.model.state_dict(), os.path.join(OUTPUT_DIR, 'best_model.pth'))
            else:
                wait += 1
                if wait >= patience:
                    print(f"Early stop @ epoch {epoch+1}")
                    break
        
        self.model.load_state_dict(torch.load(os.path.join(OUTPUT_DIR, 'best_model.pth')))
        self.model.eval()
        return self
    
    def predict_proba(self, X_cli, X_rad):
        Xc = self.scaler_cli.transform(X_cli)
        Xr = self.scaler_rad.transform(X_rad)
        ds = AttentionFusionDataset(Xc, Xr, np.zeros(len(Xc)))
        loader = DataLoader(ds, batch_size=self.batch_size, num_workers=min(4, os.cpu_count()))
        probs = []
        with torch.no_grad():
            for xc, xr, _ in loader:
                xc, xr = xc.to(self.device, non_blocking=True), xr.to(self.device, non_blocking=True)
                out, _ = self.model(xc, xr)
                probs.append(out.cpu().numpy())
        return np.vstack(probs).ravel()
    
    def predict(self, X_cli, X_rad, threshold=0.5):
        return (self.predict_proba(X_cli, X_rad) > threshold).astype(int)
    
    def get_attention_weights(self, X_cli, X_rad):
        Xc = self.scaler_cli.transform(X_cli)
        Xr = self.scaler_rad.transform(X_rad)
        ds = AttentionFusionDataset(Xc, Xr, np.zeros(len(Xc)))
        loader = DataLoader(ds, batch_size=self.batch_size)
        weights = []
        with torch.no_grad():
            for xc, xr, _ in loader:
                xc, xr = xc.to(self.device), xr.to(self.device)
                _, att = self.model(xc, xr)
                weights.append(att.cpu().numpy())
        return np.vstack(weights)


FEATURE_FILE_RAD = "backend/src/data/Data_post_extracted/Radiomics_Features.csv"
FEATURE_FILE_CLI = "backend/src/data/Data_post_extracted/Clinical_Feature.csv"
LABEL_FILE = "backend/src/data/Data_post_extracted/Clissifier_Label.csv"
MODEL_PATH = "backend/src/data/Processing/final_attention_fusion_model.joblib"
OUTPUT_DIR = "/root/CODING_PROJECT/backend/src/data/Processing"
FINAL_TEST_SIZE = 0.15
RANDOM_STATE = 42
N_REPEATS = 10
N_SPLITS = 5
N_JOBS = os.cpu_count()

os.makedirs(OUTPUT_DIR, exist_ok=True)


def compute_metrics(y_true, y_pred, y_proba):
    metrics = {
        'auc': roc_auc_score(y_true, y_proba) if len(np.unique(y_true)) > 1 else 0.5,
        'f1': f1_score(y_true, y_pred),
        'accuracy': accuracy_score(y_true, y_pred),
        'sensitivity': recall_score(y_true, y_pred),
        'specificity': recall_score(y_true, y_pred, pos_label=0),
        'precision': precision_score(y_true, y_pred)
    }
    
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    metrics['confusion_matrix'] = [[tn, fp], [fn, tp]]
    
    return metrics


def compute_ci(metric_values):
    mean_val = np.mean(metric_values)
    std_val = np.std(metric_values)
    ci_low = mean_val - 1.96 * std_val
    ci_high = mean_val + 1.96 * std_val
    return mean_val, (ci_low, ci_high)


def convert_to_serializable(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: convert_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(v) for v in obj]
    elif isinstance(obj, tuple):
        return tuple(convert_to_serializable(v) for v in obj)
    else:
        return obj


def load_raw_data(feature_file, label_file):
    df_feat = pd.read_csv(feature_file)
    df_label = pd.read_csv(label_file, header=None, names=['label', 'filename'])
    for df in (df_feat, df_label):
        df['filename'] = df['filename'].astype(str).str.split('.').str[0].str.strip().str.upper()
    return pd.merge(df_feat, df_label, on='filename')


def preprocess_data(df, fit_objects=None):
    feature_cols = [c for c in df.columns if c not in ['filename', 'label']]
    X = df[feature_cols].values
    y = df['label'].astype(int).values if 'label' in df else None
    
    if fit_objects is None:
        imp = SimpleImputer(strategy='median')
        X_imp = imp.fit_transform(X)
        vt = VarianceThreshold(threshold=0.1)
        X_var = vt.fit_transform(X_imp)
        names = [feature_cols[i] for i in vt.get_support(indices=True)]
        return X_var, y, names, {'imputer': imp, 'variance_threshold': vt}
    else:
        imp = fit_objects['imputer']
        vt = fit_objects['variance_threshold']
        X_imp = imp.transform(X)
        X_var = vt.transform(X_imp)
        names = [feature_cols[i] for i in vt.get_support(indices=True)]
        return X_var, y, names, fit_objects


def analyze_features(X, y, feature_names):
    results = {}
    p_values = []
    effect_sizes = []
    
    df = pd.DataFrame(X, columns=feature_names)
    df['label'] = y
    
    group0 = df[df['label'] == 0].drop(columns='label')
    group1 = df[df['label'] == 1].drop(columns='label')
    
    for feature in feature_names:
        t_stat, p_val = stats.ttest_ind(group0[feature], group1[feature], 
                                       equal_var=False, nan_policy='omit')
        pooled_std = np.sqrt((group0[feature].std()**2 + group1[feature].std()**2) / 2)
        d = (group1[feature].mean() - group0[feature].mean()) / pooled_std
        
        p_values.append(p_val)
        effect_sizes.append(d)
        
        results[feature] = {
            'mean_0': group0[feature].mean(),
            'mean_1': group1[feature].mean(),
            'std_0': group0[feature].std(),
            'std_1': group1[feature].std(),
            't_stat': t_stat,
            'p_value': p_val,
            'effect_size': d
        }
    
    df_results = pd.DataFrame({
        'Feature': feature_names,
        'p_value': p_values,
        'effect_size': effect_sizes
    })
    
    df_results['significance'] = df_results['p_value'].apply(
        lambda p: '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else ''
    )
    
    df_results = df_results.sort_values('effect_size', key=abs, ascending=False)
    
    return results, df_results


def train_and_evaluate_model(model, X_train, y_train, X_test, y_test, model_name):
    results = {}
    
    model.fit(X_train, y_train)
    
    y_train_pred = model.predict(X_train)
    y_train_proba = model.predict_proba(X_train)[:, 1] if hasattr(model, "predict_proba") else np.zeros(len(y_train))
    train_metrics = compute_metrics(y_train, y_train_pred, y_train_proba)
    
    y_test_pred = model.predict(X_test)
    y_test_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else np.zeros(len(y_test))
    test_metrics = compute_metrics(y_test, y_test_pred, y_test_proba)
    
    results['train'] = train_metrics
    results['test'] = test_metrics
    results['train_confusion_matrix'] = train_metrics['confusion_matrix']
    results['test_confusion_matrix'] = test_metrics['confusion_matrix']
    
    return results


def train_evaluate_with_cv(X_train, y_train, model, model_name):
    rskf = RepeatedStratifiedKFold(n_splits=N_SPLITS, n_repeats=N_REPEATS, random_state=439)
    
    auc_scores, f1_scores, sens_scores, spec_scores = [], [], [], []
    all_predictions = []
    
    total_folds = N_REPEATS * N_SPLITS
    progress_bar = tqdm(total=total_folds, desc=f"Cross-validating {model_name}")
    
    for train_index, test_index in rskf.split(X_train, y_train):
        X_fold_train, X_fold_val = X_train[train_index], X_train[test_index]
        y_fold_train, y_fold_val = y_train[train_index], y_train[test_index]
        
        fold_model = clone(model)
        fold_model.fit(X_fold_train, y_fold_train)
        
        y_pred = fold_model.predict(X_fold_val)
        y_proba = fold_model.predict_proba(X_fold_val)[:, 1] if hasattr(fold_model, "predict_proba") else None
        
        if y_proba is not None and len(np.unique(y_fold_val)) > 1:
            auc_score = roc_auc_score(y_fold_val, y_proba)
        else:
            auc_score = 0.5
        auc_scores.append(auc_score)
        f1_scores.append(f1_score(y_fold_val, y_pred))
        sens_scores.append(recall_score(y_fold_val, y_pred))
        spec_scores.append(recall_score(y_fold_val, y_pred, pos_label=0))
        
        all_predictions.append({
            'indices': test_index.tolist(),
            'y_true': y_fold_val.tolist(),
            'y_pred': y_pred.tolist()
        })
        
        progress_bar.update(1)
    
    progress_bar.close()
    
    cv_results = {
        'auc': {'mean': np.mean(auc_scores), 'ci': compute_ci(auc_scores)[1]},
        'f1': {'mean': np.mean(f1_scores), 'ci': compute_ci(f1_scores)[1]},
        'sensitivity': {'mean': np.mean(sens_scores), 'ci': compute_ci(sens_scores)[1]},
        'specificity': {'mean': np.mean(spec_scores), 'ci': compute_ci(spec_scores)[1]},
        'all_predictions': all_predictions
    }
    
    return cv_results


def train_evaluate_models(datasets, model_data):
    all_results = {}
    
    classifiers = {
        'LogisticRegression': LogisticRegression(
            penalty="l2", C=0.1, solver="lbfgs", max_iter=2000, random_state=439
        ),
        'RandomForest': RandomForestClassifier(
            n_estimators=200, max_depth=10, min_samples_split=3,
            class_weight="balanced", random_state=439, n_jobs=N_JOBS
        ),
        'XGBoost': XGBClassifier(
            learning_rate=0.1, 
            n_estimators=200, 
            max_depth=5,
            subsample=0.8, 
            colsample_bytree=0.8, 
            random_state=439, 
            n_jobs=N_JOBS
        ),
        'SVM': SVC(
            C=1.0, 
            kernel='rbf', 
            gamma='scale',
            probability=True, 
            random_state=439
        ),
        'TabPFN': TabPFNClassifier(
            device="cuda" if torch.cuda.is_available() else "cpu",
            model_path='backend/src/data/Data_post_extracted/tabpfn-v2-classifier.ckpt'
        )
    }
    
    X_train_cli, y_train, X_test_cli, y_test, _ = datasets['clinical']
    print("\n" + "="*50)
    print("Training and Evaluating Clinical Model")
    print("="*50)
    
    cli_results = {}
    
    for clf_name, clf in classifiers.items():
        if X_train_cli.shape[1] > 100 and clf_name == "TabPFN":
            print(f"Skipping TabPFN for clinical as features > 100")
            continue
            
        print(f"\nTraining {clf_name} classifier...")
        cli_results[clf_name] = {}
        
        cv_results = train_evaluate_with_cv(
            X_train_cli, y_train, clone(clf), f"Clinical_{clf_name}"
        )
        cli_results[clf_name]['cv'] = cv_results
        
        final_model = clone(clf)
        eval_results = train_and_evaluate_model(
            final_model, X_train_cli, y_train, X_test_cli, y_test, 
            f"Clinical_{clf_name}"
        )
        cli_results[clf_name]['train'] = eval_results['train']
        cli_results[clf_name]['test'] = eval_results['test']
        
        joblib.dump(final_model, os.path.join(OUTPUT_DIR, f"Classifier_clinical_{clf_name}.joblib"))
    
    best_clinical_clf = None
    best_test_auc = 0.0
    for clf_name, results in cli_results.items():
        test_auc = results['test']['auc']
        if test_auc > best_test_auc:
            best_test_auc = test_auc
            best_clinical_clf = clf_name
    
    print(f"\nBest Clinical Classifier: {best_clinical_clf} (Test AUC: {best_test_auc:.4f})")
    all_results['clinical'] = {
        'results': cli_results,
        'best_classifier': best_clinical_clf
    }
    
    X_train_rad, y_train, X_test_rad, y_test, _ = datasets['radiomics']
    print("\n" + "="*50)
    print("Training and Evaluating Radiomics Model")
    print("="*50)
    
    rad_results = {}
    
    for clf_name, clf in classifiers.items():
        if X_train_rad.shape[1] > 100 and clf_name == "TabPFN":
            print(f"Skipping TabPFN for radiomics as features > 100")
            continue
            
        print(f"\nTraining {clf_name} classifier...")
        rad_results[clf_name] = {}
        
        cv_results = train_evaluate_with_cv(
            X_train_rad, y_train, clone(clf), f"Radiomics_{clf_name}"
        )
        rad_results[clf_name]['cv'] = cv_results
        
        final_model = clone(clf)
        eval_results = train_and_evaluate_model(
            final_model, X_train_rad, y_train, X_test_rad, y_test, 
            f"Radiomics_{clf_name}"
        )
        rad_results[clf_name]['train'] = eval_results['train']
        rad_results[clf_name]['test'] = eval_results['test']
        
        joblib.dump(final_model, os.path.join(OUTPUT_DIR, f"final_radiomics_{clf_name}.joblib"))
    
    best_rad_clf = None
    best_test_auc = 0.0
    for clf_name, results in rad_results.items():
        test_auc = results['test']['auc']
        if test_auc > best_test_auc:
            best_test_auc = test_auc
            best_rad_clf = clf_name
    
    print(f"\nBest Radiomics Classifier: {best_rad_clf} (Test AUC: {best_test_auc:.4f})")
    all_results['radiomics'] = {
        'results': rad_results,
        'best_classifier': best_rad_clf
    }
    
    X_train_concat, y_train, X_test_concat, y_test, _ = datasets['concatenated']
    print("\n" + "="*50)
    print("Training and Evaluating Concatenated Model")
    print("="*50)
    
    concat_results = {}
    
    for clf_name, clf in classifiers.items():
        if X_train_concat.shape[1] > 100 and clf_name == "TabPFN":
            print(f"Skipping TabPFN for concatenated as features > 100")
            continue
            
        print(f"\nTraining {clf_name} classifier...")
        concat_results[clf_name] = {}
        
        cv_results = train_evaluate_with_cv(
            X_train_concat, y_train, clone(clf), f"Concatenated_{clf_name}"
        )
        concat_results[clf_name]['cv'] = cv_results
        
        final_model = clone(clf)
        eval_results = train_and_evaluate_model(
            final_model, X_train_concat, y_train, X_test_concat, y_test, 
            f"Concatenated_{clf_name}"
        )
        concat_results[clf_name]['train'] = eval_results['train']
        concat_results[clf_name]['test'] = eval_results['test']
        
        joblib.dump(final_model, os.path.join(OUTPUT_DIR, f"final_concatenated_{clf_name}.joblib"))
    
    best_concat_clf = None
    best_test_auc = 0.0
    for clf_name, results in concat_results.items():
        test_auc = results['test']['auc']
        if test_auc > best_test_auc:
            best_test_auc = test_auc
            best_concat_clf = clf_name
    
    print(f"\nBest Concatenated Classifier: {best_concat_clf} (Test AUC: {best_test_auc:.4f})")
    all_results['concatenated'] = {
        'results': concat_results,
        'best_classifier': best_concat_clf
    }
    
    print("\n" + "="*50)
    print("Evaluating Attention Fusion Model")
    print("="*50)
    
    fusion_results = {}
    X_train_cli, X_train_rad, y_train, X_test_cli, X_test_rad, y_test = datasets['attention']
    
    fusion_model = model_data["model"]
    
    y_train_proba = fusion_model.predict_proba(X_train_cli, X_train_rad)
    y_train_pred = fusion_model.predict(X_train_cli, X_train_rad)
    train_metrics = compute_metrics(y_train, y_train_pred, y_train_proba)
    
    y_test_proba = fusion_model.predict_proba(X_test_cli, X_test_rad)
    y_test_pred = fusion_model.predict(X_test_cli, X_test_rad)
    test_metrics = compute_metrics(y_test, y_test_pred, y_test_proba)
    
    fusion_results = {
        'train': train_metrics,
        'test': test_metrics
    }
    
    all_results['attention_fusion'] = fusion_results
    
    all_results_serializable = convert_to_serializable(all_results)
    with open(os.path.join(OUTPUT_DIR, "all_results.json"), "w") as f:
        json.dump(all_results_serializable, f, indent=4)

    return all_results


def load_data():
    df_cli = load_raw_data(FEATURE_FILE_CLI, LABEL_FILE)
    df_rad = load_raw_data(FEATURE_FILE_RAD, LABEL_FILE)
    
    df_cli = df_cli.sort_values('filename')
    df_rad = df_rad.sort_values('filename')
    
    common_filenames = set(df_cli['filename']).intersection(set(df_rad['filename']))
    df_cli = df_cli[df_cli['filename'].isin(common_filenames)]
    df_rad = df_rad[df_rad['filename'].isin(common_filenames)]
    
    if 'label' in df_rad.columns:
        df_rad = df_rad.drop(columns=['label'])
    
    df_combined = pd.merge(df_cli[['filename', 'label']], df_rad, on='filename')
    df_combined = pd.merge(df_combined, df_cli.drop(columns=['label']), on='filename')
    
    test_filenames = pd.read_csv("backend/src/data/Processing/dataset_split.txt", header=None)[0].tolist()
    df_independent_test = df_combined[df_combined['filename'].isin(test_filenames)]
    df_temp = df_combined[~df_combined['filename'].isin(test_filenames)]
    
    clinical_features = [col for col in df_cli.columns if col not in ['filename', 'label']]
    cli_features = [col for col in df_temp.columns if col in clinical_features]
    rad_features = [col for col in df_temp.columns if col not in ['filename', 'label'] and col not in clinical_features]
    
    X_temp_cli = df_temp[cli_features].values
    X_temp_rad = df_temp[rad_features].values
    y_temp = df_temp["label"].values
    
    X_ind_cli = df_independent_test[cli_features].values
    X_ind_rad = df_independent_test[rad_features].values
    y_ind = df_independent_test["label"].values
    
    return (X_temp_cli, X_temp_rad, y_temp, 
            X_ind_cli, X_ind_rad, y_ind, 
            cli_features, rad_features)


def prepare_datasets(X_temp_cli, X_temp_rad, y_temp,
                    X_ind_cli, X_ind_rad, y_ind,
                    cli_features, rad_features,
                    model_data):
    cli_preprocessor = model_data["cli_preprocessor"]
    rad_preprocessor = model_data["rad_preprocessor"]
    cli_feat_indices = model_data["cli_feature_indices"]
    rad_feat_indices = model_data["rad_feature_indices"]
    
    df_temp_cli = pd.DataFrame(X_temp_cli, columns=cli_features)
    X_temp_cli_proc, _, cli_feature_names_all, _ = preprocess_data(df_temp_cli, fit_objects=cli_preprocessor)
    
    df_temp_rad = pd.DataFrame(X_temp_rad, columns=rad_features)
    X_temp_rad_proc, _, rad_feature_names_all, _ = preprocess_data(df_temp_rad, fit_objects=rad_preprocessor)
    
    df_ind_cli = pd.DataFrame(X_ind_cli, columns=cli_features)
    X_ind_cli_proc, _, _, _ = preprocess_data(df_ind_cli, fit_objects=cli_preprocessor)
    
    df_ind_rad = pd.DataFrame(X_ind_rad, columns=rad_features)
    X_ind_rad_proc, _, _, _ = preprocess_data(df_ind_rad, fit_objects=rad_preprocessor)
    
    cli_feature_names = [cli_feature_names_all[i] for i in cli_feat_indices] if len(cli_feat_indices) > 0 else cli_feature_names_all
    rad_feature_names = [rad_feature_names_all[i] for i in rad_feat_indices] if len(rad_feat_indices) > 0 else rad_feature_names_all
    
    X_temp_cli_selected = X_temp_cli_proc[:, cli_feat_indices] if len(cli_feat_indices) > 0 else X_temp_cli_proc
    X_ind_cli_selected = X_ind_cli_proc[:, cli_feat_indices] if len(cli_feat_indices) > 0 else X_ind_cli_proc
    
    X_temp_rad_selected = X_temp_rad_proc[:, rad_feat_indices] if len(rad_feat_indices) > 0 else X_temp_rad_proc
    X_ind_rad_selected = X_ind_rad_proc[:, rad_feat_indices] if len(rad_feat_indices) > 0 else X_ind_rad_proc
    
    X_temp_concat = np.hstack([X_temp_cli_selected, X_temp_rad_selected])
    X_ind_concat = np.hstack([X_ind_cli_selected, X_ind_rad_selected])
    
    concat_feature_names = cli_feature_names + rad_feature_names
    
    return {
        "clinical": (X_temp_cli_selected, y_temp, X_ind_cli_selected, y_ind, cli_feature_names),
        "radiomics": (X_temp_rad_selected, y_temp, X_ind_rad_selected, y_ind, rad_feature_names),
        "concatenated": (X_temp_concat, y_temp, X_ind_concat, y_ind, concat_feature_names),
        "attention": (X_temp_cli_selected, X_temp_rad_selected, y_temp, 
                      X_ind_cli_selected, X_ind_rad_selected, y_ind)
    }


def main():
    print("Loading dataset...")
    (X_temp_cli, X_temp_rad, y_temp, 
     X_ind_cli, X_ind_rad, y_ind, 
     cli_features, rad_features) = load_data()
    
    print(f"Temporary set size: {len(y_temp)}, Test set size: {len(y_ind)}")
    print(f"Clinical features: {len(cli_features)}, Radiomics features: {len(rad_features)}")
    
    try:
        model_data = joblib.load(MODEL_PATH)
        print(f"Clinical features selected: {len(model_data['cli_feature_names'])}")
        print(f"Radiomics features selected: {len(model_data['rad_feature_names'])}")
    except Exception as e:
        print(f"Error loading original model: {str(e)}")
        return
    
    print("\nPreparing datasets for model comparison...")
    datasets = prepare_datasets(
        X_temp_cli, X_temp_rad, y_temp,
        X_ind_cli, X_ind_rad, y_ind,
        cli_features, rad_features,
        model_data
    )
    
    print("\nTraining and evaluating models...")
    all_results = train_evaluate_models(datasets, model_data)
    
    print("\n" + "="*100)
    print("SUMMARY OF MODEL PERFORMANCE")
    print("="*100)
    
    print("\n1. Clinical Model:")
    for clf_name, results in all_results['clinical']['results'].items():
        print(f"  {clf_name}:")
        print(f"    Train - AUC: {results['train']['auc']:.4f}, F1: {results['train']['f1']:.4f}, Accuracy: {results['train']['accuracy']:.4f}")
        print(f"             Sensitivity: {results['train']['sensitivity']:.4f}, Specificity: {results['train']['specificity']:.4f}")
        print(f"    Test  - AUC: {results['test']['auc']:.4f}, F1: {results['test']['f1']:.4f}, Accuracy: {results['test']['accuracy']:.4f}")
        print(f"             Sensitivity: {results['test']['sensitivity']:.4f}, Specificity: {results['test']['specificity']:.4f}")
    
    print("\n2. Radiomics Model:")
    for clf_name, results in all_results['radiomics']['results'].items():
        print(f"  {clf_name}:")
        print(f"    Train - AUC: {results['train']['auc']:.4f}, F1: {results['train']['f1']:.4f}, Accuracy: {results['train']['accuracy']:.4f}")
        print(f"             Sensitivity: {results['train']['sensitivity']:.4f}, Specificity: {results['train']['specificity']:.4f}")
        print(f"    Test  - AUC: {results['test']['auc']:.4f}, F1: {results['test']['f1']:.4f}, Accuracy: {results['test']['accuracy']:.4f}")
        print(f"             Sensitivity: {results['test']['sensitivity']:.4f}, Specificity: {results['test']['specificity']:.4f}")
    
    print("\n3. Concatenated Model:")
    for clf_name, results in all_results['concatenated']['results'].items():
        print(f"  {clf_name}:")
        print(f"    Train - AUC: {results['train']['auc']:.4f}, F1: {results['train']['f1']:.4f}, Accuracy: {results['train']['accuracy']:.4f}")
        print(f"             Sensitivity: {results['train']['sensitivity']:.4f}, Specificity: {results['train']['specificity']:.4f}")
        print(f"    Test  - AUC: {results['test']['auc']:.4f}, F1: {results['test']['f1']:.4f}, Accuracy: {results['test']['accuracy']:.4f}")
        print(f"             Sensitivity: {results['test']['sensitivity']:.4f}, Specificity: {results['test']['specificity']:.4f}")
    
    print("\n4. Attention Fusion Model:")
    fusion_res = all_results['attention_fusion']
    print(f"  Train - AUC: {fusion_res['train']['auc']:.4f}, F1: {fusion_res['train']['f1']:.4f}, Accuracy: {fusion_res['train']['accuracy']:.4f}")
    print(f"           Sensitivity: {fusion_res['train']['sensitivity']:.4f}, Specificity: {fusion_res['train']['specificity']:.4f}")
    print(f"  Test  - AUC: {fusion_res['test']['auc']:.4f}, F1: {fusion_res['test']['f1']:.4f}, Accuracy: {fusion_res['test']['accuracy']:.4f}")
    print(f"           Sensitivity: {fusion_res['test']['sensitivity']:.4f}, Specificity: {fusion_res['test']['specificity']:.4f}")
    
    print("\nComparison completed successfully! All results saved to", OUTPUT_DIR)


if __name__ == "__main__":
    main()