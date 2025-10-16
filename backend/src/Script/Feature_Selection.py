import os
import warnings
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
    recall_score
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
import random
from tqdm import tqdm

warnings.filterwarnings("ignore")

FEATURE_FILE_RAD = "backend/src/data/Data_post_extracted/Radiomics_Features.csv"
FEATURE_FILE_CLI = "backend/src/data/Data_post_extracted/Clinical_Feature.csv"
LABEL_FILE = "backend/src/data/Data_post_extracted/Clissifier_Label.csv"
OUTPUT_DIR = "backend/src/data/Processing"
RANDOM_SEED = 42
FINAL_TEST_SIZE = 0.15
TEMP_TEST_SIZE = 0.1765 
MIN_CLI_FEATURES = 1
MAX_CLI_DROPOUT = 1
M_VALUES = list(range(1, 34, 1))
N_SPLITS = 5
N_REPEATS = 50
N_JOBS = os.cpu_count()
N_JOBS_FEATURE_SELECTION = max(1, os.cpu_count() // 2)

os.makedirs(OUTPUT_DIR, exist_ok=True)

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
        g = torch.Generator()
        g.manual_seed(439)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(439)
        
        Xc = self.scaler_cli.fit_transform(X_cli)
        Xr = self.scaler_rad.fit_transform(X_rad)
        
        ds = AttentionFusionDataset(Xc, Xr, y)
        loader = DataLoader(
            ds, batch_size=self.batch_size, shuffle=True, 
            num_workers=min(4, os.cpu_count()),
            pin_memory=True,
            generator=g
        )
        
        self.model = AttentionFusionModel(
            self.num_cli_features,
            self.num_rad_features,
            self.hidden_dim,
            self.dropout_rate
        ).to(self.device)
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), 
            lr=self.learning_rate
        )
        
        best_loss = float('inf')
        wait = 0
        patience = 15
        
        for epoch in range(self.num_epochs):
            self.model.train()
            total_loss = 0
            
            for xc, xr, yy in loader:
                xc, xr, yy = xc.to(self.device, non_blocking=True), \
                             xr.to(self.device, non_blocking=True), \
                             yy.to(self.device, non_blocking=True)
                
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
        loader = DataLoader(
            ds, batch_size=self.batch_size, num_workers=min(4, os.cpu_count())
        )
        
        probs = []
        with torch.no_grad():
            for xc, xr, _ in loader:
                xc, xr = xc.to(self.device, non_blocking=True), \
                         xr.to(self.device, non_blocking=True)
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
    
np.random.seed(439)
random.seed(439)
torch.manual_seed(439)
if torch.cuda.is_available():
    torch.cuda.manual_seed(439)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

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

def compute_single_repeat(i, method, X, y, m, estimator=None):
    RANDOM_SEED = 439 + i
    
    if method in ["corr", "lasso"]:
        noise = np.random.normal(0, 1e-8, X.shape)
        X_perturbed = X + noise
    else:
        X_perturbed = X.copy()

    if method == "corr":
        corrs = np.array([abs(stats.pearsonr(X_perturbed[:, j], y)[0]) for j in range(X_perturbed.shape[1])])
        idx = np.argsort(corrs)[-m:]
    elif method == "lasso":
        lasso = LassoCV(cv=5, max_iter=2000, random_state=RANDOM_SEED).fit(X_perturbed, y)
        coefs = np.abs(lasso.coef_)
        idx = np.argsort(coefs)[-m:]
    elif method == "rf":
        rf = RandomForestClassifier(
            n_estimators=200, random_state=RANDOM_SEED, n_jobs=-1, class_weight="balanced"
        ).fit(X_perturbed, y)
        imp = rf.feature_importances_
        idx = np.argsort(imp)[-m:]
    elif method == "rfe":
        if estimator is None:
            raise ValueError("RFE requires an estimator")
        est_clone = clone(estimator)
        if hasattr(est_clone, 'random_state'):
            est_clone.random_state = RANDOM_SEED
        rfe = RFE(estimator=est_clone, n_features_to_select=m, step=0.1)
        rfe.fit(X_perturbed, y)
        idx = np.where(rfe.support_)[0]
    else:
        raise ValueError(f"Unknown method: {method}")
        
    return np.sort(idx)

def select_top_m_by_metric(X, y, m, method: str, estimator=None):
    n_repeats = 10
    results = Parallel(n_jobs=N_JOBS_FEATURE_SELECTION)(
        delayed(compute_single_repeat)(i, method, X, y, m, estimator)
        for i in range(n_repeats)
    )
    
    feature_counts = np.zeros(X.shape[1])
    for idx in results:
        feature_counts[idx] += 1
    
    m_final = min(m, X.shape[1])
    top_m_indices = np.argsort(feature_counts)[-m_final:]
    return np.sort(top_m_indices), feature_counts

def find_best_m(X_train, y_train, method: str, m_values, estimator=None):
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=439)
    
    def evaluate_m(m):
        if m > X_train.shape[1]:
            return 0.0, m
        try:
            idx, _ = select_top_m_by_metric(X_train, y_train, m, method, estimator=estimator)
            X_sub = X_train[:, idx]
            
            if X_sub.shape[1] == 0:
                return 0.0, m
            
            pipe = Pipeline([
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(
                    max_iter=2000, random_state=439, 
                    class_weight='balanced', n_jobs=-1
                ))
            ])
            
            scores = cross_val_score(pipe, X_sub, y_train, cv=cv, scoring="roc_auc", n_jobs=-1)
            return scores.mean(), m
        except Exception as e:
            print(f"Error evaluating m={m}: {str(e)}")
            return 0.0, m
    
    results = Parallel(n_jobs=N_JOBS)(
        delayed(evaluate_m)(m) for m in tqdm(m_values, desc=f"[{method}] Selecting optimal m")
    )
    
    auc_scores = [res[0] for res in results]
    best_auc, best_m = max(results, key=lambda x: x[0])
    
    if best_auc == 0.0:
        best_m = X_train.shape[1]
        
    return best_m, best_auc, auc_scores

def compute_auc_ci(y_true, y_prob, n_bootstraps=1000):
    rng = np.random.RandomState(439)
    n = len(y_true)
    
    def bootstrap_sample(_):
        idxs = rng.randint(0, n, n)
        if len(np.unique(y_true[idxs])) < 2:
            return 0.5
        return roc_auc_score(y_true[idxs], y_prob[idxs])
    
    scores = Parallel(n_jobs=N_JOBS)(
        delayed(bootstrap_sample)(i) for i in range(n_bootstraps)
    )
    
    scores = np.array(scores)
    scores.sort()
    low = scores[int(0.025 * len(scores))]
    high = scores[int(0.975 * len(scores))]
    
    return low, high

def compute_accuracy_ci(y_true, y_pred, n_bootstraps=1000):
    rng = np.random.RandomState(439)
    n = len(y_true)
    
    def bootstrap_sample(_):
        idxs = rng.randint(0, n, n)
        return accuracy_score(y_true[idxs], y_pred[idxs])
    
    scores = Parallel(n_jobs=N_JOBS)(
        delayed(bootstrap_sample)(i) for i in range(n_bootstraps)
    )
    
    scores = np.array(scores)
    scores.sort()
    low = scores[int(0.025 * len(scores))]
    high = scores[int(0.975 * len(scores))]
    
    return low, high

def protect_clinical_features(cli_features, preprocessed_names, m_opt):
    n_clinical = len([f for f in preprocessed_names if f in cli_features])
    max_drop = int(n_clinical * MAX_CLI_DROPOUT)
    min_retain = max(MIN_CLI_FEATURES, n_clinical - max_drop)
    
    if m_opt < min_retain:
        print(f"Adjusting clinical features from {m_opt} to {min_retain} (protect clinical)")
        return min_retain
    return m_opt

def main():
    print("Loading raw data...")
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
    
    df_temp, df_independent_test = train_test_split(
        df_combined, test_size=FINAL_TEST_SIZE, stratify=df_combined["label"], random_state=439
    )
    df_independent_test['filename'].to_csv("backend/src/data/Processing/dataset_split.txt", index=False, header=False)
    print(f"Total samples: {df_combined.shape[0]}, Independent test samples: {df_independent_test.shape[0]}")
    
    clinical_features = [col for col in df_cli.columns if col not in ['filename', 'label']]
    cli_features = [col for col in df_temp.columns if col in clinical_features]
    rad_features = [col for col in df_temp.columns if col not in ['filename', 'label'] and col not in clinical_features]
    
    print(f"Clinical features: {len(cli_features)}")
    print(f"Radiomics features: {len(rad_features)}")
    
    X_temp_cli = df_temp[cli_features].values
    X_temp_rad = df_temp[rad_features].values
    y_temp = df_temp["label"].values
    
    X_train_cli, X_val_cli, X_train_rad, X_val_rad, y_train, y_val = train_test_split(
        X_temp_cli, X_temp_rad, y_temp, 
        test_size=TEMP_TEST_SIZE, 
        stratify=y_temp, 
        random_state=439
    )
    print(f"Training set: {X_train_cli.shape[0]}, Validation set: {X_val_cli.shape[0]}")
    
    print("\nPreprocessing training set...")
    df_train_cli = pd.DataFrame(X_train_cli, columns=cli_features)
    X_train_cli_proc, _, cli_feature_names, cli_preprocessor = preprocess_data(df_train_cli)
    
    df_train_rad = pd.DataFrame(X_train_rad, columns=rad_features)
    X_train_rad_proc, _, rad_feature_names, rad_preprocessor = preprocess_data(df_train_rad)
    
    print(f"Clinical features after preprocessing: {len(cli_feature_names)}")
    print(f"Radiomics features after preprocessing: {len(rad_feature_names)}")
    
    print("Applying preprocessing to validation set...")
    df_val_cli = pd.DataFrame(X_val_cli, columns=cli_features)
    X_val_cli_proc, _, _, _ = preprocess_data(df_val_cli, fit_objects=cli_preprocessor)
    
    df_val_rad = pd.DataFrame(X_val_rad, columns=rad_features)
    X_val_rad_proc, _, _, _ = preprocess_data(df_val_rad, fit_objects=rad_preprocessor)
    
    sel_methods = {
        "corr": None,
        "lasso": None,
        "rf": None,
        "rfe": RandomForestClassifier(
            n_estimators=200, random_state=439, n_jobs=-1, class_weight="balanced"
        ),
    }
    
    best_m_info_cli = {}
    best_m_auc_curves_cli = {}
    
    print("\n==== Feature Selection for Clinical Features ====")
    for method, estimator in sel_methods.items():
        m_opt, auc_opt, auc_curve = find_best_m(
            X_train_cli_proc, y_train, method, M_VALUES, estimator=estimator
        )
        m_opt = protect_clinical_features(cli_features, cli_feature_names, m_opt)
        best_m_info_cli[method] = (m_opt, auc_opt)
        best_m_auc_curves_cli[method] = auc_curve
        print(f"Clinical - Method: {method:<12s} | Best m = {m_opt:2d} | CV AUC = {auc_opt:.4f}")
    
    best_m_info_rad = {}
    best_m_auc_curves_rad = {}
    
    print("\n==== Feature Selection for Radiomics Features ====")
    for method, estimator in sel_methods.items():
        m_opt, auc_opt, auc_curve = find_best_m(
            X_train_rad_proc, y_train, method, M_VALUES, estimator=estimator
        )
        best_m_info_rad[method] = (m_opt, auc_opt)
        best_m_auc_curves_rad[method] = auc_curve
        print(f"Radiomics - Method: {method:<12s} | Best m = {m_opt:2d} | CV AUC = {auc_opt:.4f}")
    
    selected_sets_cli = {}
    for method, (m_opt, _) in best_m_info_cli.items():
        estimator = sel_methods[method]
        idx, freq_array = select_top_m_by_metric(
            X_train_cli_proc, y_train, m_opt, method, estimator=estimator
        )
        feats = [cli_feature_names[i] for i in idx]
        selected_sets_cli[method] = {"indices": idx, "feats": feats, "freq_array": freq_array} 
    
    selected_sets_rad = {}
    for method, (m_opt, _) in best_m_info_rad.items():
        estimator = sel_methods[method]
        idx, freq_array = select_top_m_by_metric(
            X_train_rad_proc, y_train, m_opt, method, estimator=estimator
        )
        feats = [rad_feature_names[i] for i in idx]
        selected_sets_rad[method] = {"indices": idx, "feats": feats, "freq_array": freq_array}
    
    print("\nBuilding feature union for clinical and radiomics...")
    all_cli_idx = set()
    all_rad_idx = set()
    
    for method in selected_sets_cli:
        all_cli_idx |= set(selected_sets_cli[method]["indices"])
    for method in selected_sets_rad:
        all_rad_idx |= set(selected_sets_rad[method]["indices"])
    
    final_cli_idx = sorted(list(all_cli_idx))
    final_rad_idx = sorted(list(all_rad_idx))
    
    print(f"Final clinical features: {len(final_cli_idx)}")
    print(f"Final radiomics features: {len(final_rad_idx)}")
    
    X_tr_cli = X_train_cli_proc[:, final_cli_idx]
    X_tr_rad = X_train_rad_proc[:, final_rad_idx]
    X_val_cli = X_val_cli_proc[:, final_cli_idx]
    X_val_rad = X_val_rad_proc[:, final_rad_idx]
    
    print("\nTraining AttentionFusion model with feature union...")
    fusion_model = AttentionFusionClassifier(
        num_cli_features=X_tr_cli.shape[1],
        num_rad_features=X_tr_rad.shape[1],
        hidden_dim=32,
        dropout_rate=0.3,
        learning_rate=0.001,
        batch_size=16,
        num_epochs=150,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    
    fusion_model.fit(X_tr_cli, X_tr_rad, y_train)
    
    y_val_prob = fusion_model.predict_proba(X_val_cli, X_val_rad)
    y_val_pred = fusion_model.predict(X_val_cli, X_val_rad)
    
    auc_val = roc_auc_score(y_val, y_val_prob)
    auc_low, auc_high = compute_auc_ci(y_val, y_val_prob)
    acc_val = accuracy_score(y_val, y_val_pred)
    acc_low, acc_high = compute_accuracy_ci(y_val, y_val_pred)
    f1_val = f1_score(y_val, y_val_pred)
    tn, fp, fn, tp = confusion_matrix(y_val, y_val_pred).ravel()
    sensitivity_val = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity_val = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    
    att_weights = fusion_model.get_attention_weights(X_val_cli, X_val_rad)
    mean_cli_weight = att_weights[:, 0].mean()
    mean_rad_weight = att_weights[:, 1].mean()
    
    val_results = {
        "m_CLI": len(final_cli_idx),
        "m_RAD": len(final_rad_idx),
        "AUC": auc_val,
        "AUC_CI_Low": auc_low,
        "AUC_CI_High": auc_high,
        "Accuracy": acc_val,
        "Acc_CI_Low": acc_low,
        "Acc_CI_High": acc_high,
        "F1": f1_val,
        "Sensitivity": sensitivity_val,
        "Specificity": specificity_val,
        "Mean_CLI_Weight": mean_cli_weight,
        "Mean_RAD_Weight": mean_rad_weight
    }
    
    print("\n=== Validation Set Results ===")
    for k, v in val_results.items():
        print(f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}")
    
    print("\nTraining final fusion model on full dataset...")
    X_full_cli = np.vstack([X_train_cli_proc[:, final_cli_idx], X_val_cli_proc[:, final_cli_idx]])
    X_full_rad = np.vstack([X_train_rad_proc[:, final_rad_idx], X_val_rad_proc[:, final_rad_idx]])
    y_full = np.concatenate([y_train, y_val])
    
    final_fusion_model = AttentionFusionClassifier(
        num_cli_features=X_full_cli.shape[1],
        num_rad_features=X_full_rad.shape[1],
        hidden_dim=32,
        dropout_rate=0.3,
        learning_rate=0.001,
        batch_size=16,
        num_epochs=200,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    final_fusion_model.fit(X_full_cli, X_full_rad, y_full)
    
    print("\nApplying preprocessing to independent test set...")
    df_ind_test_cli = pd.DataFrame(df_independent_test[cli_features].values, columns=cli_features)
    X_ind_cli, _, _, _ = preprocess_data(df_ind_test_cli, fit_objects=cli_preprocessor)
    
    df_ind_test_rad = pd.DataFrame(df_independent_test[rad_features].values, columns=rad_features)
    X_ind_rad, _, _, _ = preprocess_data(df_ind_test_rad, fit_objects=rad_preprocessor)
    
    X_ind_cli_sub = X_ind_cli[:, final_cli_idx]
    X_ind_rad_sub = X_ind_rad[:, final_rad_idx]
    y_independent_test = df_independent_test["label"].values
    
    if X_ind_cli_sub.shape[1] == 0 or X_ind_rad_sub.shape[1] == 0:
        print("Error: No features selected for test set")
        return
    
    y_test_prob = final_fusion_model.predict_proba(X_ind_cli_sub, X_ind_rad_sub)
    y_test_pred = final_fusion_model.predict(X_ind_cli_sub, X_ind_rad_sub)
    
    auc_test = roc_auc_score(y_independent_test, y_test_prob)
    auc_test_low, auc_test_high = compute_auc_ci(y_independent_test, y_test_prob)
    acc_test = accuracy_score(y_independent_test, y_test_pred)
    acc_test_low, acc_test_high = compute_accuracy_ci(y_independent_test, y_test_pred)
    f1_test = f1_score(y_independent_test, y_test_pred)
    tn, fp, fn, tp = confusion_matrix(y_independent_test, y_test_pred).ravel()
    sensitivity_test = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity_test = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    
    test_att_weights = final_fusion_model.get_attention_weights(X_ind_cli_sub, X_ind_rad_sub)
    
    test_results = {
        "m_CLI": len(final_cli_idx),
        "m_RAD": len(final_rad_idx),
        "AUC": auc_test,
        "AUC_CI_Low": auc_test_low,
        "AUC_CI_High": auc_test_high,
        "Accuracy": acc_test,
        "Acc_CI_Low": acc_test_low,
        "Acc_CI_High": acc_test_high,
        "F1": f1_test,
        "Sensitivity": sensitivity_test,
        "Specificity": specificity_test,
        "Mean_CLI_Weight": test_att_weights[:, 0].mean() if test_att_weights.size > 0 else 0,
        "Mean_RAD_Weight": test_att_weights[:, 1].mean() if test_att_weights.size > 0 else 0
    }
    
    print("\n=== Independent Test Set Results ===")
    for k, v in test_results.items():
        print(f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}")
    
    df_val_results = pd.DataFrame([val_results])
    df_val_results.to_csv(os.path.join(OUTPUT_DIR, "validation_results.csv"), index=False)
    
    df_test_results = pd.DataFrame([test_results])
    df_test_results.to_csv(os.path.join(OUTPUT_DIR, "test_results.csv"), index=False)
    
    save_path = os.path.join(OUTPUT_DIR, "final_attention_fusion_model.joblib")
    joblib.dump({
        "model": final_fusion_model,
        "cli_preprocessor": cli_preprocessor,
        "rad_preprocessor": rad_preprocessor,
        "cli_feature_indices": final_cli_idx,
        "rad_feature_indices": final_rad_idx,
        "cli_feature_names": [cli_feature_names[i] for i in final_cli_idx],
        "rad_feature_names": [rad_feature_names[i] for i in final_rad_idx],
        "m_cli": len(final_cli_idx),
        "m_rad": len(final_rad_idx),
        "val_metrics": val_results,
        "test_metrics": test_results
    }, save_path)
    
    print(f"\nFinal model saved to: {save_path}")
    print("\nExecution completed successfully!")

if __name__ == "__main__":
    main()