import os
import argparse
import copy
import traceback
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm.auto import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             auc, roc_curve, confusion_matrix, average_precision_score,
                             classification_report, roc_auc_score, precision_recall_curve)
from sklearn.model_selection import StratifiedKFold

import warnings

warnings.filterwarnings('ignore', category=UserWarning)


def seed_everything(seed=2026):
    """Seed supported random generators."""
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


DEFAULT_X_FILE = 'X_feature_matrix_final.csv'
DEFAULT_Y_FILE = 'Y_drug_labels_final.csv'

BASE_OUTPUT_DIR = './tb_results_DL/'

DRUGS = ['AMK', 'BDQ', 'CAP', 'CFZ', 'CS', 'DLM', 'EMB', 'ETO', 'INH', 'KAN', 'LFX', 'LZD', 'MFX', 'OFX', 'PZA', 'PAS',
         'STM', 'RIF']


class PyTorchMLP(nn.Module):
    """Compact multilayer perceptron for binary drug-resistance prediction."""
    def __init__(self, input_dim):
        """Initialize model layers."""
        super(PyTorchMLP, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 32), nn.BatchNorm1d(32), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(32, 16), nn.BatchNorm1d(16), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(16, 1)
        )

    def forward(self, x):
        """Return logits for an input feature batch."""
        return self.net(x)


class CNNGWP(nn.Module):
    """One-dimensional convolutional model over genome-wide binary features."""
    def __init__(self, input_dim, filters=16, kernel_size=10):
        """Initialize convolutional model layers."""
        super(CNNGWP, self).__init__()
        self.conv1d = nn.Conv1d(1, filters, kernel_size, stride=2)
        self.pool = nn.MaxPool1d(2)
        self.flatten = nn.Flatten()
        conv_output_dim = (input_dim - kernel_size) // 2 + 1
        pool_output_dim = conv_output_dim // 2
        self.dropout = nn.Dropout(0.5)
        self.dense = nn.Linear(filters * pool_output_dim, 1)

    def forward(self, x):
        """Return logits for an input feature batch."""
        if x.dim() == 2: x = x.unsqueeze(1)
        x = torch.relu(self.conv1d(x))
        x = self.pool(x)
        x = self.flatten(x)
        x = self.dropout(x)
        return self.dense(x)


class DeepAMR(nn.Module):
    """Autoencoder-style classifier with a supervised resistance head."""
    def __init__(self, input_dim, dropout_prob=0.5):
        """Initialize encoder, classifier, and decoder layers."""
        super(DeepAMR, self).__init__()
        self.encoder = nn.Sequential(
            nn.Dropout(dropout_prob), nn.Linear(input_dim, 128), nn.ReLU(True),
            nn.Linear(128, 64), nn.ReLU(True), nn.Linear(64, 16), nn.ReLU(True)
        )
        self.classifier = nn.Sequential(nn.Linear(16, 4), nn.ReLU(True), nn.Linear(4, 1))
        self.decoder = nn.Sequential(
            nn.Linear(16, 64), nn.ReLU(True), nn.Linear(64, 128), nn.ReLU(True), nn.Linear(128, input_dim)
        )

    def forward(self, x):
        """Return classifier logits and reconstructed inputs."""
        encoded = self.encoder(x)
        task_out = self.classifier(encoded)
        decoded = self.decoder(encoded)
        return task_out, decoded


class WDNN(nn.Module):
    """Wide-and-deep neural network that combines raw and learned features."""
    def __init__(self, input_dim, dropout_prob=0.5):
        """Initialize wide-and-deep model layers."""
        super(WDNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 64)
        self.output = nn.Linear(input_dim + 64, 1)
        self.batch_norm1, self.batch_norm2, self.batch_norm3 = nn.BatchNorm1d(64), nn.BatchNorm1d(64), nn.BatchNorm1d(
            64)
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, x):
        """Return logits for an input feature batch."""
        input_data = x
        x = self.dropout(self.batch_norm1(F.relu(self.fc1(x))))
        x = self.dropout(self.batch_norm2(F.relu(self.fc2(x))))
        x = self.dropout(self.batch_norm3(F.relu(self.fc3(x))))
        x = torch.cat([input_data, x], dim=1)
        return self.output(x)


class FocalLoss(nn.Module):
    """Binary focal loss with optional batch-adaptive class weighting."""
    def __init__(self, alpha=None, gamma=2.0):
        """Initialize focal-loss parameters."""
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, targets):
        """Compute binary focal loss."""
        bce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        targets = targets.type(torch.float32)
        pt = torch.exp(-bce_loss)
        if self.alpha is None:
            pos_ratio = targets.mean().clamp(0.05, 0.95)
            alpha_dynamic = 1.0 - pos_ratio
        else:
            alpha_dynamic = self.alpha
        alpha_t = targets * alpha_dynamic + (1 - targets) * (1 - alpha_dynamic)
        focal_loss = alpha_t * (1 - pt) ** self.gamma * bce_loss
        return focal_loss.mean()


def get_who_layers(row):
    """Return WHO resistance layers for one sample."""
    def is_r(drug):
        """Return 1 when the sample is resistant to the requested drug."""
        val = row.get(drug)
        return 1 if pd.notna(val) and val == 1 else 0

    inh, rif = is_r('INH'), is_r('RIF')
    fq = 1 if any(is_r(d) == 1 for d in ['LFX', 'MFX', 'OFX']) else 0
    sl = 1 if any(is_r(d) == 1 for d in ['BDQ', 'LZD']) else 0

    layers = []
    if rif == 1:
        layers.append('RR-TB')
        if inh == 1:
            layers.append('MDR-TB')
            if fq == 1:
                layers.append('pre-XDR-TB')
                if sl == 1:
                    layers.append('XDR-TB')

    if inh == 1 and rif == 0:
        layers.append('HR-TB')

    first_line = ['INH', 'RIF', 'EMB', 'PZA']
    if all(pd.notna(row.get(d)) for d in first_line) and sum(row.get(d) for d in first_line) == 0:
        layers.append('Pan-Susceptible')

    return layers


def get_permutation_importance(model, model_name, X_test, y_test, device, num_repeats=5):
    """Estimate feature importance from permutation AUROC drops."""
    model.eval()
    X_tensor = torch.FloatTensor(X_test).to(device)
    with torch.no_grad():
        out = model(X_tensor)
        logits = out[0] if model_name == 'DeepAMR' else out
        base_probs = torch.sigmoid(logits).cpu().numpy().flatten()

    if len(np.unique(y_test)) < 2: return np.zeros(X_test.shape[1])
    base_auc = roc_auc_score(y_test, base_probs)

    importances = np.zeros(X_test.shape[1])
    for i in tqdm(range(X_test.shape[1]), desc="Permutation Importance", leave=False):
        feature_aucs = []
        for _ in range(num_repeats):
            X_permuted = X_test.copy()
            np.random.shuffle(X_permuted[:, i])
            X_permuted_tensor = torch.FloatTensor(X_permuted).to(device)
            with torch.no_grad():
                p_out = model(X_permuted_tensor)
                p_logits = p_out[0] if model_name == 'DeepAMR' else p_out
                probs = torch.sigmoid(p_logits).cpu().numpy().flatten()
            feature_aucs.append(roc_auc_score(y_test, probs))
        importances[i] = base_auc - np.mean(feature_aucs)
    return importances


def run_experiment(model_name, x_file=DEFAULT_X_FILE, y_file=DEFAULT_Y_FILE):
    """Run deep-learning evaluation for the selected model."""
    OUTPUT_DIR = os.path.join(BASE_OUTPUT_DIR, model_name)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"[{model_name}] Loading V6 configuration data...")
    X_all = pd.read_csv(x_file).set_index('uniqueid')
    if 'source_dataset' in X_all.columns: X_all = X_all.drop(columns=['source_dataset'])
    Y_all = pd.read_csv(y_file).set_index('uniqueid')
    common_ids = X_all.index.intersection(Y_all.index)
    X_all, Y_all = X_all.loc[common_ids], Y_all.loc[common_ids]

    _has_mps = hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if _has_mps else 'cpu')
    print(f"Compute device: {device} | valid sample count: {len(X_all)}")

    all_drug_summaries = []
    all_who_fold_metrics = []
    y_pred_matrix = pd.DataFrame(index=Y_all.index, columns=DRUGS)
    y_prob_matrix = pd.DataFrame(index=Y_all.index, columns=DRUGS)

    for drug in DRUGS:
        if drug not in Y_all.columns: continue
        try:
            y_drug = Y_all[drug].dropna().astype(int)
            X_drug = X_all.loc[y_drug.index]

            y_counts = y_drug.value_counts().to_dict()
            if len(y_counts) < 2 or y_counts.get(1, 0) < 5: continue

            print(f"\n--- Processing: {drug.ljust(4)} | {model_name} | sample distribution: {y_counts} ---")
            drug_plot_dir = os.path.join(OUTPUT_DIR, 'plots', drug)
            os.makedirs(drug_plot_dir, exist_ok=True)

            skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=2026)
            fold_results = []
            fold_who_metrics = []
            target_layers = ['RR-TB', 'MDR-TB', 'pre-XDR-TB', 'XDR-TB', 'HR-TB', 'Pan-Susceptible']
            total_cm = np.zeros((2, 2), dtype=int)
            X_np, y_np = X_drug.values, y_drug.values

            for fold, (train_val_idx, test_idx) in enumerate(skf.split(X_np, y_np)):
                X_train_val, X_test = X_np[train_val_idx], X_np[test_idx]
                y_train_val, y_test = y_np[train_val_idx], y_np[test_idx]

                inner_skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=2026)
                inner_splits = list(inner_skf.split(X_train_val, y_train_val))
                inner_train_idx, inner_val_idx = inner_splits[0]
                X_train, X_val = X_train_val[inner_train_idx], X_train_val[inner_val_idx]
                y_train, y_val = y_train_val[inner_train_idx], y_train_val[inner_val_idx]

                train_dataset = TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train).view(-1, 1))
                # Drop a final singleton batch to keep BatchNorm stable.
                drop_last = len(train_dataset) > 1 and (len(train_dataset) % 32) <= 1
                train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, drop_last=drop_last)

                if model_name == 'MLP':
                    model = PyTorchMLP(X_np.shape[1]).to(device)
                elif model_name == 'CNNGWP':
                    model = CNNGWP(X_np.shape[1]).to(device)
                elif model_name == 'DeepAMR':
                    model = DeepAMR(X_np.shape[1]).to(device)
                elif model_name == 'WDNN':
                    model = WDNN(X_np.shape[1]).to(device)

                criterion_class = FocalLoss(alpha=None, gamma=2.0).to(device)

                optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-3)
                scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5)

                MAX_EPOCHS = 150
                patience = 15
                best_val_loss = float('inf')
                patience_counter = 0
                best_model_state = None
                best_epoch = 0

                train_losses = []
                val_losses = []

                for epoch in range(MAX_EPOCHS):
                    model.train()
                    epoch_tr_loss = 0
                    for batch_X, batch_y in train_loader:
                        batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                        optimizer.zero_grad()
                        if model_name == 'DeepAMR':
                            task_out, decoded = model(batch_X)
                            loss = criterion_class(task_out, batch_y) + nn.MSELoss()(decoded, batch_X)
                        else:
                            outputs = model(batch_X)
                            loss = criterion_class(outputs, batch_y)
                        loss.backward()
                        optimizer.step()
                        epoch_tr_loss += loss.item()

                    avg_tr_loss = epoch_tr_loss / len(train_loader)
                    train_losses.append(avg_tr_loss)

                    # Use the validation split, not the test fold, for early-stopping decisions.
                    model.eval()
                    with torch.no_grad():
                        X_val_tensor = torch.FloatTensor(X_val).to(device)
                        y_val_tensor = torch.FloatTensor(y_val).view(-1, 1).to(device)
                        if model_name == 'DeepAMR':
                            v_out, dec = model(X_val_tensor)
                            val_loss = criterion_class(v_out, y_val_tensor) + nn.MSELoss()(dec, X_val_tensor)
                        else:
                            v_out = model(X_val_tensor)
                            val_loss = criterion_class(v_out, y_val_tensor)

                        v_loss_val = val_loss.item()
                        val_losses.append(v_loss_val)

                    scheduler.step(v_loss_val)

                    if v_loss_val < best_val_loss:
                        best_val_loss = v_loss_val
                        best_epoch = epoch
                        patience_counter = 0
                        best_model_state = copy.deepcopy(model.state_dict())
                    else:
                        patience_counter += 1

                    if patience_counter >= patience: break

                if best_model_state is not None:
                    model.load_state_dict(best_model_state)

                is_converged = patience_counter >= patience
                actual_epochs = len(train_losses)

                model.eval()
                with torch.no_grad():
                    X_tr_t = torch.FloatTensor(X_train).to(device)
                    logits_tr = model(X_tr_t)[0] if model_name == 'DeepAMR' else model(X_tr_t)
                    y_pred_tr = (torch.sigmoid(logits_tr).cpu().numpy().flatten() >= 0.5).astype(int)
                    train_acc = accuracy_score(y_train, y_pred_tr)

                    X_te_t = torch.FloatTensor(X_test).to(device)
                    logits_te = model(X_te_t)[0] if model_name == 'DeepAMR' else model(X_te_t)
                    probs_te = torch.sigmoid(logits_te).cpu().numpy().flatten()
                    preds_te = (probs_te >= 0.5).astype(int)
                    test_acc = accuracy_score(y_test, preds_te)

                    overfit_gap = train_acc - test_acc

                y_pred_matrix.loc[y_drug.index[test_idx], drug] = preds_te
                y_prob_matrix.loc[y_drug.index[test_idx], drug] = probs_te

                cm = confusion_matrix(y_test, preds_te, labels=[0, 1])
                tn, fp, fn, tp = cm.ravel()
                total_cm += cm

                fold_results.append({
                    'AUROC': roc_auc_score(y_test, probs_te) if len(np.unique(y_test)) > 1 else np.nan,
                    'AUPRC': average_precision_score(y_test, probs_te),
                    'Train_Acc': train_acc,
                    'Test_Acc': test_acc,
                    'Overfit_Gap': overfit_gap,
                    'Best_Epoch': best_epoch + 1,
                    'Converged': is_converged,
                    'Sensitivity/Recall': recall_score(y_test, preds_te, zero_division=0),
                    'Specificity': tn / (tn + fp) if (tn + fp) > 0 else 0,
                    'Accuracy': test_acc,
                    'Precision': precision_score(y_test, preds_te, zero_division=0),
                    'F1-score': f1_score(y_test, preds_te, zero_division=0),
                    'PPV': tp / (tp + fp) if (tp + fp) > 0 else 0,
                    'NPV': tn / (tn + fn) if (tn + fn) > 0 else 0
                })

                if fold == 0:
                    c_status = "OK" if is_converged else "FAILED"
                    print(
                        f"      Fold 0: Focal+ES ({c_status}, Best Ep {best_epoch + 1}) | TrainAcc={train_acc:.3f} | TestAcc={test_acc:.3f} | Gap={overfit_gap:.3f}")

                    plt.figure(figsize=(8, 6))
                    x_axis = range(1, actual_epochs + 1)
                    plt.plot(x_axis, train_losses, label='Train Loss', color='blue', lw=2)
                    plt.plot(x_axis, val_losses, label='Validation Loss', color='orange', lw=2)
                    plt.axvline(x=best_epoch + 1, color='red', linestyle='--', label=f'Best Epoch ({best_epoch + 1})')

                    plt.legend()
                    plt.ylabel('Loss (Focal)', fontweight='bold')
                    plt.xlabel('Epochs', fontweight='bold')
                    plt.title(f'{drug} {model_name} Loss with Early Stopping', fontweight='bold')
                    plt.grid(True, linestyle=':', alpha=0.7)
                    plt.tight_layout()
                    plt.savefig(os.path.join(drug_plot_dir, 'loss_curve.png'), dpi=300)
                    plt.close()

                    importances = get_permutation_importance(model, model_name, X_test, y_test, device)
                    feat_imp = pd.DataFrame({'Feature': X_drug.columns, 'Importance': importances})
                    top_20 = feat_imp.sort_values('Importance', ascending=False).head(20)
                    plt.figure(figsize=(10, 8))
                    sns.barplot(x='Importance', y='Feature', data=top_20, palette='magma', hue='Feature', legend=False)
                    plt.title(f'{drug} {model_name} Feature Importance')
                    plt.tight_layout()
                    plt.savefig(os.path.join(drug_plot_dir, 'importance.png'))
                    plt.close()

                # Release accelerator memory between folds when running multiple models.
                del model, optimizer, scheduler
                if device.type == 'cuda':
                    torch.cuda.empty_cache()

                y_true_fold = Y_all.loc[y_drug.index[test_idx], DRUGS]
                y_pred_fold = y_pred_matrix.loc[y_drug.index[test_idx], DRUGS]

                y_true_who_fold = y_true_fold.apply(get_who_layers, axis=1)
                y_pred_who_fold = y_pred_fold.apply(get_who_layers, axis=1)

                fold_idx_set = y_pred_fold.index
                for layer in target_layers:
                    if layer == 'RR-TB':
                        eligible_mask = Y_all['RIF'].notna()
                    elif layer in ['MDR-TB', 'HR-TB']:
                        eligible_mask = Y_all['INH'].notna() & Y_all['RIF'].notna()
                    elif layer == 'pre-XDR-TB':
                        fq_tested = Y_all[['LFX', 'MFX', 'OFX']].notna().any(axis=1)
                        eligible_mask = Y_all['INH'].notna() & Y_all['RIF'].notna() & fq_tested
                    elif layer == 'XDR-TB':
                        fq_tested = Y_all[['LFX', 'MFX', 'OFX']].notna().any(axis=1)
                        sl_tested = Y_all[['BDQ', 'LZD']].notna().any(axis=1)
                        eligible_mask = Y_all['INH'].notna() & Y_all['RIF'].notna() & fq_tested & sl_tested
                    elif layer == 'Pan-Susceptible':
                        eligible_mask = Y_all[['INH', 'RIF', 'EMB', 'PZA']].notna().all(axis=1)
                    else:
                        continue

                    fold_eligible = eligible_mask.loc[fold_idx_set]
                    y_t = y_true_who_fold.loc[fold_eligible].apply(lambda x: 1 if layer in x else 0)
                    y_p = y_pred_who_fold.loc[fold_eligible].apply(lambda x: 1 if layer in x else 0)

                    if len(y_t) > 0:
                        w_tn, w_fp, w_fn, w_tp = confusion_matrix(y_t, y_p, labels=[0, 1]).ravel()
                        fold_who_metrics.append({
                            'Drug': drug, 'Fold': fold + 1, 'Layer': layer,
                            'Sensitivity/Recall': recall_score(y_t, y_p, zero_division=0),
                            'Specificity': w_tn / (w_tn + w_fp) if (w_tn + w_fp) > 0 else 0,
                            'Accuracy': accuracy_score(y_t, y_p),
                            'Precision': precision_score(y_t, y_p, zero_division=0),
                            'F1-score': f1_score(y_t, y_p, zero_division=0),
                            'PPV': w_tp / (w_tp + w_fp) if (w_tp + w_fp) > 0 else 0,
                            'NPV': w_tn / (w_tn + w_fn) if (w_tn + w_fn) > 0 else 0,
                            'Support(True_Count)': y_t.sum()
                        })

            plt.figure(figsize=(6, 5))
            sns.heatmap(total_cm, annot=True, fmt='d', cmap='Blues', xticklabels=['S (0)', 'R (1)'],
                        yticklabels=['S (0)', 'R (1)'])
            plt.title(f'{drug} Total CM ({model_name})')
            plt.tight_layout()
            plt.savefig(os.path.join(drug_plot_dir, 'confusion_matrix.png'))
            plt.close()

            fold_df = pd.DataFrame(fold_results)
            fold_df.insert(0, 'Fold', range(1, len(fold_df) + 1))
            fold_df.insert(0, 'Drug', drug)
            fold_df.to_csv(os.path.join(OUTPUT_DIR, f'per_fold_{model_name}_{drug}.csv'), index=False)

            if fold_who_metrics:
                fold_who_df = pd.DataFrame(fold_who_metrics)
                fold_who_df.to_csv(os.path.join(OUTPUT_DIR, f'per_fold_WHO_{model_name}_{drug}.csv'), index=False)
                all_who_fold_metrics.extend(fold_who_metrics)

            metric_cols = [c for c in fold_df.columns if c not in ('Drug', 'Fold')]
            drug_mean = fold_df[metric_cols].mean().to_dict()
            drug_std = fold_df[metric_cols].std().to_dict()
            drug_mean.update({f'{k}_std': v for k, v in drug_std.items()})
            drug_mean['Drug'] = drug
            drug_mean['Count_S(0)'] = y_counts.get(0, 0)
            drug_mean['Count_R(1)'] = y_counts.get(1, 0)
            drug_mean['Total_Samples'] = len(y_drug)
            all_drug_summaries.append(drug_mean)

        except Exception as e:
            print(f"\n[ERROR] Drug {drug} failed and will be skipped: {e}")
            traceback.print_exc()
            continue

    summary_df = pd.DataFrame(all_drug_summaries)
    mean_cols = [c for c in summary_df.columns if not c.endswith('_std') and c not in ('Drug', 'Count_S(0)', 'Count_R(1)', 'Total_Samples')]
    std_cols = [c for c in summary_df.columns if c.endswith('_std')]
    agg_row = {}
    for mc in mean_cols:
        agg_row[mc] = summary_df[mc].mean()
        sc = mc + '_std'
        if sc in summary_df.columns:
            agg_row[sc] = summary_df[sc].mean()
    agg_row['Drug'] = 'AGGREGATED_MACRO_AVG'
    summary_df = pd.concat([summary_df, pd.DataFrame([agg_row])], ignore_index=True)
    summary_df.to_csv(os.path.join(OUTPUT_DIR, f'drug_performance_{model_name}.csv'), index=False)

    y_pred_matrix.to_csv(os.path.join(OUTPUT_DIR, f'y_pred_matrix_{model_name}.csv'))
    y_prob_matrix.to_csv(os.path.join(OUTPUT_DIR, f'y_prob_matrix_{model_name}.csv'))

    if all_who_fold_metrics:
        all_who_df = pd.DataFrame(all_who_fold_metrics)
        who_metric_cols = [c for c in all_who_df.columns if c not in ('Drug', 'Fold', 'Layer')]
        who_summary = all_who_df.groupby(['Drug', 'Layer'])[who_metric_cols].agg(['mean', 'std']).reset_index()
        who_summary.columns = ['_'.join(c).rstrip('_') if c[1] else c[0] for c in who_summary.columns]
        who_summary.to_csv(os.path.join(OUTPUT_DIR, f'WHO_layered_metrics_{model_name}.csv'), index=False)

    print(f"\nPlotting combined evaluation curves for {model_name} (Combined ROC & PRC)...")
    plt.rcParams.update({'font.size': 12, 'font.family': 'sans-serif'})

    fig_roc, ax_roc = plt.subplots(figsize=(9, 8))
    fig_prc, ax_prc = plt.subplots(figsize=(9, 8))
    colors = plt.cm.tab20(np.linspace(0, 1, len(DRUGS)))

    for idx, drug in enumerate(DRUGS):
        if drug not in y_prob_matrix.columns or y_prob_matrix[drug].isna().all(): continue
        valid_idx = y_prob_matrix[drug].dropna().index
        y_true_plot = Y_all.loc[valid_idx, drug].astype(int)
        y_prob_plot = y_prob_matrix.loc[valid_idx, drug].astype(float)

        if len(y_true_plot.unique()) < 2: continue

        fpr, tpr, _ = roc_curve(y_true_plot, y_prob_plot)
        roc_auc = auc(fpr, tpr)
        ax_roc.plot(fpr, tpr, color=colors[idx], lw=2, label=f'{drug} (AUC = {roc_auc:.3f})')

        precision_curve, recall_curve, _ = precision_recall_curve(y_true_plot, y_prob_plot)
        prc_auc = average_precision_score(y_true_plot, y_prob_plot)
        ax_prc.plot(recall_curve, precision_curve, color=colors[idx], lw=2, label=f'{drug} (AUPRC = {prc_auc:.3f})')

    ax_roc.plot([0, 1], [0, 1], color='gray', lw=1.5, linestyle='--')
    ax_roc.set_xlim([0.0, 1.0]);
    ax_roc.set_ylim([0.0, 1.05])
    ax_roc.set_xlabel('False Positive Rate', fontweight='bold');
    ax_roc.set_ylabel('True Positive Rate', fontweight='bold')
    ax_roc.set_title(f'Combined ROC Curves ({model_name})', fontweight='bold')
    ax_roc.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), borderaxespad=0.)
    ax_roc.grid(True, linestyle=':', alpha=0.7)
    fig_roc.tight_layout()
    fig_roc.savefig(os.path.join(OUTPUT_DIR, f'Combined_ROC_Curves_{model_name}.png'), dpi=300, bbox_inches='tight')
    plt.close(fig_roc)

    ax_prc.set_xlim([0.0, 1.0]);
    ax_prc.set_ylim([0.0, 1.05])
    ax_prc.set_xlabel('Recall', fontweight='bold');
    ax_prc.set_ylabel('Precision', fontweight='bold')
    ax_prc.set_title(f'Combined PR Curves ({model_name})', fontweight='bold')
    ax_prc.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), borderaxespad=0.)
    ax_prc.grid(True, linestyle=':', alpha=0.7)
    fig_prc.tight_layout()
    fig_prc.savefig(os.path.join(OUTPUT_DIR, f'Combined_PRC_Curves_{model_name}.png'), dpi=300, bbox_inches='tight')
    plt.close(fig_prc)

    print(f"\n{model_name} experiment complete. Results saved in: {OUTPUT_DIR}")


if __name__ == "__main__":
    seed_everything(seed=2026)

    parser = argparse.ArgumentParser(description="Run Deep Learning TB Resistance Models")
    parser.add_argument('--model', type=str, default='ALL',
                        choices=['MLP', 'CNNGWP', 'DeepAMR', 'WDNN', 'ALL'],
                        help="Choose the model to run: MLP, CNNGWP, DeepAMR, WDNN, or ALL")
    parser.add_argument('--x-file', type=str, default=DEFAULT_X_FILE,
                        help="Path to the X feature matrix CSV")
    parser.add_argument('--y-file', type=str, default=DEFAULT_Y_FILE,
                        help="Path to the Y drug labels CSV")
    args = parser.parse_args()

    os.makedirs(BASE_OUTPUT_DIR, exist_ok=True)

    if args.model == 'ALL':
        models_to_run = ['MLP', 'CNNGWP', 'DeepAMR', 'WDNN']
        print("Starting automatic run mode: MLP, CNNGWP, DeepAMR, and WDNN will run sequentially...")
        for m in models_to_run:
            print(f"\n{'=' * 50}\nStarting model: {m}\n{'=' * 50}")
            run_experiment(m, x_file=args.x_file, y_file=args.y_file)
        print(f"\nAll models complete. Results archived in: {BASE_OUTPUT_DIR}")
    else:
        run_experiment(args.model, x_file=args.x_file, y_file=args.y_file)
