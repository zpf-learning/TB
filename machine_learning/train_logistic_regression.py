import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             auc, roc_curve, confusion_matrix, average_precision_score,
                             classification_report, precision_recall_curve)
from sklearn.model_selection import StratifiedKFold

warnings.filterwarnings('ignore', category=UserWarning)

X_FILE = 'X_feature_matrix_final.csv'
Y_FILE = 'Y_drug_labels_final.csv'
OUTPUT_DIR = './tb_results_LR/'

DRUGS = ['AMK', 'BDQ', 'CAP', 'CFZ', 'CS', 'DLM', 'EMB', 'ETO', 'INH', 'KAN', 'LFX', 'LZD', 'MFX', 'OFX', 'PZA', 'PAS',
         'STM', 'RIF']


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


def run_experiment_final():
    """Run logistic-regression evaluation across target drugs."""
    print("Loading data...")
    X_raw = pd.read_csv(X_FILE).set_index('uniqueid')
    if 'source_dataset' in X_raw.columns:
        X_raw = X_raw.drop(columns=['source_dataset'])
    Y_raw = pd.read_csv(Y_FILE).set_index('uniqueid')

    common_ids = X_raw.index.intersection(Y_raw.index)
    X, Y = X_raw.loc[common_ids], Y_raw.loc[common_ids]

    print(f"Valid total sample count: {len(X)}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    all_drug_summaries = []
    y_pred_matrix = pd.DataFrame(index=Y.index, columns=DRUGS)
    y_prob_matrix = pd.DataFrame(index=Y.index, columns=DRUGS)

    for drug in DRUGS:
        if drug not in Y.columns: continue
        y_drug = Y[drug].dropna().astype(int)
        X_drug = X.loc[y_drug.index]

        counts = y_drug.value_counts()
        if len(counts) < 2 or counts.min() < 3:
            print(f"--- Skipping drug: {drug.ljust(4)} | insufficient sample size or single class ---")
            continue

        print(f"--- Processing: {drug.ljust(4)} (LR) | sample size (N={len(y_drug)}, R={counts.get(1, 0)}) ---")

        drug_plot_dir = os.path.join(OUTPUT_DIR, 'plots', drug)
        os.makedirs(drug_plot_dir, exist_ok=True)

        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=2026)
        fold_results = []
        target_layers = ['RR-TB', 'MDR-TB', 'pre-XDR-TB', 'XDR-TB', 'HR-TB', 'Pan-Susceptible']
        all_who_fold_metrics = []
        total_cm = np.zeros((2, 2), dtype=int)

        for fold, (train_idx, test_idx) in enumerate(skf.split(X_drug, y_drug)):
            X_train, X_test = X_drug.iloc[train_idx], X_drug.iloc[test_idx]
            y_train, y_test = y_drug.iloc[train_idx], y_drug.iloc[test_idx]

            model = LogisticRegression(C=0.9, solver='liblinear', class_weight='balanced', max_iter=1000)
            model.fit(X_train, y_train)

            iters_done = model.n_iter_[0]
            is_converged = iters_done < model.max_iter

            y_prob_train = model.predict_proba(X_train)[:, 1]
            y_pred_train = (y_prob_train >= 0.5).astype(int)

            y_prob = model.predict_proba(X_test)[:, 1]
            y_pred = (y_prob >= 0.5).astype(int)

            train_acc = accuracy_score(y_train, y_pred_train)
            test_acc = accuracy_score(y_test, y_pred)
            overfit_gap = train_acc - test_acc

            y_pred_matrix.loc[y_test.index, drug] = y_pred
            y_prob_matrix.loc[y_test.index, drug] = y_prob

            tn, fp, fn, tp = confusion_matrix(y_test, y_pred, labels=[0, 1]).ravel()
            total_cm += confusion_matrix(y_test, y_pred, labels=[0, 1])

            fold_results.append({
                'AUROC': auc(roc_curve(y_test, y_prob)[0], roc_curve(y_test, y_prob)[1]) if len(
                    np.unique(y_test)) > 1 else np.nan,
                'AUPRC': average_precision_score(y_test, y_prob),
                'Train_Acc': train_acc,
                'Test_Acc': test_acc,
                'Overfit_Gap': overfit_gap,
                'Sensitivity/Recall': recall_score(y_test, y_pred, zero_division=0),
                'Specificity': tn / (tn + fp) if (tn + fp) > 0 else 0,
                'Accuracy': test_acc,
                'Precision': precision_score(y_test, y_pred, zero_division=0),
                'F1-score': f1_score(y_test, y_pred, zero_division=0),
                'PPV': tp / (tp + fp) if (tp + fp) > 0 else 0,
                'NPV': tn / (tn + fn) if (tn + fn) > 0 else 0
            })

            if fold == 0:
                conv_status = "OK" if is_converged else "FAILED"
                print(
                    f"    Fold 0: Conv={conv_status}({iters_done} iters) | TrainAcc={train_acc:.3f} | TestAcc={test_acc:.3f} | Gap={overfit_gap:.3f}")

            if fold == 0:
                feat_imp = pd.DataFrame({'Feature': X_drug.columns, 'Coef': model.coef_[0]})
                feat_imp['AbsCoef'] = feat_imp['Coef'].abs()
                top_20 = feat_imp.sort_values('AbsCoef', ascending=False).head(20)
                plt.figure(figsize=(10, 8))
                colors = ['red' if x > 0 else 'blue' for x in top_20['Coef']]
                sns.barplot(x='Coef', y='Feature', data=top_20, palette=colors, hue='Feature', legend=False)
                plt.title(f'{drug} Feature Importance (LR)')
                plt.tight_layout()
                plt.savefig(os.path.join(drug_plot_dir, 'importance.png'))
                plt.close()

            y_true_fold = Y.loc[y_test.index, DRUGS]
            y_pred_fold = y_pred_matrix.loc[y_test.index, DRUGS]

            y_true_who_fold = y_true_fold.apply(get_who_layers, axis=1)
            y_pred_who_fold = y_pred_fold.apply(get_who_layers, axis=1)

            fold_idx_set = y_pred_fold.index
            for layer in target_layers:
                if layer == 'RR-TB':
                    eligible_mask = Y['RIF'].notna()
                elif layer in ['MDR-TB', 'HR-TB']:
                    eligible_mask = Y['INH'].notna() & Y['RIF'].notna()
                elif layer == 'pre-XDR-TB':
                    fq_tested = Y[['LFX', 'MFX', 'OFX']].notna().any(axis=1)
                    eligible_mask = Y['INH'].notna() & Y['RIF'].notna() & fq_tested
                elif layer == 'XDR-TB':
                    fq_tested = Y[['LFX', 'MFX', 'OFX']].notna().any(axis=1)
                    sl_tested = Y[['BDQ', 'LZD']].notna().any(axis=1)
                    eligible_mask = Y['INH'].notna() & Y['RIF'].notna() & fq_tested & sl_tested
                elif layer == 'Pan-Susceptible':
                    eligible_mask = Y[['INH', 'RIF', 'EMB', 'PZA']].notna().all(axis=1)
                else:
                    continue

                fold_eligible = eligible_mask.loc[fold_idx_set]
                y_t = y_true_who_fold.loc[fold_eligible].apply(lambda x: 1 if layer in x else 0)
                y_p = y_pred_who_fold.loc[fold_eligible].apply(lambda x: 1 if layer in x else 0)

                if len(y_t) > 0:
                    w_tn, w_fp, w_fn, w_tp = confusion_matrix(y_t, y_p, labels=[0, 1]).ravel()
                    all_who_fold_metrics.append({
                        'Fold': fold + 1, 'Layer': layer,
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
        sns.heatmap(total_cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['S (0)', 'R (1)'], yticklabels=['S (0)', 'R (1)'])
        plt.title(f'{drug} Total CM')
        plt.savefig(os.path.join(drug_plot_dir, 'confusion_matrix.png'))
        plt.close()

        fold_df = pd.DataFrame(fold_results)
        fold_df.insert(0, 'Fold', range(1, len(fold_df) + 1))
        fold_df.insert(0, 'Drug', drug)
        fold_df.to_csv(os.path.join(OUTPUT_DIR, f'per_fold_LR_{drug}.csv'), index=False)

        if all_who_fold_metrics:
            fold_who_df = pd.DataFrame(all_who_fold_metrics)
            fold_who_df.insert(0, 'Drug', drug)
            fold_who_df.to_csv(os.path.join(OUTPUT_DIR, f'per_fold_WHO_LR_{drug}.csv'), index=False)

        metric_cols = [c for c in fold_df.columns if c not in ('Drug', 'Fold')]
        drug_avg = fold_df[metric_cols].mean().to_dict()
        drug_std = fold_df[metric_cols].std().to_dict()
        drug_avg.update({f'{k}_std': v for k, v in drug_std.items()})
        drug_avg['Drug'] = drug
        drug_avg['Count_S(0)'] = counts.get(0, 0)
        drug_avg['Count_R(1)'] = counts.get(1, 0)
        drug_avg['Total_Samples'] = len(y_drug)
        all_drug_summaries.append(drug_avg)

    summary_df = pd.DataFrame(all_drug_summaries)
    mean_cols = [c for c in summary_df.columns if not c.endswith('_std') and c not in ('Drug', 'Count_S(0)', 'Count_R(1)', 'Total_Samples')]
    agg_row = {}
    for mc in mean_cols:
        agg_row[mc] = summary_df[mc].mean()
        sc = mc + '_std'
        if sc in summary_df.columns:
            agg_row[sc] = summary_df[sc].mean()
    agg_row['Drug'] = 'AGGREGATED_MACRO_AVG'
    summary_df = pd.concat([summary_df, pd.DataFrame([agg_row])], ignore_index=True)
    summary_df.to_csv(os.path.join(OUTPUT_DIR, 'drug_performance_summary_LR.csv'), index=False)

    y_pred_matrix.to_csv(os.path.join(OUTPUT_DIR, 'y_pred_matrix_LR.csv'))
    y_prob_matrix.to_csv(os.path.join(OUTPUT_DIR, 'y_prob_matrix_LR.csv'))
    print(">>> Predicted probability and class matrices have been saved locally.")

    print("\nPlotting combined evaluation curves (Combined ROC & PRC)...")
    plt.rcParams.update({'font.size': 12, 'font.family': 'sans-serif'})

    fig_roc, ax_roc = plt.subplots(figsize=(9, 8))
    fig_prc, ax_prc = plt.subplots(figsize=(9, 8))
    colors = plt.cm.tab20(np.linspace(0, 1, len(DRUGS)))

    for idx, drug in enumerate(DRUGS):
        if drug not in y_prob_matrix.columns or y_prob_matrix[drug].isna().all():
            continue
        valid_idx = y_prob_matrix[drug].dropna().index
        y_true_plot = Y.loc[valid_idx, drug].astype(int)
        y_prob_plot = y_prob_matrix.loc[valid_idx, drug].astype(float)
        if len(y_true_plot.unique()) < 2:
            continue
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
    ax_roc.set_title('Combined ROC Curves Across Drugs (LR)', fontweight='bold')
    ax_roc.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), borderaxespad=0.)
    ax_roc.grid(True, linestyle=':', alpha=0.7)
    fig_roc.tight_layout()
    fig_roc.savefig(os.path.join(OUTPUT_DIR, 'Combined_ROC_Curves_LR.png'), dpi=300, bbox_inches='tight')
    plt.close(fig_roc)

    ax_prc.set_xlim([0.0, 1.0]);
    ax_prc.set_ylim([0.0, 1.05])
    ax_prc.set_xlabel('Recall', fontweight='bold');
    ax_prc.set_ylabel('Precision', fontweight='bold')
    ax_prc.set_title('Combined Precision-Recall Curves Across Drugs (LR)', fontweight='bold')
    ax_prc.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), borderaxespad=0.)
    ax_prc.grid(True, linestyle=':', alpha=0.7)
    fig_prc.tight_layout()
    fig_prc.savefig(os.path.join(OUTPUT_DIR, 'Combined_PRC_Curves_LR.png'), dpi=300, bbox_inches='tight')
    plt.close(fig_prc)

    if all_who_fold_metrics:
        all_who_df = pd.DataFrame(all_who_fold_metrics)
        who_metric_cols = [c for c in all_who_df.columns if c not in ('Drug', 'Fold', 'Layer')]
        who_summary = all_who_df.groupby(['Drug', 'Layer'])[who_metric_cols].agg(['mean', 'std']).reset_index()
        who_summary.columns = ['_'.join(c).rstrip('_') if c[1] else c[0] for c in who_summary.columns]
        who_summary.to_csv(os.path.join(OUTPUT_DIR, 'WHO_layered_metrics_LR.csv'), index=False)

    print(f"\nLR experiment complete. Results saved in: {OUTPUT_DIR}")


if __name__ == "__main__":
    run_experiment_final()
