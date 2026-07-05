import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             auc, roc_curve, confusion_matrix, average_precision_score,
                             classification_report, precision_recall_curve)
from sklearn.model_selection import StratifiedKFold

warnings.filterwarnings('ignore', category=UserWarning)

X_FILE = 'X_feature_matrix_final_with_lineage_final.csv'
Y_FILE = 'Y_drug_labels_final_with_lineage.csv'
BASE_OUTPUT_DIR = './tb_results_RF_lineage/'

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

    # Pan-susceptible requires all four first-line drugs to be tested and susceptible.
    first_line = ['INH', 'RIF', 'EMB', 'PZA']
    if all(pd.notna(row.get(d)) for d in first_line) and sum(row.get(d) for d in first_line) == 0:
        layers.append('Pan-Susceptible')

    return layers


def run_rf_experiment_by_lineage():
    """Run lineage-stratified random-forest evaluation."""
    print("Loading data...")
    X_raw = pd.read_csv(X_FILE).set_index('uniqueid')
    Y_raw = pd.read_csv(Y_FILE).set_index('uniqueid')

    if 'lineage' not in X_raw.columns:
        print("Error: 'lineage' column not found in the X file")
        return

    X_with_lin = X_raw[X_raw['lineage'].notna()].copy()
    all_lineages = X_with_lin['lineage'].unique()

    common_ids = X_with_lin.index.intersection(Y_raw.index)
    X_main, Y_main = X_with_lin.loc[common_ids], Y_raw.loc[common_ids]

    print(f"Valid total sample count: {len(X_main)} | lineage list: {list(all_lineages)}")

    for lin in all_lineages:
        print(f"\n" + "=" * 60)
        print(f">>> Processing lineage (RF): {lin}")
        print("=" * 60)

        lin_mask = X_main['lineage'] == lin
        X_lin = X_main[lin_mask].drop(columns=['lineage', 'source_dataset'], errors='ignore')
        Y_lin = Y_main[lin_mask]

        if len(X_lin) < 10:
            print(f"  [!] Lineage {lin} has insufficient samples ({len(X_lin)}); skipping.")
            continue

        lin_output_dir = os.path.join(BASE_OUTPUT_DIR, str(lin))
        os.makedirs(lin_output_dir, exist_ok=True)

        all_drug_summaries = []
        y_pred_matrix = pd.DataFrame(index=Y_lin.index, columns=DRUGS)
        y_prob_matrix = pd.DataFrame(index=Y_lin.index, columns=DRUGS)
        all_who_fold_metrics_collector = []

        for drug in DRUGS:
            if drug not in Y_lin.columns: continue

            y_drug = Y_lin[drug].dropna().astype(int)
            X_drug = X_lin.loc[y_drug.index]

            counts = y_drug.value_counts()
            if len(counts) < 2 or counts.min() < 3:
                reason = "single class" if len(counts) < 2 else f"too few positive samples ({counts.min()})"
                print(f"  [-] {drug.ljust(4)} : skipped ({reason})")
                continue

            print(f"  [+] {drug.ljust(4)} : training RF (N={len(y_drug)}, R={counts.get(1, 0)})")

            n_folds = min(5, counts.min())
            skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=2026)
            fold_results = []
            target_layers = ['RR-TB', 'MDR-TB', 'pre-XDR-TB', 'XDR-TB', 'HR-TB', 'Pan-Susceptible']
            all_who_fold_metrics = []
            total_cm = np.zeros((2, 2), dtype=int)

            drug_plot_dir = os.path.join(lin_output_dir, 'plots', drug)
            os.makedirs(drug_plot_dir, exist_ok=True)

            for fold, (train_idx, test_idx) in enumerate(skf.split(X_drug, y_drug)):
                X_train, X_test = X_drug.iloc[train_idx], X_drug.iloc[test_idx]
                y_train, y_test = y_drug.iloc[train_idx], y_drug.iloc[test_idx]

                model = RandomForestClassifier(
                    n_estimators=100,
                    class_weight='balanced',
                    random_state=42,
                    n_jobs=-1
                )
                model.fit(X_train, y_train)

                iters_done = model.n_estimators
                is_converged = True

                y_prob_train = model.predict_proba(X_train)[:, 1]
                y_pred_train = (y_prob_train >= 0.5).astype(int)
                train_acc = accuracy_score(y_train, y_pred_train)

                y_prob = model.predict_proba(X_test)[:, 1]
                y_pred = (y_prob >= 0.5).astype(int)
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
                    print(
                        f"      Fold 0: Conv=OK({iters_done} trees) | TrainAcc={train_acc:.3f} | TestAcc={test_acc:.3f} | Gap={overfit_gap:.3f}")

                    feat_imp = pd.DataFrame({'Feature': X_drug.columns, 'Importance': model.feature_importances_})
                    top_20 = feat_imp.sort_values('Importance', ascending=False).head(20)
                    plt.figure(figsize=(10, 8))
                    sns.barplot(x='Importance', y='Feature', data=top_20, hue='Feature', palette='viridis',
                                legend=False)
                    plt.title(f'{drug} RF Feature Importance ({lin})')
                    plt.tight_layout()
                    plt.savefig(os.path.join(drug_plot_dir, 'importance.png'))
                    plt.close()

                y_true_fold = Y_lin.loc[y_test.index, DRUGS]
                y_pred_fold = y_pred_matrix.loc[y_test.index, DRUGS]

                y_true_who_fold = y_true_fold.apply(get_who_layers, axis=1)
                y_pred_who_fold = y_pred_fold.apply(get_who_layers, axis=1)

                fold_idx_set = y_pred_fold.index
                for layer in target_layers:
                    if layer == 'RR-TB':
                        eligible_mask = Y_lin['RIF'].notna()
                    elif layer in ['MDR-TB', 'HR-TB']:
                        eligible_mask = Y_lin['INH'].notna() & Y_lin['RIF'].notna()
                    elif layer == 'pre-XDR-TB':
                        fq_tested = Y_lin[['LFX', 'MFX', 'OFX']].notna().any(axis=1)
                        eligible_mask = Y_lin['INH'].notna() & Y_lin['RIF'].notna() & fq_tested
                    elif layer == 'XDR-TB':
                        fq_tested = Y_lin[['LFX', 'MFX', 'OFX']].notna().any(axis=1)
                        sl_tested = Y_lin[['BDQ', 'LZD']].notna().any(axis=1)
                        eligible_mask = Y_lin['INH'].notna() & Y_lin['RIF'].notna() & fq_tested & sl_tested
                    elif layer == 'Pan-Susceptible':
                        eligible_mask = Y_lin[['INH', 'RIF', 'EMB', 'PZA']].notna().all(axis=1)
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
            sns.heatmap(total_cm, annot=True, fmt='d', cmap='Blues', xticklabels=['S (0)', 'R (1)'],
                        yticklabels=['S (0)', 'R (1)'])
            plt.title(f'{drug} Total CM (RF - {lin})')
            plt.tight_layout()
            plt.savefig(os.path.join(drug_plot_dir, 'confusion_matrix.png'))
            plt.close()

            if fold_results:
                fold_df = pd.DataFrame(fold_results)
                fold_df.insert(0, 'Fold', range(1, len(fold_df) + 1))
                fold_df.insert(0, 'Drug', drug)
                fold_df.to_csv(os.path.join(lin_output_dir, f'per_fold_RF_{lin}_{drug}.csv'), index=False)

                metric_cols = [c for c in fold_df.columns if c not in ('Drug', 'Fold')]
                drug_avg = fold_df[metric_cols].mean().to_dict()
                drug_std = fold_df[metric_cols].std().to_dict()
                drug_avg.update({f'{k}_std': v for k, v in drug_std.items()})
                drug_avg['Drug'] = drug
                drug_avg['Count_S(0)'] = counts.get(0, 0)
                drug_avg['Count_R(1)'] = counts.get(1, 0)
                drug_avg['Total_Samples'] = len(y_drug)
                all_drug_summaries.append(drug_avg)

                if all_who_fold_metrics:
                    fold_who_df = pd.DataFrame(all_who_fold_metrics)
                    fold_who_df.insert(0, 'Drug', drug)
                    fold_who_df.to_csv(os.path.join(lin_output_dir, f'per_fold_WHO_RF_{lin}_{drug}.csv'), index=False)
                    all_who_fold_metrics_collector.extend(all_who_fold_metrics)

        if all_drug_summaries:
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
            summary_df.to_csv(os.path.join(lin_output_dir, f'RF_performance_{lin}.csv'), index=False)

            y_pred_matrix.to_csv(os.path.join(lin_output_dir, f'y_pred_matrix_RF_{lin}.csv'))
            y_prob_matrix.to_csv(os.path.join(lin_output_dir, f'y_prob_matrix_RF_{lin}.csv'))

            print(f"  Plotting combined evaluation curves for lineage {lin}...")
            plt.rcParams.update({'font.size': 12, 'font.family': 'sans-serif'})
            fig_roc, ax_roc = plt.subplots(figsize=(9, 8))
            fig_prc, ax_prc = plt.subplots(figsize=(9, 8))
            colors = plt.cm.tab20(np.linspace(0, 1, len(DRUGS)))

            has_curves = False
            for idx, drug in enumerate(DRUGS):
                if drug not in y_prob_matrix.columns or y_prob_matrix[drug].isna().all(): continue
                valid_idx = y_prob_matrix[drug].dropna().index
                y_true_plot = Y_lin.loc[valid_idx, drug].astype(int)
                y_prob_plot = y_prob_matrix.loc[valid_idx, drug].astype(float)

                if len(y_true_plot.unique()) < 2: continue
                has_curves = True

                fpr, tpr, _ = roc_curve(y_true_plot, y_prob_plot)
                roc_auc = auc(fpr, tpr)
                ax_roc.plot(fpr, tpr, color=colors[idx], lw=2, label=f'{drug} (AUC = {roc_auc:.3f})')

                precision_curve, recall_curve, _ = precision_recall_curve(y_true_plot, y_prob_plot)
                prc_auc = average_precision_score(y_true_plot, y_prob_plot)
                ax_prc.plot(recall_curve, precision_curve, color=colors[idx], lw=2,
                            label=f'{drug} (AUPRC = {prc_auc:.3f})')

            if has_curves:
                ax_roc.plot([0, 1], [0, 1], color='gray', lw=1.5, linestyle='--')
                ax_roc.set_xlim([0.0, 1.0]);
                ax_roc.set_ylim([0.0, 1.05])
                ax_roc.set_xlabel('False Positive Rate', fontweight='bold');
                ax_roc.set_ylabel('True Positive Rate', fontweight='bold')
                ax_roc.set_title(f'Combined ROC Curves (RF - {lin})', fontweight='bold')
                ax_roc.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), borderaxespad=0.)
                ax_roc.grid(True, linestyle=':', alpha=0.7)
                fig_roc.tight_layout()
                fig_roc.savefig(os.path.join(lin_output_dir, f'Combined_ROC_Curves_RF_{lin}.png'), dpi=300,
                                bbox_inches='tight')

                ax_prc.set_xlim([0.0, 1.0]);
                ax_prc.set_ylim([0.0, 1.05])
                ax_prc.set_xlabel('Recall', fontweight='bold');
                ax_prc.set_ylabel('Precision', fontweight='bold')
                ax_prc.set_title(f'Combined PR Curves (RF - {lin})', fontweight='bold')
                ax_prc.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), borderaxespad=0.)
                ax_prc.grid(True, linestyle=':', alpha=0.7)
                fig_prc.tight_layout()
                fig_prc.savefig(os.path.join(lin_output_dir, f'Combined_PRC_Curves_RF_{lin}.png'), dpi=300,
                                bbox_inches='tight')
            plt.close(fig_roc);
            plt.close(fig_prc)

            if all_who_fold_metrics_collector:
                all_who_df = pd.DataFrame(all_who_fold_metrics_collector)
                who_metric_cols = [c for c in all_who_df.columns if c not in ('Drug', 'Fold', 'Layer')]
                who_summary = all_who_df.groupby(['Drug', 'Layer'])[who_metric_cols].agg(['mean', 'std']).reset_index()
                who_summary.columns = ['_'.join(c).rstrip('_') if c[1] else c[0] for c in who_summary.columns]
                who_summary.to_csv(os.path.join(lin_output_dir, f'WHO_layered_metrics_RF_{lin}.csv'), index=False)

        print(f"--- [Complete] Lineage {lin} RF evaluation ---")

    print(f"\nAll lineage evaluations complete. Results saved in: {BASE_OUTPUT_DIR}")


if __name__ == "__main__":
    run_rf_experiment_by_lineage()
