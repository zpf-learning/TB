import pandas as pd
import numpy as np
import os

X_MATRIX_FILE = 'X_feature_matrix_final.csv'
PHENOTYPE_FILE = 'phenotype.csv'
OUTPUT_Y_FILE = 'Y_drug_labels_final.csv'

TARGET_DRUGS = ['AMK', 'BDQ', 'CAP', 'CFZ', 'CS', 'DLM', 'EMB', 'ETO', 'INH', 'KAN', 'LFX', 'LZD', 'MFX', 'OFX', 'PZA', 'PAS', 'STM', 'RIF']


def generate_y_matrix():
    """Build the label matrix aligned to the feature matrix."""
    print("--- 1. Data loading and alignment ---")
    if not os.path.exists(X_MATRIX_FILE) or not os.path.exists(PHENOTYPE_FILE):
        print("Error: X matrix or phenotype file not found. Check the paths.")
        return

    df_x = pd.read_csv(X_MATRIX_FILE, usecols=['uniqueid'])
    ordered_ids = df_x['uniqueid'].astype(str).tolist()

    df_pheno = pd.read_csv(PHENOTYPE_FILE, low_memory=False)

    df_pheno.columns = df_pheno.columns.str.strip()

    required_cols = ['sra_accession', 'drug_abbr', 'binary_result']
    for col in required_cols:
        if col not in df_pheno.columns:
            print(f"Error: phenotype file does not contain column '{col}'. Current columns: {df_pheno.columns.tolist()}")
            return

    df_pheno['sra_accession'] = df_pheno['sra_accession'].astype(str).str.strip()
    df_pheno['drug_abbr'] = df_pheno['drug_abbr'].astype(str).str.strip()

    print(f"--- 2. Drug filtering and binarization (target drugs: {len(TARGET_DRUGS)}) ---")

    df_filtered = df_pheno[df_pheno['drug_abbr'].isin(TARGET_DRUGS)].copy()

    df_filtered['status'] = (
        df_filtered['binary_result']
        .astype(str)
        .str.strip()
        .map({'S': 0, 'R': 1})
    )

    print("--- 3. Matrix pivoting from long to wide format ---")

    y_matrix = df_filtered.pivot_table(
        index='sra_accession',
        columns='drug_abbr',
        values='status'
    )

    y_matrix = y_matrix.reindex(ordered_ids)

    y_matrix.index.name = 'uniqueid'
    y_matrix = y_matrix.reset_index()

    print("\n--- 4. Result summary table ---")
    stats_data = []

    for drug in TARGET_DRUGS:
        if drug in y_matrix.columns:
            counts = y_matrix[drug].value_counts(dropna=False)
            r = counts.get(1.0, 0)
            s = counts.get(0.0, 0)
            nan = y_matrix[drug].isna().sum()

            stats_data.append({
                'Drug': drug,
                'Resistant R (1)': int(r),
                'Susceptible S (0)': int(s),
                'Missing (NaN)': int(nan)
            })
        else:
            stats_data.append({
                'Drug': drug,
                'Resistant R (1)': 0,
                'Susceptible S (0)': 0,
                'Missing (NaN)': len(y_matrix)
            })

    stats_df = pd.DataFrame(stats_data)

    print("-" * 55)
    print(stats_df.to_string(index=False))
    print("-" * 55)

    y_matrix.to_csv(OUTPUT_Y_FILE, index=False, encoding='utf-8-sig')
    print(f"\nFinal Y label matrix saved to: {OUTPUT_Y_FILE}")
    print(f"Final shape: {y_matrix.shape} (row count is fully aligned with the X matrix)")


if __name__ == "__main__":
    generate_y_matrix()
