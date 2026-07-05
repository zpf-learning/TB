import pandas as pd
import math
import os
import numpy as np
import re


FILE_CONFIG = {
    'literature': 'literature_mutation_all.csv',
    'china_dataset': 'china_mutation_all.csv',
    'farhat_dataset': 'farhat_mutation_all.csv',
    'cryptic_dataset': 'cryptic_mutations_all.csv',
    'extra_samples': 'extra_all_mutations.csv',
}

INPUT_TARGET_GENE_FILE = 'gene_drug_tier1(original).csv'

PHENOTYPE_FILE = 'phenotype.csv'

OUTPUT_FEATURE_FILE = 'X_feature_matrix_final.csv'
OUTPUT_LABEL_FILE = 'Y_drug_labels_final.csv'

BASE_RATIO = 30 / 10000



def load_and_merge_files(file_config_dict):
    """Load and merge normalized mutation source files."""
    df_list = []
    print(f"--- 1. Preparing to merge {len(file_config_dict)} data sources ---")

    for custom_name, fpath in file_config_dict.items():
        if os.path.exists(fpath):
            try:
                df = pd.read_csv(fpath)

                df.columns = [c.lower() for c in df.columns]
                if 'uniqueid' not in df.columns and '\ufeffuniqueid' in df.columns:
                    df.rename(columns={'\ufeffuniqueid': 'uniqueid'}, inplace=True)

                if 'uniqueid' in df.columns:
                    original_len = len(df)
                    df['uniqueid'] = df['uniqueid'].replace(r'^\s*$', np.nan, regex=True)
                    df.dropna(subset=['uniqueid'], inplace=True)
                    dropped_count = original_len - len(df)
                    if dropped_count > 0:
                        print(f"    [Info] {custom_name}: removed {dropped_count} rows with missing uniqueid values")
                else:
                    print(f"  -> [{custom_name}] Critical error: uniqueid column not found; skipping this file.")
                    continue

                df['uniqueid'] = df['uniqueid'].astype(str).str.strip()
                df['uniqueid'] = df['uniqueid'].str.replace('_clockwork', '', regex=False)

                source_label = 'cryptic_dataset' if custom_name == 'extra_samples' else custom_name
                df['source_dataset'] = source_label

                if custom_name == 'cryptic_dataset':
                    target_col = 'mutations' if 'mutations' in df.columns else 'mutation'

                    if target_col in df.columns:
                        df[target_col] = df[target_col].astype(str).str.split(',')
                        df = df.explode(target_col)
                        df[target_col] = df[target_col].str.strip()

                        # Extract gene and mutation components with a whitelist-aware pattern.
                        # Allow PE_PGRS genes, numeric kd_ag antigens, and standard gene names.
                        split_data = df[target_col].str.extract(r'^(PE_PGRS\d+|\d+kd_ag|[a-zA-Z0-9\-]+)_(.*)$')
                        df['gene'] = split_data[0]
                        df['mutation'] = split_data[1]

                        df.dropna(subset=['gene', 'mutation'], inplace=True)

                for col in ['gene', 'mutation', 'codes_protein', 'gene_position', 'indel_length']:
                    if col not in df.columns:
                        df[col] = np.nan

                df_list.append(df)
                print(f"  -> [{custom_name}] loaded successfully and assigned to source group: [{source_label}]")
            except Exception as e:
                print(f"  -> [{custom_name}] failed to read: {e}")
        else:
            print(f"  -> [{custom_name}] file not found: {fpath}")

    if not df_list: return pd.DataFrame()

    merged_df = pd.concat(df_list, ignore_index=True)

    merged_df.drop_duplicates(subset=['uniqueid', 'gene', 'mutation'], inplace=True)

    return merged_df


def classify_variant(row):
    """Return the derived feature category for one mutation row."""
    gene = str(row.get('gene', ''))
    mutation = str(row.get('mutation', ''))

    is_coding = None
    if pd.notna(row.get('codes_protein')):
        val = str(row['codes_protein']).strip().lower()
        if val in ['true', '1', 'yes']:
            is_coding = True
        elif val in ['false', '0', 'no']:
            is_coding = False

    if is_coding is None and pd.notna(row.get('gene_position')):
        try:
            is_coding = (float(row['gene_position']) >= 0)
        except:
            pass

    try:
        indel_len = float(row.get('indel_length', 0))
    except:
        indel_len = 0
    is_indel_explicit = (indel_len != 0) and (not pd.isna(row.get('indel_length')))


    looks_like_indel = bool(re.search(r'(del|ins|fs)', mutation.lower()))

    looks_like_promoter = bool(re.search(r'(^|[a-zA-Z\.])-\d+', mutation)) or ('_-' in mutation)

    looks_like_amino_acid_snp = bool(re.search(r'^[a-zA-Z]\d+[a-zA-Z\*!]$', mutation))

    if is_coding is None:
        if looks_like_promoter:
            is_coding = False
        elif looks_like_amino_acid_snp:
            is_coding = True
        else:
            is_coding = True

    is_indel = is_indel_explicit or looks_like_indel

    if is_coding:
        if is_indel:
            if is_indel_explicit:
                return f"{gene}_coding_indel" if abs(indel_len) % 3 == 0 else f"{gene}_coding_frameshift"
            else:
                return f"{gene}_coding_indel"
        else:
            if len(mutation) > 1 and mutation[0] == mutation[-1] and mutation[0].isalpha():
                return "synonymous"
            return f"{gene}_coding_snp"
    else:
        return f"{gene}_noncoding_indel" if is_indel else f"{gene}_noncoding_snp"



def main():
    """Build aligned feature and label matrices."""
    mutations_df = load_and_merge_files(FILE_CONFIG)
    if mutations_df.empty:
        print("Error: no data were loaded")
        return

    print("\n--- 1.5 Checking phenotype sample list ---")
    try:
        phenotype_df = pd.read_csv(PHENOTYPE_FILE)
        if 'sra_accession' in phenotype_df.columns:
            valid_pheno_ids = phenotype_df['sra_accession'].astype(str).str.strip().unique()
            print(f"-> Loaded {len(valid_pheno_ids)} unique phenotyped samples from the phenotype file (sra_accession)")

            original_sample_count = mutations_df['uniqueid'].nunique()
            mutations_df = mutations_df[mutations_df['uniqueid'].isin(valid_pheno_ids)].copy()
            filtered_sample_count = mutations_df['uniqueid'].nunique()

            print(f"-> Original merged matrix sample count: {original_sample_count}")
            print(f"-> Cleaning removed {original_sample_count - filtered_sample_count} samples without phenotype labels.")
            print(f"-> Final retained genotype sample count: {filtered_sample_count}")

            if mutations_df.empty:
                print("Error: no data remain after filtering. Check whether ID formats are consistent.")
                return
        else:
            print(f"Error: 'sra_accession' column not found; filtering cannot continue.")
            return
    except FileNotFoundError:
        print(f"Error: phenotype file {PHENOTYPE_FILE} not found. Check the path.")
        return

    source_map = mutations_df[['uniqueid', 'source_dataset']].drop_duplicates(subset=['uniqueid'])

    try:
        gene_drug_df = pd.read_csv(INPUT_TARGET_GENE_FILE)
        target_genes_list = gene_drug_df['gene'].unique()
    except FileNotFoundError:
        print(f"Error: file not found: {INPUT_TARGET_GENE_FILE}")
        return

    print("\n--- 2. Data preprocessing and sample statistics ---")

    filtered_df = mutations_df[mutations_df['gene'].isin(target_genes_list)].copy()

    total_samples = filtered_df['uniqueid'].nunique()
    print(f"-> Valid sample count after retaining target-gene mutations only: {total_samples}")

    print("\n--- 3. Building label matrix (Y) ---")
    merged_labels_df = pd.merge(filtered_df, gene_drug_df[['gene', 'drug']], on='gene', how='left')

    labels_matrix = pd.crosstab(merged_labels_df['uniqueid'], merged_labels_df['drug'])
    labels_matrix = (labels_matrix > 0).astype(int)

    labels_matrix.reset_index(inplace=True)
    labels_matrix['uniqueid'] = labels_matrix['uniqueid'].astype(str).str.strip()

    print("\n--- 4. Building feature matrix (X) ---")
    print("  -> Computing derived categories...")

    filtered_df['gene_mutation'] = filtered_df['gene'] + '_' + filtered_df['mutation']
    filtered_df['derived_category'] = filtered_df.apply(classify_variant, axis=1)

    threshold = max(2, math.ceil(total_samples * BASE_RATIO))
    print(f"  -> Frequency threshold set to: >= {threshold} samples")

    mutation_counts = filtered_df.groupby('gene_mutation')['uniqueid'].nunique()
    common_muts_list = mutation_counts[mutation_counts >= threshold].index.tolist()
    rare_muts_list = mutation_counts[mutation_counts < threshold].index.tolist()

    print(f"  -> Common mutations kept as individual columns: {len(common_muts_list)}")
    print(f"  -> Rare mutations prepared for aggregation: {len(rare_muts_list)}")

    print("  -> Building common-mutation matrix...")
    df_common = filtered_df[filtered_df['gene_mutation'].isin(common_muts_list)]
    X_common = pd.crosstab(df_common['uniqueid'], df_common['gene_mutation'])
    X_common = (X_common > 0).astype(int)

    print("  -> Building aggregated rare-mutation matrix...")
    df_rare = filtered_df[
        (filtered_df['gene_mutation'].isin(rare_muts_list)) &
        (filtered_df['derived_category'] != 'synonymous') &
        (filtered_df['derived_category'].notna())
        ]

    if not df_rare.empty:
        X_rare = pd.crosstab(df_rare['uniqueid'], df_rare['derived_category'])
        X_rare = (X_rare > 0).astype(int)

        rare_cols_keep = [c for c in X_rare.columns if X_rare[c].sum() >= threshold]
        X_rare = X_rare[rare_cols_keep]
    else:
        X_rare = pd.DataFrame()

    X_final = pd.concat([X_common, X_rare], axis=1).fillna(0).astype(int)

    X_final.reset_index(inplace=True)
    X_final.rename(columns={'index': 'uniqueid'}, inplace=True)
    X_final['uniqueid'] = X_final['uniqueid'].astype(str).str.strip()

    print("\n--- 5. Data alignment and export ---")
    target_ids = labels_matrix['uniqueid'].unique()

    X_final.set_index('uniqueid', inplace=True)
    X_final = X_final.reindex(target_ids, fill_value=0)
    X_final.reset_index(inplace=True)

    Y_final = labels_matrix.set_index('uniqueid').reindex(target_ids).reset_index()

    X_final.sort_values('uniqueid', inplace=True)
    Y_final.sort_values('uniqueid', inplace=True)

    X_final = pd.merge(X_final, source_map, on='uniqueid', how='left')

    cols = X_final.columns.tolist()
    if 'source_dataset' in cols:
        cols.remove('source_dataset')
        meta_cols = ['uniqueid', 'source_dataset']
        feat_cols = [c for c in cols if c != 'uniqueid']
        X_final = X_final[meta_cols + feat_cols]

    if len(X_final) == len(Y_final):
        X_final.to_csv(OUTPUT_FEATURE_FILE, index=False, encoding='utf-8-sig')
        Y_final.to_csv(OUTPUT_LABEL_FILE, index=False, encoding='utf-8-sig')
        print("-> Processing complete. X feature matrix and Y label matrix have been saved.")
    else:
        print(f"Error: final X and Y row counts do not match.")

    print("\n--- 6. Final modeling-cohort source statistics ---")
    final_sample_count = len(X_final)
    print(f"Final feature matrix sample count: {final_sample_count}")

    if 'source_dataset' in X_final.columns:
        source_counts = X_final['source_dataset'].value_counts()
        source_percentages = (source_counts / final_sample_count) * 100

        stats_df = pd.DataFrame({
            'Sample count (N)': source_counts,
            'Percentage (%)': source_percentages.round(2)
        })

        print("-" * 35)
        print(stats_df.to_string())
        print("-" * 35)
    else:
        print("Warning: source_dataset column not found in the final matrix; source statistics cannot be reported.")


if __name__ == "__main__":
    main()
