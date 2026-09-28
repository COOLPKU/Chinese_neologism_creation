import pandas as pd
import numpy as np

def calculate_fleiss_kappa(ratings, n):
    """
    Computes the Fleiss' Kappa value as described in (Fleiss, 1971)
    https://en.wikipedia.org/wiki/Fleiss%27_kappa
    
    :param ratings: a matrix of shape (N, k) where N is the number of subjects and k is the number of categories.
                    The element (i, j) represents the number of raters who assigned the i-th subject to the j-th category.
    :param n: number of raters per subject
    :return: Fleiss' Kappa
    """
    N, k = ratings.shape
    n_raters = n
    
    # Calculate p_j: the proportion of all assignments which were to the j-th category
    p = np.sum(ratings, axis=0) / (N * n_raters)
    
    # Calculate P_i: the extent to which raters agree for the i-th subject
    P_i = (np.sum(ratings**2, axis=1) - n_raters) / (n_raters * (n_raters - 1))
    
    # Calculate P_bar: the mean of P_i's
    P_bar = np.mean(P_i)
    
    # Calculate P_e: the sum of squares of p_j's
    P_e = np.sum(p**2)
    
    # Calculate Kappa
    if P_e == 1:
        return 1
    kappa = (P_bar - P_e) / (1 - P_e)
    
    return kappa

def main():
    files = {
        'liuyang': r'd:\new_word\discriminate\final_data_1124\annotation\annotation_data_liuyang.xlsx',
        'lql': r'd:\new_word\discriminate\final_data_1124\annotation\annotation_data_lql.xlsx',
        'whs': r'd:\new_word\discriminate\final_data_1124\annotation\annotation_data_whs.xlsx'
    }
    
    dfs = {}
    for name, path in files.items():
        try:
            # Read excel, assuming first sheet
            df = pd.read_excel(path)
            # Filter out rows where '词形' is NaN (empty rows)
            df = df.dropna(subset=['词形'])
            # Keep only relevant columns
            if '标注' not in df.columns:
                print(f"Warning: '标注' column not found in {name}")
                continue
            
            # Treat NaN or 0 as 0 (Not Accepted), 1 as 1 (Accepted)
            df['标注'] = pd.to_numeric(df['标注'], errors='coerce').fillna(0).astype(int)
            df['标注'] = df['标注'].apply(lambda x: 1 if x == 1 else 0)
            
            dfs[name] = df[['词形', '标注']].copy()
            
            # Drop duplicates first to avoid merge explosion
            dfs[name] = dfs[name].drop_duplicates(subset=['词形'])

            # Limit to first 200 words as requested
            if len(dfs[name]) > 200:
                print(f"Truncating {name} to first 200 rows.")
                dfs[name] = dfs[name].iloc[:200]
                
            print(f"Loaded {name}: {len(dfs[name])} rows")
        except Exception as e:
            print(f"Error loading {name}: {e}")
            return

    # Check if words are aligned
    words_liuyang = dfs['liuyang']['词形'].values
    words_lql = dfs['lql']['词形'].values
    words_whs = dfs['whs']['词形'].values
    
    # Reset index to ensure alignment by index if we assume order is same but indices are messed up
    for name in dfs:
        dfs[name] = dfs[name].reset_index(drop=True)

    # Try to merge on '词形' to be safe
    merged = dfs['liuyang'].rename(columns={'标注': 'liuyang'}).merge(
        dfs['lql'].rename(columns={'标注': 'lql'}), on='词形', how='inner'
    ).merge(
        dfs['whs'].rename(columns={'标注': 'whs'}), on='词形', how='inner'
    )
    
    if len(merged) != 200:
        print(f"Warning: Merged data has {len(merged)} rows, expected 200.")
    else:
        print(f"Merged data successfully, {len(merged)} rows.")

    # We have already filled NaNs with 0, so no need to drop rows based on labels
    # original_len = len(merged)
    # merged = merged.dropna(subset=['liuyang', 'lql', 'whs'])
    # if len(merged) < original_len:
    #     print(f"Dropped {original_len - len(merged)} rows due to missing labels.")

    print(f"Final dataset size for calculation: {len(merged)}")
    
    # Prepare data for Fleiss' Kappa
    # We need to count how many raters assigned each category (0 and 1) for each subject
    # Categories: 0, 1
    n_categories = 2
    n_raters = 3
    n_subjects = len(merged)
    
    # Reset index of merged to iterate properly
    merged = merged.reset_index(drop=True)

    ratings_matrix = np.zeros((n_subjects, n_categories), dtype=int)
    
    for i, row in merged.iterrows():
        # Count 0s and 1s
        labels = [row['liuyang'], row['lql'], row['whs']]
        for label in labels:
            if label == 0:
                ratings_matrix[i, 0] += 1
            elif label == 1:
                ratings_matrix[i, 1] += 1
            else:
                # Handle unexpected labels if necessary
                pass
                
    kappa = calculate_fleiss_kappa(ratings_matrix, n_raters)
    print(f"\nFleiss' Kappa: {kappa:.4f}")

    # Calculate Acceptance Rates (Positive Rates)
    print("\n--- Acceptance Rates (Label = 1) ---")
    for name in ['liuyang', 'lql', 'whs']:
        count = merged[name].sum()
        total = len(merged)
        print(f"{name}: {count}/{total} ({count/total:.2%})")
        
    # Majority Vote
    merged['sum_score'] = merged['liuyang'] + merged['lql'] + merged['whs']
    majority_accepted = (merged['sum_score'] >= 2).sum()
    unanimous_accepted = (merged['sum_score'] == 3).sum()
    
    print(f"Majority Vote (>=2): {majority_accepted}/{len(merged)} ({majority_accepted/len(merged):.2%})")
    print(f"Unanimous Vote (3): {unanimous_accepted}/{len(merged)} ({unanimous_accepted/len(merged):.2%})")
    
    # Calculate Pairwise Agreement
    # (liuyang, lql), (liuyang, whs), (lql, whs)
    pairs = [('liuyang', 'lql'), ('liuyang', 'whs'), ('lql', 'whs')]
    agreements = []
    for p1, p2 in pairs:
        agree = (merged[p1] == merged[p2]).mean()
        agreements.append(agree)
        print(f"Agreement ({p1} vs {p2}): {agree:.4f}")
        
    print(f"Average Pairwise Agreement: {np.mean(agreements):.4f}")

if __name__ == "__main__":
    main()
