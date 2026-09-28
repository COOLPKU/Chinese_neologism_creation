import pandas as pd

files = {
    'whs': r'd:\new_word\discriminate\final_data_1124\annotation\annotation_data_part2_whs.xlsx',
    'lql': r'd:\new_word\discriminate\final_data_1124\annotation\annotation_data_part2_lql.xlsx',
    'liuyang': r'd:\new_word\discriminate\final_data_1124\annotation\annotation_data_part2_liuyang.xlsx'
}

for name, path in files.items():
    print(f"\n{'='*60}")
    print(f"File: {name}")
    print('='*60)
    df = pd.read_excel(path)
    print(f"Total rows: {len(df)}")
    print(f"Columns: {df.columns.tolist()}")
    
    if '词形' in df.columns:
        print(f"词形 not null: {df['词形'].notna().sum()}")
        print(f"\nFirst 5 rows with 词形:")
        print(df[df['词形'].notna()].head())
    else:
        print("WARNING: No '词形' column found!")
