import json
import pandas as pd

METADATA_PATH = "C:\\Users\\gaura\\.cache\\huggingface\\hub\\datasets--AI4A-lab--RecruitView\\snapshots\\0cfa07ed0a43622f9104592b100d7bf3a25f6140\\metadata.jsonl"

def inspect_metadata(file_path):
    """
    Loads and inspects the RecruitView metadata.

    Args:
        file_path (str): The path to the metadata.jsonl file.
    """
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            data.append(json.loads(line))

    df = pd.DataFrame(data)

    print("--- Metadata Inspection ---")
    print(f"Number of records: {len(df)}")
    print("\n--- Columns ---")
    print(df.columns)
    
    print("\n--- First 5 rows ---")
    print(df.head())

    print("\n--- Value Ranges for Targets ---")
    expected_targets = [
        'agreeableness', 'conscientiousness', 'extraversion', 'neuroticism', 'openness',
        'confidence_score', 'overall_score', 'facial_expression', 'speaking_speed',
        'vocal_pitch', 'vocal_volume', 'eye_contact'
    ]
    
    actual_cols = df.columns.tolist()
    
    available_targets = list(set(expected_targets) & set(actual_cols))
    missing_targets = list(set(expected_targets) - set(actual_cols))
    extra_cols = list(set(actual_cols) - set(expected_targets))
    
    print("\n--- Target Column Analysis ---")
    print(f"Available targets for analysis: {available_targets}")
    print(f"Targets specified in PROCESS.md but missing in data: {missing_targets}")
    print(f"Columns in data but not in PROCESS.md targets: {extra_cols}")

    for target in available_targets:
        print(f"{target}: {df[target].min()} - {df[target].max()}")

    print("\n--- Video Paths ---")
    # The PROCESS.md specifies video_path, but the dataframe has file_name.
    # We will assume file_name is the correct column for now.
    if 'file_name' in df.columns:
        print(df['file_name'].head())
    else:
        print("'file_name' column not found.")


if __name__ == "__main__":
    inspect_metadata(METADATA_PATH)
