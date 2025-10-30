import os
import re
import pandas as pd
from pathlib import Path
from matchfile3checker import get_package_from_java_content

# Configuration
A_BASE_PATH = "/model/data/Research/tools/Checkstyle/excel_output"
B_BASE_PATH = "/model/data/Research/temp"
OUTPUT_BASE_PATH = "/model/data/Research/temp_merged_output"  # New output directory
SOURCE_PROJECT_BASE_DIR = Path("/model/data/Research/sourceProject/actionableCS")
# Smell conversion mapping
SMELL_CONVERSION = {
    "complex class": "God Class",
    "FeatureEnvy": "Feature Envy",
    "ParametersPerMethod": "Long Parameter List",
    "complex method": "Complex Method"
}


def load_excel_file(file_path):
    """Load Excel file into a DataFrame."""
    try:
        return pd.read_excel(file_path,
        na_values=["", "NA", "N/A", "null", "NaN"],  # 指定哪些值被视为 NaN
        keep_default_na=False  # 不自动解析默认的 NaN 值（如空单元格）
    ).fillna("")
    except Exception as e:
        print(f"Error loading Excel file {file_path}: {e}")
        return None

def load_csv_file(file_path):
    """Load CSV file into a DataFrame."""
    try:
        return pd.read_csv(file_path,
        na_values=["", "NA", "N/A", "null", "NaN"],  # 指定哪些值被视为 NaN
        keep_default_na=False  # 不自动解析默认的 NaN 值（如空单元格）
    ).fillna("")
    except Exception as e:
        print(f"Error loading CSV file {file_path}: {e}")
        return None

def convert_smell(smell):
    """Convert smell values based on the mapping, keep unchanged if not in mapping."""
    return SMELL_CONVERSION.get(smell, smell)

def match_file_path(a_path, b_path):
    """Check if A's file path is a suffix of B's file path."""
    if pd.isna(a_path) or pd.isna(b_path):
        return False
    # Normalize paths to handle different separators
    a_path = str(a_path).split('/', 1)[1].replace("\\", "/")
    b_path = str(b_path).replace("\\", "/")
    return b_path.endswith(a_path)

def transform_file_path(a_path, project, version):
    """Transform A's File Path to match B's format."""
    if pd.isna(a_path):
        return a_path
    a_path = str(a_path).replace("\\", "/")
    # Expected A path: <project>-<version>/<rest>
    # Strip <project>-<version>/ from the start
    prefix = f"{version}/"
    # if a_path.startswith(prefix):
    #     rest = a_path[len(prefix):]
    # else:
    rest = str(a_path).split('/', 1)[1]# Fallback: keep as-is if prefix doesn't match
    # Prepend B-style prefix
    # return f"/model/data/Research/sourceProject/actionableCS/{project}/{version}/{rest}"
    return f"{rest}"
def get_project_versions(base_path):
    """Scan directory for projects and their versions."""
    projects = {}
    for project in os.listdir(base_path):
        project_path = os.path.join(base_path, project)
        if os.path.isdir(project_path):
            versions = [os.path.splitext(v)[0] for v in os.listdir(project_path) if v.endswith('.csv')]
            projects[project] = versions
    return projects

def merge_data(a_df, b_df, project, version):
    """Merge data from A into B, appending unmatched A rows as new entries."""
    if a_df is None:
        return b_df
    if b_df is None:
        b_df = pd.DataFrame(columns=["Project", "Version", "Package Name", "Type Name", "Method Name", "Smell", "checker", "File Path"])

    # Apply smell conversion to A
    a_df = a_df.copy()
    a_df["smell"] = a_df["smell"].apply(convert_smell)
    a_df["Package Name"] = a_df['File Path'].apply(lambda x: get_package_from_java_content(os.path.join(SOURCE_PROJECT_BASE_DIR, project, version, x)))
    # Ensure checker column exists in BSOURCE_PROJECT_BASE_DIR / project_csv / '_'.join(['STRUTS',version_csv])
    b_df = b_df.copy()
    if "checker" not in b_df.columns:
        b_df["checker"] = ""

    # Track matched rows in A
    a_matched = set()
    # Iterate over B to find matches and update checker
    for b_idx in b_df.index:
        b_row = b_df.loc[b_idx]
        # b_package = b_row["Package Name"]
        b_file = b_row['File Path']
        b_type = b_row["Type Name"]
        b_method = "" if pd.isna(b_row["Method Name"]) else b_row["Method Name"]
        b_smell = b_row["Smell"]

        if pd.isna(b_type) or pd.isna(b_method) or pd.isna(b_smell):
            continue

        # Find matching rows in A
        matches = a_df[
            (a_df["File Path"] == b_file) &
            (a_df["Class Name"] == b_type) &
            (a_df["Method Name"] == b_method) &
            (a_df["smell"] == b_smell)
        ]

        # If there’s a match, append ",checkstyle" to checker
        if not matches.empty:
            checker = b_df.at[b_idx, "checker"]
            if pd.isna(checker) or checker == "":
                b_df.at[b_idx, "checker"] = "checkstyle"
            elif "checkstyle" not in checker:
                b_df.at[b_idx, "checker"] = f"{checker},checkstyle"
            a_matched.update(matches.index)

    # Handle unmatched A rows
    unmatched_a = a_df[~a_df.index.isin(a_matched)]
    if not unmatched_a.empty:
        # Adjust version to remove project prefix (e.g., dubbo-dubbo-2.7.0 -> dubbo-2.7.0)
        adjusted_version = re.search(r'\d.*', version).group(0)
        # Create new rows for B from unmatched A
        new_rows = pd.DataFrame({
            "Project": project,
            "Version": adjusted_version,
            "Package Name": unmatched_a["Package Name"],
            "Type Name": unmatched_a["Class Name"],
            "Method Name": unmatched_a["Method Name"],
            "Smell": unmatched_a["smell"],
            "checker": "checkstyle",
            "File Path": unmatched_a["File Path"]
        })
        # Append new rows to B
        b_df = pd.concat([b_df, new_rows], ignore_index=True)

    # Keep only required columns
    required_columns = ["Project", "Version", "Package Name","Type Name", "Method Name", "Smell", "checker", "File Path"]
    b_df = b_df[required_columns]
    return b_df


def save_merged_file(df, output_path):
    """Save merged DataFrame to CSV."""
    try:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df.to_csv(output_path, index=False)
        print(f"Saved merged file to {output_path}")
    except Exception as e:
        print(f"Error saving file {output_path}: {e}")

def process_project_version(project, version, a_base_path, b_base_path, output_base_path):
    """Process a single project version."""
    # Paths for A and B files
    a_file = os.path.join(a_base_path, project, f"{version}.csv")
    b_file = os.path.join(b_base_path, project, f"{version}.csv")
    output_file = os.path.join(output_base_path, project, f"{version}.csv")

    # Load files
    a_df = load_csv_file(a_file) if os.path.exists(a_file) else None
    b_df = load_csv_file(b_file) if os.path.exists(b_file) else None

    if b_df is None:
        print(f"No CSV file found for {project}/{version}, skipping.")
        return

    # Merge data
    merged_df = merge_data(a_df, b_df, project, version)

    # Save result
    save_merged_file(merged_df, output_file)

def main():
    """Main function to process all projects and versions."""
    # Get projects and versions from B (since B is the base dataset)
    projects = get_project_versions(B_BASE_PATH)

    for project, versions in projects.items():
        print(f"Processing project: {project}")
        for version in versions:
            print(f"  Processing version: {version}")
            process_project_version(project, version, A_BASE_PATH, B_BASE_PATH, OUTPUT_BASE_PATH)

if __name__ == "__main__":
    main()
