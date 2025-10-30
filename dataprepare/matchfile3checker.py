import pandas as pd
from pathlib import Path
import re
import os # For os.sep, though pathlib often handles it.

# --- Configuration ---
OUTPUT_BASE_DIR = Path("/model/data/Research/temp")
SOURCE_PROJECT_BASE_DIR = Path("/model/data/Research/sourceProject/actionableCS")
FILE_PATH_COLUMN_NAME = "File Path"
JAVA_FILE_NOT_FOUND_MSG = "File Not Found"
AMBIGUOUS_MSG_PREFIX = "Ambiguous"

# Regex to find package declaration
PACKAGE_REGEX = re.compile(r"^\s*package\s+([a-zA-Z_][\w\.]*)\s*;")

# Global cache for source file indexes:
# { (project_name, version_name): { (pkg_from_file, type_from_file): "abs/path/to/file.java", ... }, ... }
SOURCE_FILE_INDEX_CACHE = {}

def get_package_from_java_content(java_file_path: Path):
    """
    Reads a Java file and extracts the package name from its package declaration.
    Returns the package name string, "" for default package, or None if error.
    """
    package_name = "" # Default to empty string (default package)
    encodings_to_try = ['utf-8', 'latin-1', 'iso-8859-1', 'windows-1252']
    content_successfully_read_or_parsed = False

    for encoding in encodings_to_try:
        try:
            with open(java_file_path, 'r', encoding=encoding, errors='ignore') as f:
                # Limit lines read for performance; package is usually at the top.
                lines_to_check = 100
                for i, line in enumerate(f):
                    if i >= lines_to_check and content_successfully_read_or_parsed: # already found
                        break
                    match = PACKAGE_REGEX.match(line)
                    if match:
                        package_name = match.group(1)
                        content_successfully_read_or_parsed = True
                        break
                    if i >= lines_to_check: # Reached limit without finding
                        break
                # If after checking N lines we found the package, or even if we didn't but read successfully.
                # The important part is that we could open and read with this encoding.
                content_successfully_read_or_parsed = True # Mark as successfully processed with this encoding
                break # Successfully processed with this encoding, no need to try others
        except UnicodeDecodeError:
            continue
        except Exception as e:
            # print(f"Warning: Could not read {java_file_path} with encoding {enc}: {e}")
            continue # Try next encoding

    if not content_successfully_read_or_parsed:
        print(f"Warning: Could not read or parse package from {java_file_path} after trying all encodings.")
        return None # Indicate failure to determine package
    return package_name


def build_version_file_index(source_version_path: Path) -> dict:
    """
    Builds an index for a specific project version's source files.
    Index: { (package_name_from_file, type_name_from_file): "abs/path/to/file.java", ... }
    """
    version_index = {}
    if not source_version_path.is_dir():
        print(f"Warning: Source directory not found for indexing: {source_version_path}")
        return version_index

    print(f"Indexing Java files in: {source_version_path}...")
    java_files_scanned = 0
    java_files_indexed = 0
    for java_file in source_version_path.rglob("*.java"):
        java_files_scanned += 1
        type_name_from_file = java_file.stem # Filename without .java extension

        package_name_from_file = get_package_from_java_content(java_file)

        if package_name_from_file is not None:
            version_index[(package_name_from_file, type_name_from_file)] = str(Path(java_file).relative_to(source_version_path))
            java_files_indexed += 1
        else:
            print(f"Notice: Could not determine package for {java_file.name}, it won't be indexed precisely by package.")
            # Optionally, index it with a special package key if needed, or just by type.
            # For this problem, if package can't be determined, it might not match well.

    print(f"Scanned {java_files_scanned} .java files. Indexed {java_files_indexed} files for {source_version_path.name}.")
    return version_index

def normalize_csv_package_name(package_name_csv_val):
    """Handles (Default), NaN, None, or empty strings from CSV for package name."""
    if pd.isna(package_name_csv_val) or package_name_csv_val in ["(Default)", ""]:
        return ""
    for i, char in enumerate(str(package_name_csv_val)):
        if char.isupper():
            return re.sub(r'\.[A-Z_].*$', '', package_name_csv_val)
    return str(package_name_csv_val)


def process_csv_file(csv_path: Path):
    """
    Processes a single CSV file: reads it, finds Java file paths, and writes back.
    """
    failed = 0
    print(f"\nProcessing CSV: {csv_path.name} in project {csv_path.parent.name}")
    try:
        df = pd.read_csv(csv_path, dtype=str) # Read all as string initially
        df.fillna("", inplace=True) # Replace NaN with empty strings
    except Exception as e:
        print(f"Error reading CSV {csv_path}: {e}")
        return

    # Ensure required columns exist
    required_cols = ['Project', 'Version', 'Type Name', 'Package Name']
    for col in required_cols:
        if col not in df.columns:
            print(f"Error: CSV {csv_path.name} is missing required column '{col}'. Skipping this file.")
            return

    new_file_paths_column = []

    for index, row in df.iterrows():
        project_csv = str(row['Project']).strip()
        version_csv = str(row['Version']).strip()
        type_name_csv = str(row['Type Name']).strip()
        package_name_csv = str(row['Package Name']).strip()

        if not project_csv or not version_csv or not type_name_csv:
            new_file_paths_column.append("Missing Key Info in Row")
            continue
            
        # Get or build the file index for this project and version
        cache_key = (project_csv, version_csv)
        if cache_key not in SOURCE_FILE_INDEX_CACHE:
            source_version_dir = SOURCE_PROJECT_BASE_DIR / project_csv / '-'.join([project_csv, version_csv])
            SOURCE_FILE_INDEX_CACHE[cache_key] = build_version_file_index(source_version_dir)

        version_file_index = SOURCE_FILE_INDEX_CACHE[cache_key]
        if not version_file_index: # No files indexed for this version
            failed +=1
            new_file_paths_column.append(JAVA_FILE_NOT_FOUND_MSG + " (No source index)")
            continue    

        # Prepare for search
        # For Outer$Inner, actual file is Outer.java
        searchable_type_name = type_name_csv.split('$')[0]
        normalized_package_csv = normalize_csv_package_name(package_name_csv)

        # Find candidates by Type Name
        candidate_files = []
        for (pkg_from_file, type_from_file), abs_path in version_file_index.items():
            if type_from_file == searchable_type_name:
                candidate_files.append({'path': abs_path, 'package': pkg_from_file})

        # Evaluate candidates
        if not candidate_files:
            failed += 1
            new_file_paths_column.append(JAVA_FILE_NOT_FOUND_MSG)
        elif len(candidate_files) == 1:
            new_file_paths_column.append(candidate_files[0]['path'])
        else: # Multiple candidates found for Type Name, use Package Name to disambiguate
            exact_package_matches = [
                cand['path'] for cand in candidate_files if cand['package'] == normalized_package_csv
            ]
            if len(exact_package_matches) == 1:
                new_file_paths_column.append(exact_package_matches[0])
            elif len(exact_package_matches) == 0:
                paths_str = "; ".join([cand['path'] for cand in candidate_files])
                new_file_paths_column.append(f"{AMBIGUOUS_MSG_PREFIX}: Type '{searchable_type_name}' found in {len(candidate_files)} locations, none in package '{normalized_package_csv}'. Paths: {paths_str[:200]}")
            else: # > 1 exact_package_matches (e.g., duplicate files in the same package and type)
                paths_str = "; ".join(exact_package_matches)
                new_file_paths_column.append(f"{AMBIGUOUS_MSG_PREFIX}: Type '{searchable_type_name}' and Package '{normalized_package_csv}' matched {len(exact_package_matches)} files. Paths: {paths_str[:200]}")

    df[FILE_PATH_COLUMN_NAME] = new_file_paths_column
    try:
        df.to_csv(csv_path, index=False, encoding='utf-8')
        print(f"Successfully updated CSV: {csv_path.name}, failed find {failed} java file")
    except Exception as e:
        print(f"Error writing updated CSV {csv_path}: {e}")


def main():
    if not OUTPUT_BASE_DIR.is_dir():
        print(f"Error: Output base directory not found: {OUTPUT_BASE_DIR}")
        return
    if not SOURCE_PROJECT_BASE_DIR.is_dir():
        print(f"Error: Source project base directory not found: {SOURCE_PROJECT_BASE_DIR}")
        return

    # Iterate through project folders in the output directory
    for project_output_dir in OUTPUT_BASE_DIR.iterdir():
        if project_output_dir.is_dir():
            # Iterate through CSV files in this project's output folder
            for csv_file in project_output_dir.rglob("*.csv"): # rglob for subdirs if any
                process_csv_file(csv_file)

    print("\n--- All CSV processing complete ---")

if __name__ == "__main__":
    main()

