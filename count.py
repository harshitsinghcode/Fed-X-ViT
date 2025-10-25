import os
from pathlib import Path

# --- Configuration ---
# 1. Set the path to your main data folder
ROOT_FOLDER = Path(r"D:\FedXViT\SplitData")

# 2. Define the client and split folders to search
CLIENT_DIRS = ["CLIENT_0_DATA", "CLIENT_1_DATA", "CLIENT_2_DATA"]
SPLIT_DIRS = ["Training", "Validation", "Testing"]

# 3. Define the file types to count
FILE_EXTENSIONS = {".jpg", ".jpeg", ".png"}
# ---------------------

def count_images(directory_path: Path) -> int:
    """Recursively counts all files with matching extensions in a directory."""
    file_count = 0
    
    # Use glob to find all files recursively
    # The "**" means it will search all subfolders
    for file_path in directory_path.glob("**/*"):
        if file_path.suffix.lower() in FILE_EXTENSIONS:
            file_count += 1
    return file_count

if __name__ == "__main__":
    # A dictionary to store the grand totals
    grand_totals = {"Training": 0, "Validation": 0, "Testing": 0}

    print(f"--- 📊 Starting Data Count in {ROOT_FOLDER} ---")
    
    # Loop through each client
    for client in CLIENT_DIRS:
        print(f"\n--- Processing {client} ---")
        client_path = ROOT_FOLDER / client
        
        if not client_path.is_dir():
            print(f"  ⚠️  Warning: Folder not found: {client_path}")
            continue

        # Loop through Training, Validation, and Testing
        for split in SPLIT_DIRS:
            current_path = client_path / split
            
            if not current_path.is_dir():
                print(f"  - {split}: 0 images (Directory not found)")
                continue

            # Count the images in this folder and all its subfolders
            count = count_images(current_path)
            
            # Print the count for this specific subfolder
            print(f"  ✅ {split}: {count} images")
            
            # Add to the grand total
            grand_totals[split] += count

    # --- Print the Final Summary ---
    print("\n" + "="*40)
    print("      FINAL COMBINED TOTALS")
    print("="*40)
    print(f"  Total Training Images (All Clients): {grand_totals['Training']}")
    print(f"  Total Validation Images (All Clients): {grand_totals['Validation']}")
    print(f"  Total Testing Images (All Clients): {grand_totals['Testing']}")
    print("="*40)