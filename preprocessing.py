# D:\FedXViT\preprocessing.py
import os
from PIL import Image
from tqdm import tqdm
import pandas as pd

def make_uniform_folder(source_dir, target_dir, size=(224, 224), target_format='jpeg'):
    """Resizes and saves images from a source folder to a target folder."""
    os.makedirs(target_dir, exist_ok=True)
    for fname in tqdm(os.listdir(source_dir), desc=f"Processing {os.path.basename(source_dir)}"):
        if not fname.lower().endswith(('.jpg', '.jpeg', '.png', '.tif', '.bmp')):
            continue
        try:
            img = Image.open(os.path.join(source_dir, fname)).convert("RGB")
            img = img.resize(size)
            
            base_name = os.path.splitext(fname)[0]
            clean_name = f"{base_name}.{target_format}"
            img.save(os.path.join(target_dir, clean_name), target_format.upper())

        except Exception as e:
            print(f"Skipping {fname}: {e}")

def generate_metadata_csv(processed_dir, output_csv_path):
    """Generates a metadata CSV from the processed folder structure."""
    data = []
    for category in ['Brain Tumor', 'Healthy']:
        folder_path = os.path.join(processed_dir, category)
        for filename in os.listdir(folder_path):
            # The class is 'tumor' for 'Brain Tumor' and 'normal' for 'Healthy'
            class_label = 'tumor' if category == 'Brain Tumor' else 'normal'
            data.append({'image': os.path.splitext(filename)[0], 'class': class_label})
    
    df = pd.DataFrame(data)
    df.to_csv(output_csv_path, index=False)
    print(f"Metadata CSV generated at {output_csv_path}")


if __name__ == "__main__":
    SRC_ROOT = r'dataset/Brain Tumor Data Set/Brain Tumor Data Set'
    DST_ROOT = r'dataset/Processed'
    METADATA_CSV_PATH = r'dataset/metadata.csv'

    print("--- Starting Dataset Preprocessing ---")
    
    # Process both Brain Tumor and Healthy folders
    for category in ['Brain Tumor', 'Healthy']:
        in_dir = os.path.join(SRC_ROOT, category)
        out_dir = os.path.join(DST_ROOT, category)
        if os.path.isdir(in_dir):
            make_uniform_folder(in_dir, out_dir, size=(224, 224))
        else:
            print(f"Warning: Source directory not found - {in_dir}")
            
    # Generate the metadata file
    generate_metadata_csv(DST_ROOT, METADATA_CSV_PATH)
    
    print("\n--- Preprocessing Complete ---")