# D:\FedXViT\dataset.py
import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset

class BrainMRIDataset(Dataset):
    def __init__(self, data_dir, metadata_csv, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        
        print(f"  - Loading metadata from {metadata_csv}...")
        try:
            self.metadata = pd.read_csv(metadata_csv)
        except FileNotFoundError:
            raise FileNotFoundError(f"Metadata CSV not found at {metadata_csv}. Please run preprocessing.py first.")

        self.metadata['target'] = self.metadata['class'].map({'normal': 0, 'tumor': 1})
        self.metadata.dropna(subset=['target'], inplace=True)
        self.metadata['target'] = self.metadata['target'].astype(int)
        print(f"  - Found {len(self.metadata)} records.")
        
    def __len__(self):
        return len(self.metadata)
    
    def __getitem__(self, idx):
        # ... (rest of the file is the same as before) ...
        row = self.metadata.iloc[idx]
        subfolder = 'Brain Tumor' if row['class'] == 'tumor' else 'Healthy'
        filename = f"{row['image']}.jpeg"
        img_path = os.path.join(self.data_dir, subfolder, filename)
        
        try:
            image = Image.open(img_path).convert('RGB')
            label = row['target']
        except FileNotFoundError:
            print(f"Warning: File not found {img_path}. Returning next sample.")
            return self.__getitem__((idx + 1) % len(self))

        if self.transform:
            image = self.transform(image)
            
        return image, label