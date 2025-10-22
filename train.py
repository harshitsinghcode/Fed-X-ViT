# D:\FedXViT\train.py
import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import ImageFolder
from torchvision import transforms
from tqdm import tqdm

from models.hybrid import HybridModel

def get_data_loaders(data_dir, batch_size, val_split):
    train_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(20),
        transforms.TrivialAugmentWide(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    test_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    train_path = os.path.join(data_dir, 'Training')
    test_path = os.path.join(data_dir, 'Testing')
    
    full_train_dataset = ImageFolder(train_path, transform=train_transforms)
    
    train_size = int((1 - val_split) * len(full_train_dataset))
    val_size = len(full_train_dataset) - train_size
    train_dataset, val_dataset = random_split(full_train_dataset, [train_size, val_size])
    
    test_dataset = ImageFolder(test_path, transform=test_transforms)

    print(f"  - Class mapping: {full_train_dataset.class_to_idx}")
    print(f"  - Total training images: {len(full_train_dataset)}")
    print(f"  - Splitting into {len(train_dataset)} train and {len(val_dataset)} validation images.")
    print(f"  - Total test images: {len(test_dataset)}")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    
    return train_loader, val_loader, test_loader, full_train_dataset.class_to_idx

def run_epoch(model, dataloader, criterion, optimizer, device, is_training):
    model.train() if is_training else model.eval()
    total_loss, correct_preds, total_samples = 0.0, 0, 0
    desc = "Training" if is_training else "Validating/Testing"
    with torch.set_grad_enabled(is_training):
        for images, labels in tqdm(dataloader, desc=desc):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            if is_training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            correct_preds += (preds == labels).sum().item()
            total_samples += labels.size(0)
    return total_loss / total_samples, correct_preds / total_samples

def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🚀 Using device: {device}")

    print("\n🧠 Initializing DataLoaders for 4-class dataset...")
    train_loader, val_loader, test_loader, class_map = get_data_loaders(args.data_dir, args.batch_size, args.val_split)
    
    print("\n🛠️ Building the Hybrid Model...")
    model = HybridModel(num_classes=args.num_classes).to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    best_val_acc = 0.0
    print("\n🔥 Starting training loop...")
    for epoch in range(args.epochs):
        print(f"\n--- Epoch {epoch + 1}/{args.epochs} ---")
        train_loss, train_acc = run_epoch(model, train_loader, criterion, optimizer, device, is_training=True)
        val_loss, val_acc = run_epoch(model, val_loader, criterion, None, device, is_training=False)
        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"Valid Loss: {val_loss:.4f} | Valid Acc: {val_acc:.4f}")
        scheduler.step()
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), args.model_save_path)
            print(f"✅ New best model saved with accuracy: {best_val_acc:.4f}")

    print("\nTraining finished!")
    print(f"🏆 Best Validation Accuracy: {best_val_acc:.4f}")
    
    print("\n🔬 Loading best model and running final evaluation on the unseen test set...")
    model.load_state_dict(torch.load(args.model_save_path))
    test_loss, test_acc = run_epoch(model, test_loader, criterion, None, device, is_training=False)
    print(f"🎯 Final Test Set Performance:")
    print(f"  - Test Loss: {test_loss:.4f}")
    print(f"  - Test Accuracy: {test_acc:.4f}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train a SOTA Hybrid Model for 4-Class Brain Tumor Classification")
    parser.add_argument('--data_dir', type=str, default='dataset', help='Root directory containing Training and Testing folders')
    parser.add_argument('--epochs', type=int, default=30, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-5, help='Weight decay for AdamW')
    parser.add_argument('--val_split', type=float, default=0.2, help='Validation split ratio from the training set')
    parser.add_argument('--num_classes', type=int, default=4, help='Number of classes')
    parser.add_argument('--model_save_path', type=str, default='best_multiclass_model.pth', help='Path to save the best model')
    
    args = parser.parse_args()
    main(args)