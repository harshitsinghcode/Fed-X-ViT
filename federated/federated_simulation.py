import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from dataset import BrainMRIDataset
from models.hybrid_model import HybridModel
from client import FLClient
from server import FLServer
from utils import split_dataset
import torch.nn as nn
import torch.optim as optim

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def create_client_loaders():
    data_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    full_dataset = BrainMRIDataset('dataset/Processed',
                                  'dataset/Brain Tumor Data Set/metadata.csv',
                                  transform=data_transforms)

    client_subsets = split_dataset(full_dataset, n_clients=3)
    client_loaders = []

    for subset in client_subsets:
        loader = DataLoader(subset, batch_size=32, shuffle=True, num_workers=4, pin_memory=True)
        client_loaders.append(loader)
    return client_loaders

def main():
    client_loaders = create_client_loaders()

    clients = []
    criterion = nn.CrossEntropyLoss()

    for loader in client_loaders:
        model = HybridModel(num_classes=2).to(device)
        optimizer = optim.Adam(model.parameters(), lr=1e-4)

        # For local validation, reuse client loader here (or create val loaders per client)
        client = FLClient(model=model, train_loader=loader, val_loader=loader,
                          device=device, criterion=criterion, optimizer=optimizer)
        clients.append(client)

    server = FLServer(clients=clients, device=device)
    server.run_federated_learning(rounds=5, local_epochs=1)

if __name__ == "__main__":
    main()
