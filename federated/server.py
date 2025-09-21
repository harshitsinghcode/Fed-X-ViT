import torch
from models.hybrid_model import HybridModel
from aggregation import fed_avg

class FLServer:
    def __init__(self, clients, device):
        self.clients = clients
        self.device = device
        self.global_model = HybridModel(num_classes=2).to(device)  # Adjust classes as needed

    def run_federated_learning(self, rounds=10, local_epochs=1):
        for r in range(rounds):
            print(f"\n=== Federated Round {r + 1} ===")
            global_weights = self.global_model.state_dict()
            client_updates = []

            for idx, client in enumerate(self.clients):
                client.model.load_state_dict(global_weights)
                weights, val_loss, val_acc = client.train_local(epochs=local_epochs)
                num_samples = len(client.train_loader.dataset)
                client_updates.append((weights, num_samples))
                print(f"Client {idx + 1} validation accuracy: {val_acc:.4f}")

            # Aggregate weights from all clients with FedAvg
            new_global_weights = fed_avg(client_updates)
            self.global_model.load_state_dict(new_global_weights)
            print(f"Round {r + 1} aggregation complete.")

        torch.save(self.global_model.state_dict(), "fedxvit_global_model.pth")
        print("Federated training finished and global model saved.")
