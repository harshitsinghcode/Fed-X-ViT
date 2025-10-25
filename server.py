# import flwr as fl
# import torch
# import os
# import mlflow
# from collections import OrderedDict
# from azure.identity import DefaultAzureCredential
# from azure.ai.ml import MLClient
# from models.hybrid_model import HybridModel 
# from client import FlowerClient, load_client_data
# from typing import List, Tuple, Dict, Union
# from flwr.common import Metrics
# from sklearn.metrics import classification_report
# from torch.utils.data import DataLoader, ConcatDataset
# from torchvision.datasets import ImageFolder
# from torchvision import transforms

# # --- 1. CONNECT TO AZURE ML ---
# print("--- [SERVER LOG] ---")
# print("🧠 Connecting to Azure ML Workspace...")
# ml_client = MLClient.from_config(credential=DefaultAzureCredential())
# mlflow_tracking_uri = ml_client.workspaces.get(ml_client.workspace_name).mlflow_tracking_uri
# mlflow.set_tracking_uri(mlflow_tracking_uri)
# print(f"✅ Connected to Azure ML: {ml_client.workspace_name}")

# # --- 2. LOAD YOUR PRE-TRAINED "STAR PLAYER" MODEL ---
# print("🧠 Loading pre-trained 4-class model (best_multiclass_model.pth)...")
# model_architecture = HybridModel(num_classes=2) 
# model_dict = model_architecture.state_dict()
# pretrained_dict = torch.load("best_multiclass_model.pth", map_location="cpu")
# pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict and model_dict[k].shape == v.shape}
# model_dict.update(pretrained_dict)
# model_architecture.load_state_dict(model_dict)
# print(f"✅ Transferred {len(pretrained_dict)} layers. New 2-class model is ready.")

# initial_parameters = fl.common.ndarrays_to_parameters(
#     [val.cpu().numpy() for _, val in model_architecture.state_dict().items()]
# )

# # --- 3. DEFINE A METRICS AGGREGATION FUNCTION ---
# server_round_g = 0 
# def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
#     global server_round_g
#     accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
#     losses = [num_examples * m["loss"] for num_examples, m in metrics]
#     examples = [num_examples for num_examples, _ in metrics]
    
#     if sum(examples) == 0:
#         return {"accuracy": 0.0, "loss": 0.0} 
        
#     avg_accuracy = sum(accuracies) / sum(examples)
#     avg_loss = sum(losses) / sum(examples)
    
#     print(f"--- [SERVER LOG] Global Val Accuracy: {avg_accuracy:.4f} | Global Val Loss: {avg_loss:.4f} ---")
#     mlflow.log_metric("global_val_accuracy", avg_accuracy, step=server_round_g)
#     mlflow.log_metric("global_val_loss", avg_loss, step=server_round_g)
#     return {"accuracy": avg_accuracy, "loss": avg_loss}

# # --- 4. CREATE A CUSTOM STRATEGY WITH AZURE LOGGING ---
# class AzureFedAvg(fl.server.strategy.FedAvg):
#     def aggregate_fit(self, server_round, results, failures):
#         global server_round_g 
#         server_round_g = server_round
        
#         print(f"--- [SERVER LOG - ROUND {server_round}] ---")
#         if failures:
#             print(f"🚨 Failures reported: {failures}")
        
#         agg_params, agg_metrics = super().aggregate_fit(server_round, results, failures)
        
#         if agg_metrics:
#             mlflow.log_metric("Round", server_round)
#             mlflow.log_metric("Clients in Round", len(results))

#         if agg_params is not None and server_round == 5: 
#              print("🧠 Saving final aggregated model to 'final_federated_model.pth'...")
#              nd_arrays = fl.common.parameters_to_ndarrays(agg_params)
#              params_dict = zip(model_architecture.state_dict().keys(), nd_arrays)
#              state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
#              torch.save(state_dict, "final_federated_model.pth")
             
#         return agg_params, agg_metrics

# # --- 5. DEFINE THE CLIENT-SPAWNING FUNCTION ---
# def client_fn(cid: str) -> fl.client.Client:
#     print(f"--- [SERVER LOG] Spawning Client {cid} ---")
#     client_data_path = f"D:\\FedXViT\\SplitData\\CLIENT_{cid}_DATA" 
#     print(f"Client {cid} will load data from: {client_data_path}")

#     client_model = HybridModel(num_classes=2)
    
#     # This line now correctly unpacks the (train_loader, val_loader) tuple
#     train_loader, val_loader = load_client_data(client_data_path) 
    
#     return FlowerClient(client_model, train_loader, val_loader).to_client()

# # --- 6. START THE MLFLOW EXPERIMENT RUN ---
# with mlflow.start_run(run_name="FedXViT_Local_Sim_with_Eval") as run:
#     run_id = run.info.run_id
#     print(f"🔥 Starting MLflow Run: {run_id}")
#     print(f"Follow progress in Azure ML Studio: {mlflow.get_tracking_uri()}")

#     mlflow.log_params({
#         "strategy": "FedAvg (Transfer Learning)",
#         "num_clients": 3,
#         "num_rounds": 5,
#         "architecture": "Hybrid (EfficientNetV2 + ViT)"
#     })

#     strategy = AzureFedAvg(
#         fraction_fit=1.0,
#         min_fit_clients=3,
#         min_available_clients=3,
#         initial_parameters=initial_parameters,
#         evaluate_metrics_aggregation_fn=weighted_average, 
#     )
    
#     print("🚀 Starting local simulation on RTX 3060...")
#     fl.simulation.start_simulation(
#         client_fn=client_fn,
#         num_clients=3,
#         config=fl.server.ServerConfig(num_rounds=5),
#         strategy=strategy,
#         # --- THIS IS THE PARALLEL COMMAND ---
#         # We are telling Ray to run 3 clients in parallel,
#         # each using 1/3rd of the GPU.
#         client_resources={"num_gpus": 0.33} 
#     )

#     # --- 7. AUTOMATIC FINAL TESTING ---
#     print("\n🏁 Federated training complete. Starting final test on unseen data...")
#     DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
#     final_model = HybridModel(num_classes=2)
#     try:
#         final_model.load_state_dict(torch.load("final_federated_model.pth"))
#     except FileNotFoundError:
#         print("🚨 ERROR: 'final_federated_model.pth' not found. Skipping final test.")
#         mlflow.set_tag("status", "failed_to_save_model")
#         exit()
        
#     final_model.to(DEVICE)
#     final_model.eval()

#     test_transforms = transforms.Compose([
#         transforms.Resize((224, 224)),
#         transforms.ToTensor(),
#         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
#     ])
    
#     client_0_test = ImageFolder(os.path.join(r"D:\FedXViT\SplitData", "CLIENT_0_DATA", "Testing"), transform=test_transforms)
#     client_1_test = ImageFolder(os.path.join(r"D:\FedXViT\SplitData", "CLIENT_1_DATA", "Testing"), transform=test_transforms)
#     client_2_test = ImageFolder(os.path.join(r"D:\FedXViT\SplitData", "CLIENT_2_DATA", "Testing"), transform=test_transforms)
    
#     combined_test_dataset = ConcatDataset([client_0_test, client_1_test, client_2_test])
#     test_loader = DataLoader(combined_test_dataset, batch_size=16, shuffle=False)
#     print(f"✅ Combined test dataset created with {len(combined_test_dataset)} total images.")
    
#     criterion = torch.nn.CrossEntropyLoss()
#     total_loss, correct, total = 0.0, 0, 0
#     all_preds = []
#     all_labels = []

#     with torch.no_grad():
#         for images, labels in test_loader:
#             images, labels = images.to(DEVICE), labels.to(DEVICE)
#             outputs = final_model(images)
#             loss = criterion(outputs, labels)
#             total_loss += loss.item() * images.size(0)
#             preds = outputs.argmax(dim=1)
#             correct += (preds == labels).sum().item()
#             total += labels.size(0)
#             all_preds.extend(preds.cpu().numpy())
#             all_labels.extend(labels.cpu().numpy())

#     final_test_loss = total_loss / total
#     final_test_accuracy = correct / total
#     class_names = ['Healthy', 'Tumor']
    
#     print("\n" + "="*40)
#     print("      FINAL FEDERATED MODEL PERFORMANCE")
#     print("="*40)
#     print(f"  Accuracy on Combined Test Set: {final_test_accuracy * 100:.2f}%")
#     print(f"  Average Loss on Combined Test Set: {final_test_loss:.4f}")
#     print("="*40)
#     print("\n--- Detailed Classification Report ---")
#     print(classification_report(all_labels, all_preds, target_names=class_names))

#     # --- 8. UPLOAD FINAL RESULTS TO AZURE ---
#     print("--- [SERVER LOG] Uploading final results to Azure ML ---")
#     mlflow.log_metric("final_test_accuracy", final_test_accuracy)
#     mlflow.log_metric("final_test_loss", final_test_loss)
    
#     try:
#         mlflow.log_artifact("final_federated_model.pth", artifact_path="models")
#         mlflow.set_tag("status", "completed_with_test")
#         print("✅ Final model uploaded to Azure.")
#     except Exception as e:
#         print(f"⚠️ Could not log artifact: {e}")

#     print(f"✅ Simulation finished. All results logged to Azure ML Run ID: {run_id}")

import flwr as fl
import torch
import os
import mlflow
from collections import OrderedDict
from azure.identity import DefaultAzureCredential
from azure.ai.ml import MLClient
from models.hybrid_model import HybridModel 
from client import FlowerClient, load_client_data
from typing import List, Tuple, Dict, Union
from flwr.common import Metrics
from sklearn.metrics import classification_report
from torch.utils.data import DataLoader, ConcatDataset
from torchvision.datasets import ImageFolder
from torchvision import transforms

# --- 1. CONNECT TO AZURE ML ---
print("--- [SERVER LOG] ---")
print("🧠 Connecting to Azure ML Workspace...")
ml_client = MLClient.from_config(credential=DefaultAzureCredential())
mlflow_tracking_uri = ml_client.workspaces.get(ml_client.workspace_name).mlflow_tracking_uri
mlflow.set_tracking_uri(mlflow_tracking_uri)
print(f"✅ Connected to Azure ML: {ml_client.workspace_name}")

# --- 2. LOAD YOUR PRE-TRAINED "STAR PLAYER" MODEL ---
print("🧠 Loading pre-trained 4-class model (best_multiclass_model.pth)...")
model_architecture = HybridModel(num_classes=2) 
model_dict = model_architecture.state_dict()
pretrained_dict = torch.load("best_multiclass_model.pth", map_location="cpu")
pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict and model_dict[k].shape == v.shape}
model_dict.update(pretrained_dict)
model_architecture.load_state_dict(model_dict)
print(f"✅ Transferred {len(pretrained_dict)} layers. New 2-class model is ready.")

initial_parameters = fl.common.ndarrays_to_parameters(
    [val.cpu().numpy() for _, val in model_architecture.state_dict().items()]
)

# --- 3. DEFINE A METRICS AGGREGATION FUNCTION ---
server_round_g = 0 
def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    global server_round_g
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    losses = [num_examples * m["loss"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]
    
    if sum(examples) == 0:
        return {"accuracy": 0.0, "loss": 0.0} 
        
    avg_accuracy = sum(accuracies) / sum(examples)
    avg_loss = sum(losses) / sum(examples)
    
    print(f"--- [SERVER LOG] Global Val Accuracy: {avg_accuracy:.4f} | Global Val Loss: {avg_loss:.4f} ---")
    mlflow.log_metric("global_val_accuracy", avg_accuracy, step=server_round_g)
    mlflow.log_metric("global_val_loss", avg_loss, step=server_round_g)
    return {"accuracy": avg_accuracy, "loss": avg_loss}

# --- 4. CREATE A CUSTOM STRATEGY WITH AZURE LOGGING ---
class AzureFedAvg(fl.server.strategy.FedAvg):
    def aggregate_fit(self, server_round, results, failures):
        global server_round_g 
        server_round_g = server_round
        
        print(f"--- [SERVER LOG - ROUND {server_round}] ---")
        if failures:
            print(f"🚨 Failures reported: {failures}")
        
        agg_params, agg_metrics = super().aggregate_fit(server_round, results, failures)
        
        if agg_metrics:
            mlflow.log_metric("Round", server_round)
            mlflow.log_metric("Clients in Round", len(results))

        if agg_params is not None and server_round == 5: 
             print("🧠 Saving final aggregated model to 'final_federated_model.pth'...")
             nd_arrays = fl.common.parameters_to_ndarrays(agg_params)
             params_dict = zip(model_architecture.state_dict().keys(), nd_arrays)
             state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
             torch.save(state_dict, "final_federated_model.pth")
             
        return agg_params, agg_metrics

# --- 5. DEFINE THE CLIENT-SPAWNING FUNCTION ---
def client_fn(cid: str) -> fl.client.Client:
    print(f"--- [SERVER LOG] Spawning Client {cid} ---")
    client_data_path = f"D:\\FedXViT\\SplitData\\CLIENT_{cid}_DATA" 
    print(f"Client {cid} will load data from: {client_data_path}")

    client_model = HybridModel(num_classes=2)
    train_loader, val_loader = load_client_data(client_data_path) 
    
    return FlowerClient(client_model, train_loader, val_loader).to_client()

# --- 6. START THE MLFLOW EXPERIMENT RUN ---
with mlflow.start_run(run_name="FedXViT_Local_Sim_with_Eval") as run:
    run_id = run.info.run_id
    print(f"🔥 Starting MLflow Run: {run_id}")
    print(f"Follow progress in Azure ML Studio: {mlflow.get_tracking_uri()}")

    mlflow.log_params({
        "strategy": "FedAvg (Transfer Learning)",
        "num_clients": 3,
        "num_rounds": 5,
        "architecture": "Hybrid (EfficientNetV2 + ViT)"
    })

    strategy = AzureFedAvg(
        fraction_fit=1.0,
        min_fit_clients=3,
        min_available_clients=3,
        initial_parameters=initial_parameters,
        evaluate_metrics_aggregation_fn=weighted_average, 
    )
    
    simulation_success = False # Flag to track if simulation completes
    
    try:
        print("🚀 Starting local simulation on RTX 3060 (SEQUENTIAL MODE)...")
        fl.simulation.start_simulation(
            client_fn=client_fn,
            num_clients=3,
            config=fl.server.ServerConfig(num_rounds=5),
            strategy=strategy,
            # --- THIS IS THE FIX ---
            # This tells Ray to give 100% of the GPU to one client at a time.
            client_resources={"num_gpus": 1.0} 
            # --- END OF FIX ---
        )
        simulation_success = True # Mark as success
    
    except Exception as e:
        print(f"🚨 SIMULATION CRASHED: {e}")
        mlflow.set_tag("status", "crashed")
    
    # --- 7. AUTOMATIC FINAL TESTING (NOW SAFE) ---
    if simulation_success:
        print("\n🏁 Federated training complete. Starting final test on unseen data...")
        DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        final_model = HybridModel(num_classes=2)
        try:
            final_model.load_state_dict(torch.load("final_federated_model.pth"))
        except FileNotFoundError:
            print("🚨 ERROR: 'final_federated_model.pth' not found. Skipping final test.")
            mlflow.set_tag("status", "failed_to_save_model")
            exit()
            
        final_model.to(DEVICE)
        final_model.eval()

        test_transforms = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        client_0_test = ImageFolder(os.path.join(r"D:\FedXViT\SplitData", "CLIENT_0_DATA", "Testing"), transform=test_transforms)
        client_1_test = ImageFolder(os.path.join(r"D:\FedXViT\SplitData", "CLIENT_1_DATA", "Testing"), transform=test_transforms)
        client_2_test = ImageFolder(os.path.join(r"D:\FedXViT\SplitData", "CLIENT_2_DATA", "Testing"), transform=test_transforms)
        
        combined_test_dataset = ConcatDataset([client_0_test, client_1_test, client_2_test])
        test_loader = DataLoader(combined_test_dataset, batch_size=16, shuffle=False)
        print(f"✅ Combined test dataset created with {len(combined_test_dataset)} total images.")
        
        criterion = torch.nn.CrossEntropyLoss()
        total_loss, correct, total = 0.0, 0, 0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                outputs = final_model(images)
                loss = criterion(outputs, labels)
                total_loss += loss.item() * images.size(0)
                preds = outputs.argmax(dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        final_test_loss = total_loss / total
        final_test_accuracy = correct / total
        class_names = ['Healthy', 'Tumor']
        
        print("\n" + "="*40)
        print("      FINAL FEDERATED MODEL PERFORMANCE")
        print("="*40)
        print(f"  Accuracy on Combined Test Set: {final_test_accuracy * 100:.2f}%")
        print(f"  Average Loss on Combined Test Set: {final_test_loss:.4f}")
        print("="*40)
        print("\n--- Detailed Classification Report ---")
        print(classification_report(all_labels, all_preds, target_names=class_names))

        # --- 8. UPLOAD FINAL RESULTS TO AZURE ---
        print("--- [SERVER LOG] Uploading final results to Azure ML ---")
        mlflow.log_metric("final_test_accuracy", final_test_accuracy)
        mlflow.log_metric("final_test_loss", final_test_loss)
        
        try:
            mlflow.log_artifact("final_federated_model.pth", artifact_path="models")
            mlflow.set_tag("status", "completed_with_test")
            print("✅ Final model uploaded to Azure.")
        except Exception as e:
            print(f"⚠️ Could not log artifact: {e}")

        print(f"✅ Simulation finished. All results logged to Azure ML Run ID: {run_id}")
    
    else:
        print("🏁 Simulation failed. Final testing and artifact upload will be skipped.")