import torch
from train import train_one_epoch, validate  # reuse existing functions

class FLClient:
    def __init__(self, model, train_loader, val_loader, device, criterion, optimizer):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.criterion = criterion
        self.optimizer = optimizer

    def train_local(self, epochs=1):
        self.model.to(self.device)
        for _ in range(epochs):
            train_one_epoch(self.model, self.train_loader, self.criterion, self.optimizer)
        val_loss, val_acc = validate(self.model, self.val_loader, self.criterion)
        return self.model.state_dict(), val_loss, val_acc
