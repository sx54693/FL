import flwr as fl
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from model_def import PneumoniaCNN

DEVICE = torch.device("cpu")

def load_data(client_id):
    try:
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])
        dataset = datasets.ImageFolder(f"federated_data/client_{client_id}/train", transform=transform)
        loader = DataLoader(dataset, batch_size=8, shuffle=True)
        return loader
    except Exception as e:
        print(f"❌ Data loading failed: {e}")
        exit()

class FLClient(fl.client.NumPyClient):
    def __init__(self, cid):
        print("🛠️ FLClient initialized")
        self.model = PneumoniaCNN().to(DEVICE)
        self.trainloader = load_data(cid)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)

    def get_parameters(self, config):
        print("📦 Sending model parameters to server")
        return [val.cpu().numpy() for val in self.model.state_dict().values()]

    def fit(self, parameters, config):
        print("🔥 Starting local training")
        self.model.train()
        for param, val in zip(self.model.state_dict().values(), parameters):
            param.copy_(torch.tensor(val))

        for images, labels in self.trainloader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()

        torch.save(self.model.state_dict(), "model.pth")
        print("✅ model.pth saved after training")
        return self.get_parameters(config={}), len(self.trainloader.dataset), {}

    def evaluate(self, parameters, config):
        return 0.0, len(self.trainloader.dataset), {}

if __name__ == "__main__":
    import sys
    client_id = sys.argv[1] if len(sys.argv) > 1 else "1"
    print("✅ client.py started")
    fl.client.start_numpy_client(server_address="localhost:8081", client=FLClient(client_id))
