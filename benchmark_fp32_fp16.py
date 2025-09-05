import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch.amp import GradScaler, autocast
import time
import nvtx

# Transform: normalize and convert to tensor
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

# Load MNIST
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=2, pin_memory=True)


# Define a Simple CNN Model
class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(64 * 6 * 6, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        return self.net(x)


def train_fp32(model, train_loader, device):
    print("\n🧠 Starting FP32 training...")
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    model.train()
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    start_event.record()

    for epoch in range(1):  # Keep it short for benchmarking
        with nvtx.annotate(f"Epoch {epoch} - FP32", color="blue"):
            for batch_idx, (data, target) in enumerate(train_loader):
                with nvtx.annotate("Data Prep", color="purple"):
                    data, target = data.to(device, non_blocking=True), target.to(device, non_blocking=True)

                optimizer.zero_grad()

                with nvtx.annotate("Forward Pass", color="green"):
                    output = model(data)
                    loss = criterion(output, target)

                with nvtx.annotate("Backward Pass", color="red"):
                    loss.backward()

                with nvtx.annotate("Optimizer Step", color="yellow"):
                    optimizer.step()

    end_event.record()
    torch.cuda.synchronize()

    elapsed_time = start_event.elapsed_time(end_event)  # in milliseconds
    memory_used = torch.cuda.max_memory_allocated(device) / 1e6  # in MB

    print(f"⏱️ FP32 Epoch Time: {elapsed_time:.2f} ms")
    print(f"💾 FP32 Max Memory Used: {memory_used:.2f} MB")



def train_fp16(model, train_loader, device):
    print("\n⚡ Starting FP16 (Mixed Precision) training...")
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    scaler = GradScaler(device='cuda')

    model.train()
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    start_event.record()

    for epoch in range(1):  # Keep it short for benchmarking
        with nvtx.annotate(f"Epoch {epoch} - FP16", color="blue"):
            for batch_idx, (data, target) in enumerate(train_loader):
                with nvtx.annotate("Data Prep", color="purple"):
                    data, target = data.to(device, non_blocking=True), target.to(device, non_blocking=True)

                optimizer.zero_grad()

                with nvtx.annotate("Forward Pass", color="green"):
                    with autocast(device_type='cuda'):
                        output = model(data)
                        loss = criterion(output, target)

                with nvtx.annotate("Backward Pass", color="red"):
                    scaler.scale(loss).backward()

                with nvtx.annotate("Optimizer Step", color="yellow"):
                    scaler.step(optimizer)
                    scaler.update()

    end_event.record()
    torch.cuda.synchronize()

    elapsed_time = start_event.elapsed_time(end_event)  # in milliseconds
    memory_used = torch.cuda.max_memory_allocated(device) / 1e6  # in MB

    print(f"⏱️ FP16 Epoch Time: {elapsed_time:.2f} ms")
    print(f"💾 FP16 Max Memory Used: {memory_used:.2f} MB")



if __name__ == "__main__":
    # Instantiate model fresh for each run
    model_fp32 = SimpleCNN()
    train_fp32(model_fp32, train_loader, device)

    # Reset CUDA memory stats before FP16 run
    torch.cuda.reset_peak_memory_stats(device)

    model_fp16 = SimpleCNN()
    train_fp16(model_fp16, train_loader, device)