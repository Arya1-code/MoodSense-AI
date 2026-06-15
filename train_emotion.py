import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torchvision.models import (
    efficientnet_b0,
    EfficientNet_B0_Weights
)

# =====================================================
# DEVICE
# =====================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Using:", device)

# =====================================================
# TRANSFORMS
# =====================================================

train_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
])

test_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# =====================================================
# DATASETS
# =====================================================

train_dataset = datasets.ImageFolder(
    "datasets/raw/fer2013/train",
    transform=train_transform
)

test_dataset = datasets.ImageFolder(
    "datasets/raw/fer2013/test",
    transform=test_transform
)

# =====================================================
# DATALOADERS
# =====================================================

train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True,
    num_workers=0,
    pin_memory=True
)

test_loader = DataLoader(
    test_dataset,
    batch_size=32,
    num_workers=0,
    pin_memory=True
)

print("Train:", len(train_dataset))
print("Test :", len(test_dataset))
print("Classes:", train_dataset.classes)

# =====================================================
# MODEL
# =====================================================

model = efficientnet_b0(
    weights=EfficientNet_B0_Weights.DEFAULT
)

model.classifier[1] = nn.Linear(
    model.classifier[1].in_features,
    7
)

model = model.to(device)

# =====================================================
# TEST FORWARD PASS
# =====================================================

images, labels = next(iter(train_loader))

images = images.to(device)

outputs = model(images)

print("Output Shape:", outputs.shape)

# =====================================================
# LOSS + OPTIMIZER
# =====================================================

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=1e-4
)

# =====================================================
# TRAINING
# =====================================================

num_epochs = 1

for epoch in range(num_epochs):

    model.train()

    running_loss = 0
    correct = 0
    total = 0

    for images, labels in train_loader:

        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(outputs, labels)

        loss.backward()

        optimizer.step()

        running_loss += loss.item()

        _, predicted = torch.max(outputs, 1)

        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total

    print(
        f"Epoch [{epoch+1}/{num_epochs}] "
        f"Loss: {running_loss/len(train_loader):.4f} "
        f"Accuracy: {accuracy:.2f}%"
    )

# =====================================================
# SAVE MODEL
# =====================================================

torch.save(
    model.state_dict(),
    "models/emotion_efficientnet.pth"
)

print("Model Saved!")