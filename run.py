import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from utils import *
from net import *
import random
import numpy as np

torch.manual_seed(2)
random.seed(2)
np.random.seed(2)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(2)

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Data preprocessing
transform = transforms.Compose([
    transforms.Resize(128),
    transforms.CenterCrop(128),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Custom dataset
class PlantDiseaseDataset(datasets.ImageFolder):
    def __init__(self, root, transform=None):
        super().__init__(root, transform=transform)

# Train the teacher model
def train_teacher():
    teacher = build_model('teacher')
    train_set = PlantDiseaseDataset('dataTrain', transform=transform)
    train_loader = DataLoader(train_set, batch_size=512, shuffle=True)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(teacher.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4)

    for epoch in range(50):
        teacher.train()
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = teacher(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        print(f'Epoch {epoch + 1}/50 | Loss: {loss.item():.4f}')

    torch.save(teacher.state_dict(), 'teacher.pth')


# Knowledge distillation
def distill():
    # Initialize models
    teacher = build_model('teacher').eval()
    teacher.load_state_dict(torch.load('teacher.pth'))
    student = build_model('student')
    projector = FeatureProjector(compression_factor=0.5).to(device)

    # Optimizer
    optimizer = optim.SGD([
        {'params': student.parameters()},
        {'params': projector.parameters()}
    ], lr=0.1, momentum=0.9, weight_decay=1e-4)

    # Data loading
    train_loader = DataLoader(PlantDiseaseDataset('dataTrain', transform=transform),
                              batch_size=512, shuffle=True)

    # Training loop
    for epoch in range(50):
        student.train()
        total_loss = 0.0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            batch_size = images.size(0)

            # Forward pass
            with torch.no_grad():
                t_features = get_features(teacher, images)
                t_logits = teacher(images)
            s_features = get_features(student, images)
            s_logits = student(images)

            # Project student features
            try:
                s_features = projector(s_features)
            except ValueError as e:
                print(f"Feature dimension error: {e}")
                print("Actual feature dimensions:", [f.shape for f in s_features])
                raise

            # Compute the IRM matrix
            t_irm = compute_irm_matrix(t_features, batch_size)
            s_irm = compute_irm_matrix(s_features, batch_size)

            # Compute IRM loss
            L_irm = sum(irm_loss(t_irm[i], s_irm[i]) for i in range(len(t_irm))) / len(t_irm)

            # Compute IRM-t loss
            L_irmt = irm_t_loss(
                torch.stack([f.mean(dim=1) for f in t_features], dim=1),
                torch.stack([f.mean(dim=1) for f in s_features], dim=1)
            )

            # Compute classification loss
            L_ce = nn.CrossEntropyLoss()(student(images), labels)

            # Compute Softmax loss
            t_probs = nn.functional.softmax(t_logits / 2.0, dim=1)
            s_probs = nn.functional.softmax(s_logits / 2.0, dim=1)
            L_logits = torch.mean((t_probs - s_probs) ** 2)

            # Combine the loss functions
            alpha = 0.5
            beta = 0.3
            gamma = 0.2

            loss = alpha * L_irmt + (1 - alpha) * L_irm + beta * L_logits + gamma * L_ce

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f'Epoch {epoch + 1}/50 | Loss: {total_loss / len(train_loader):.4f}')

    torch.save(student.state_dict(), 'student.pth')


# Evaluation function
def evaluate(model_path, model_type):
    model = build_model(model_type)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    test_set = PlantDiseaseDataset('dataTest', transform=transform)
    test_loader = DataLoader(test_set, batch_size=512)

    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f'Accuracy: {100 * correct / total:.2f}%')


if __name__ == '__main__':
    print('=== Training Teacher ===')
    train_teacher()

    print('\n=== Distilling ===')
    distill()

    print('\n=== Evaluation ===')
    print('Teacher:')
    evaluate('teacher.pth', 'teacher')
    print('Student:')
    evaluate('student.pth', 'student')