import torch
import torch.nn as nn

from torchvision import models

# Feature dimension configuration
teacher_dims = [256, 512, 1024, 2048]  # Output dimensions of each layer in ResNet50
# teacher_dims = [64, 192, 384, 256]      # Output dimensions of each layer in AlexNet
student_dims = [24, 32, 96, 1280]  # Output dimensions of each layer in MobileNetV2

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class FeatureProjector(nn.Module):
    """Feature Dimension Projector"""

    def __init__(self, compression_factor=0.5):
        super().__init__()

        # Adjust projection layer dimensions based on compression factor
        self.compressed_teacher_dims = [int(dim * compression_factor) for dim in teacher_dims]

        self.proj_layers = nn.ModuleList([
            nn.Linear(student_dims[0], self.compressed_teacher_dims[0]),
            nn.Linear(student_dims[1], self.compressed_teacher_dims[1]),
            nn.Linear(student_dims[2], self.compressed_teacher_dims[2]),
            nn.Linear(student_dims[3], self.compressed_teacher_dims[3]),
        ])

    def forward(self, features):
        """Project student features into compressed teacher feature space"""
        projected_features = []
        for i, (proj, feat) in enumerate(zip(self.proj_layers, features)):
            if feat.size(1) != student_dims[i]:
                raise ValueError(f"Feature dimension mismatch at layer {i}: expected {student_dims[i]}, but got {feat.size(1)}")
            # Perform the dimension compression projection
            projected_features.append(proj(feat))
        return projected_features

# Initialize model
def build_model(model_type, num_classes=38):
    model = None
    if model_type == 'teacher':
        model = models.resnet50(pretrained=True)
        model.fc = nn.Linear(2048, num_classes)
        # model = models.alexnet(pretrained=True)
        # model.classifier[6] = nn.Linear(4096, num_classes)  # Classification layer for AlexNet
    elif model_type == 'student':
        model = models.mobilenet_v2(pretrained=True)
        model.classifier[1] = nn.Linear(1280, num_classes)
    return model.to(device)

def get_features(model, input_data):
    """Unified feature extraction function"""
    features = []

    # Define hook function
    def hook_fn(module, input, output):
        pooled = nn.AdaptiveAvgPool2d(1)(output)
        features.append(pooled.flatten(1))

    # Register hooks
    hooks = []
    if isinstance(model, models.ResNet):
        layers = [model.layer1, model.layer2, model.layer3, model.layer4]
    elif isinstance(model, models.MobileNetV2):
        layers = [
            model.features[3],  # Corresponds to ResNet's layer1
            model.features[6],  # Corresponds to ResNet's layer2
            model.features[13],  # Corresponds to ResNet's layer3
            model.features[-1]  # Corresponds to ResNet's layer4
        ]
    # if isinstance(model, models.AlexNet):
    #     layers = [model.features[0], model.features[3], model.features[6], model.features[10]]  # Feature layers for AlexNet
    # elif isinstance(model, models.MobileNetV2):
    #     layers = [
    #         model.features[3],  # Corresponds to AlexNet's layer1
    #         model.features[6],  # Corresponds to AlexNet's layer2
    #         model.features[13],  # Corresponds to AlexNet's layer3
    #         model.features[-1]  # Corresponds to AlexNet's layer4
    #     ]

    for layer in layers:
        hook = layer.register_forward_hook(hook_fn)
        hooks.append(hook)

    # Forward pass
    with torch.no_grad():
        model(input_data)

    # Remove hooks
    for hook in hooks:
        hook.remove()

    return features