import torch
import torch.nn as nn

from torchvision import models

def compute_irm_matrix(features_list, n_samples):
    """
    Compute the normalized IRM matrix. This matrix represents the Euclidean distance between features at each layer.

    Inputs:
        features_list: List[torch.Tensor], a list containing features from multiple layers, where each element is a tensor of shape [batch_size, feature_dim].
                        - If the features are from the teacher model, it represents the teacher's features per layer [num_layers, batch_size, feat_dim]
                        - If the features are from the student model, it represents the student's features per layer [num_layers, batch_size, feat_dim]
        n_samples: int, the number of samples in the current batch, i.e., batch_size

    Outputs:
        irm_matrix: torch.Tensor, a tensor of shape [num_layers, n_samples, n_samples], the IRM matrix.
                    Each layer's IRM matrix represents the squared Euclidean distance between each pair of samples at that layer.
    """
    n_layers = len(features_list)
    irm_matrix = torch.zeros((n_layers, n_samples, n_samples), device=features_list[0].device)

    for layer_idx in range(n_layers):
        features = features_list[layer_idx]  # Features of the current layer [batch, feat_dim]

        # Normalize the features to make feature distributions across different layers similar
        features = nn.functional.normalize(features, dim=1)

        # Compute squared Euclidean distance between all pairs of samples, avoiding explicit nested loops
        # Calculate squared distance between samples:
        # (x_i - x_j)^2 = ||x_i||^2 + ||x_j||^2 - 2 * <x_i, x_j>
        # Calculate the norm (i.e., ||x_i||^2) for each sample
        norm = torch.sum(features ** 2, dim=1, keepdim=True)  # [batch_size, 1]

        # Compute inner product (i.e., <x_i, x_j>)
        dist_sq = norm + norm.transpose(0, 1) - 2 * torch.matmul(features, features.T)

        # Store the squared distance matrix (Note: We already have the squared Euclidean distance)
        irm_matrix[layer_idx] = dist_sq

    return irm_matrix

def irm_loss(H_T, H_S):
    """
    Compute the IRM loss: The formula is ∑ || H_T - H_S ||_2^2
    Inputs:
        H_T: Teacher model's IRM matrix [batch_size, num_layers, num_layers]
        H_S: Student model's IRM matrix [batch_size, num_layers, num_layers]
    Outputs:
        IRM loss value
    """
    return torch.mean((H_T - H_S) ** 2)

def irm_t_loss(t_features, s_features):
    """
    Compute the IRM-t loss: The formula is ||(D_g^T - D_h^T) - (D_g^S - D_h^S)||_2^2
    Inputs:
        t_features: Features extracted by the teacher model [num_layers, batch_size, feat_dim]
        s_features: Features extracted by the student model [num_layers, batch_size, feat_dim]
    Outputs:
        IRM-t loss value
    """
    num_layers = len(t_features)
    total_loss = 0.0

    # Iterate through all combinations of layers and compute the differences in feature flow across layers
    for g in range(num_layers):
        for h in range(num_layers):
            if g != h:  # Only compute feature flow differences between different layers
                t_flow = t_features[g] - t_features[h]
                s_flow = s_features[g] - s_features[h]
                total_loss += torch.mean((t_flow - s_flow) ** 2)

    # return total_loss / (num_layers * (num_layers - 1))
    return total_loss / num_layers
