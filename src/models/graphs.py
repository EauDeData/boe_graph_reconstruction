import torch
from torch_geometric.nn import GAT
import torch.nn as nn
class GraphConv(nn.Module):
    def __init__(self, text_features_size, visual_feature_size,
                 in_channels, hidden_channels, num_layers, out_channels, dropout=.1, num_categories=2, device='cuda'):
        super().__init__()
        in_channels = in_channels + in_channels % 2
        print('Graph model loaded with', num_categories, 'output size.')
        self.upsale_text = nn.Linear(text_features_size, in_channels // 2)
        self.upscale_img = nn.Linear(visual_feature_size, in_channels // 2)
        self.regression_angle = nn.Linear(out_channels, num_categories)
        self.regression_distance = nn.Linear(out_channels, num_categories)

        self.activation = nn.GELU()
        self.gat = GAT(in_channels + 3, hidden_channels, num_layers, out_channels, dropout=dropout)

        self.feature_size = in_channels + 3
        # self.output_learnt_emb = torch.nn.Embedding(1, in_channels)
        self.device=device
    @staticmethod
    def dot_product_matrix(tensor):
        # Assuming tensor has shape (batch_size, feature_size)
        # Reshape tensor to (batch_size, 1, feature_size)
        tensor = tensor.unsqueeze(1)

        # Compute dot product by matrix multiplication
        dot_product = torch.bmm(tensor, tensor.transpose(1, 2))

        return dot_product.squeeze()  # Squeeze the tensor to remove the extra dimension

    def forward(self, image_features, text_features, content_indices, gt_indices, edges, ars):


        node_features = torch.zeros((max(content_indices) + 1, self.feature_size), device=image_features.device)
        # output_features = self.output_learnt_emb(torch.zeros(len(gt_indices),
        #                                                     device=image_features.device,
        #                                                     dtype=torch.int64) )
        image_features = self.upscale_img(image_features)
        text_features = self.upsale_text(text_features)
        input_features = torch.cat((image_features, text_features, ars), 1)
        # node_features[gt_indices, ] = output_features
        node_features[content_indices] = input_features

        node_features = self.activation(node_features)

        output_features_node_features = self.activation(self.gat(
            x=node_features,
            edge_index=edges.T.to(torch.int64).to(image_features.device)
        ))

        angles = self.regression_angle(output_features_node_features)

        angles_left = angles[edges[:, 0], :][:, None, :]
        angles_right = angles[edges[:, 1], :][:, :, None]
        angles_prediction = torch.bmm(angles_left, angles_right).squeeze()[:, None]

        distances = self.regression_distance(output_features_node_features)

        dist_left = distances[edges[:, 0], :][:, None, :]
        dist_right = distances[edges[:, 1], :][:, :, None]
        dist_prediction = torch.bmm(dist_left, dist_right).squeeze()[:, None]

        prediction = torch.cat((angles_prediction, dist_prediction), dim=-1)

        return {
            'image_features': output_features_node_features,
            'regressed_features': prediction
        }

