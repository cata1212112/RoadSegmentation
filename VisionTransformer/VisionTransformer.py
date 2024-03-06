import torch
import torch.nn as nn
import torch.nn.functional as F

height = 100
width = 100
color_channels = 3
patch_size = 5

number_of_patches = int(height * width / patch_size ** 2)

EMBEDDING_SIZE = 128
NUMBER_OF_HEADS = 1
HIDDEN_SIZE = 256
TE_NUM_LAYERS = 3


class PatchEmbedding(nn.Module):
    def __init__(self, in_channels=3, patch_size=16, embedding_dim=768):
        super().__init__()

        self.in_channels = 3
        self.patch_size = 3
        self.embedding_dim = EMBEDDING_SIZE

        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=128, kernel_size=patch_size,
                              stride=2)
        self.conv12 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=patch_size,
                              stride=1, padding=2)

        self.conv2 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3,
                               stride=2)
        self.conv22 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3,
                               stride=1, padding=1)

        self.conv3 = nn.Conv2d(in_channels=256, out_channels=embedding_dim, kernel_size=3,
                               stride=1)

        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=1)
        self.flatten = nn.Flatten(start_dim=2, end_dim=3)
        self.linear = nn.Linear(embedding_dim, embedding_dim)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)

        x = self.conv12(x)
        x = F.relu(x)
        x = self.max_pool(x)

        x = self.conv2(x)
        x = F.relu(x)
        x = self.conv22(x)
        x = F.relu(x)
        x = self.max_pool(x)
        x = self.conv3(x)
        x = F.relu(x)

        x = self.flatten(x)

        x = x.permute(0, 2, 1)

        x = self.linear(x)
        return x
        # return x.permute(0, 2, 1)


class VisionTransformer(nn.Module):
    def __init__(self):
        super().__init__()

        self.patch_embedding = PatchEmbedding(patch_size=patch_size, embedding_dim=EMBEDDING_SIZE)
        self.cls_token = nn.Parameter(torch.rand(1, 1, EMBEDDING_SIZE))
        self.dist_token = nn.Parameter(torch.rand(1, 1, EMBEDDING_SIZE))
        self.positional_embedding = nn.Parameter(torch.rand(1, 2 + number_of_patches, EMBEDDING_SIZE))
        self.embedding_dropout = nn.Dropout(p=0.1)
        self.transformer = nn.TransformerEncoder(encoder_layer=nn.TransformerEncoderLayer(d_model=EMBEDDING_SIZE, nhead=NUMBER_OF_HEADS, dim_feedforward=HIDDEN_SIZE, activation="relu", dropout=0.1, batch_first=True, norm_first=True), num_layers=TE_NUM_LAYERS)
        self.norm = nn.LayerNorm(EMBEDDING_SIZE)

    def forward(self, x):
        x = self.patch_embedding(x)
        cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)
        dist_tokens = self.dist_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, dist_tokens, x), dim=1)
        x = x + self.positional_embedding
        x = self.embedding_dropout(x)
        x = self.transformer(x)
        x = self.norm(x)
        return x