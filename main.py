from torch.utils.data import DataLoader
from starter_code.utils import load_case
import numpy as np
from dataset import KitsDataset
import torch
import torch.nn as nn
import torch.optim as optim


class UNET(nn.Module):
    def __init__(
            self, in_channels=1, out_channels=1, features=[64, 128, 256, 512],
    ):
        super(UNET, self).__init__()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Down part of UNET
        for feature in features:
            self.downs.append(nn.Sequential(
                nn.Conv2d(in_channels, feature, kernel_size=(3, 3), stride=1, padding=1, bias=False),
                nn.BatchNorm2d(feature),
                nn.ReLU(inplace=True),
                nn.Conv2d(feature, feature, kernel_size=(3, 3), stride=1, padding=1, bias=False),
                nn.BatchNorm2d(feature),
                nn.ReLU(inplace=True))
            )
            in_channels = feature

        # Up part of UNET
        for feature in reversed(features):
            self.ups.append(
                nn.ConvTranspose2d(
                    feature*2, feature, kernel_size=2, stride=2,
                )
            )
            self.ups.append(nn.Sequential(
                nn.Conv2d(feature*2, feature, kernel_size=(3, 3), stride=1, padding=1, bias=False),
                nn.BatchNorm2d(feature),
                nn.ReLU(inplace=True),
                nn.Conv2d(feature, feature, kernel_size=(3, 3), stride=1, padding=1, bias=False),
                nn.BatchNorm2d(feature),
                nn.ReLU(inplace=True))
            )

        self.bottleneck = nn.Sequential(
            nn.Conv2d(features[-1], features[-1]*2, kernel_size=(3, 3), stride=1, padding=1, bias=False),
            nn.BatchNorm2d(features[-1]*2),
            nn.ReLU(inplace=True),
            nn.Conv2d(features[-1]*2, features[-1]*2, kernel_size=(3, 3), stride=1, padding=1, bias=False),
            nn.BatchNorm2d(features[-1]*2),
            nn.ReLU(inplace=True)
        )
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        skip_connections = []

        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]

        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            skip_connection = skip_connections[idx//2]

            concat_skip = torch.cat((skip_connection, x), dim=1)
            x = self.ups[idx+1](concat_skip)

        return self.final_conv(x)


#Hyperparamters
learning_rate = 0.001
device = "cuda" if torch.cuda.is_available() else "cpu"
batch_size = 4
epochs = 1

def train_fn(model, optimizer, loss_fn, loader):

    for batch_idx, (data, targets) in enumerate(loader):
        data = data.float().to(device=device)
        data = torch.unsqueeze(data, 1)
        targets = targets.float().to(device=device)
        targets = torch.unsqueeze(targets, 1)

        # forward
        predictions = model(data)
        loss = loss_fn(predictions, targets)

        # backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


def model_evaluation(loader):
    model.eval()
    total_pixels = 0
    total_correct = 0

    with torch.no_grad():
        for data, target in loader:
            data = data.float().to(device=device)
            data = torch.unsqueeze(data, 1)

            predictions = model(data)
            predictions = torch.squeeze(predictions, 1)

            total_pixels = total_pixels + torch.numel(predictions)
            predictions = torch.round(predictions).cpu().numpy().clip(min=0, max=2)
            true_value = target.cpu().numpy()

            total_correct = total_correct + np.sum(predictions == true_value)

    print(f"accuracy is {(total_correct/total_pixels)*100}")


if __name__ == "__main__":

    model = UNET(in_channels=1, out_channels=1).to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(epochs):
        for idx, patient in enumerate(range(20)):
            print("training case" + str(idx))
            volume, segmentation = load_case(0)

            train_dataset = KitsDataset(volume, segmentation)
            train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

            train_fn(model, optimizer, loss_fn, train_loader)

    valid_volume, valid_segmentation = load_case(51)
    valid_dataset = KitsDataset(valid_volume, valid_segmentation)
    valid_loader = DataLoader(dataset=valid_dataset, batch_size=batch_size, shuffle=False)

    model_evaluation(valid_loader)

