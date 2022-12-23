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
        self.upConv = nn.ModuleList()
        self.downConv = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Down part of the UNET, double convolution each time
        for feature in features:
            self.downConv.append(nn.Sequential(
                nn.Conv2d(in_channels, feature, kernel_size=(3, 3), stride=1, padding=1, bias=False),
                nn.BatchNorm2d(feature),
                nn.ReLU(inplace=True),
                nn.Conv2d(feature, feature, kernel_size=(3, 3), stride=1, padding=1),
                nn.ReLU(inplace=True))
            )
            in_channels = feature

        # Up part of UNET, convolutional transpose for upscaling back to the orifinal resolution
        for feature in reversed(features):
            self.upConv.append(
                nn.ConvTranspose2d(
                    feature*2, feature, kernel_size=2, stride=2,
                )
            )
            self.upConv.append(nn.Sequential(
                nn.Conv2d(feature*2, feature, kernel_size=(3, 3), stride=1, padding=1, bias=False),
                nn.BatchNorm2d(feature),
                nn.ReLU(inplace=True),
                nn.Conv2d(feature, feature, kernel_size=(3, 3), stride=1, padding=1),
                nn.ReLU(inplace=True))
            )

        #bottom part of the u net
        self.bottom = nn.Sequential(
            nn.Conv2d(features[-1], features[-1]*2, kernel_size=(3, 3), stride=1, padding=1, bias=False),
            nn.BatchNorm2d(features[-1]*2),
            nn.ReLU(inplace=True),
            nn.Conv2d(features[-1]*2, features[-1]*2, kernel_size=(3, 3), stride=1, padding=1),
            nn.ReLU(inplace=True)
        )

        #last convolution
        self.last_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        skip_connections = []

        #left side of the u net
        for down in self.downConv:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        #bottom part
        x = self.bottom(x)

        #reverse the skip connections so that the last is now first
        skip_connections = skip_connections[::-1]

        #right side of the u net
        up_length = len(self.upConv)
        for idx in range(0, up_length, 2):
            x = self.upConv[idx](x)
            skip_connection = skip_connections[idx//2]

            concat_skip = torch.cat((skip_connection, x), dim=1)
            x = self.upConv[idx+1](concat_skip)

        return self.last_conv(x)


#Hyperparamters
learning_rate = 0.001
device = "cuda" if torch.cuda.is_available() else "cpu"
batch_size = 4
epochs = 1

def train_fn(model, optimizer, loss_fn, loader):

    for batch_idx, (data, targets) in enumerate(loader):
        data = data.float().to(device=device)
        data = torch.unsqueeze(data, 1)  #fixing data shape
        targets = targets.float().to(device=device)
        targets = torch.unsqueeze(targets, 1) #fixing target shape

        # forward
        predictions = model(data)
        loss = loss_fn(predictions, targets)

        # backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


def model_evaluation(loader):
    #accuracy test
    model.eval()
    total_pixels = 0
    total_correct = 0

    with torch.no_grad():
        for data, target in loader:
            data = data.float().to(device=device)
            data = torch.unsqueeze(data, 1) #fixing data shape

            predictions = model(data)
            predictions = torch.squeeze(predictions, 1)

            total_pixels = total_pixels + torch.numel(predictions)
            predictions = torch.round(predictions).cpu().numpy().clip(min=0, max=2) #converting tensor to numpy
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
            #loading images of case idx
            volume, segmentation = load_case(0)

            train_dataset = KitsDataset(volume, segmentation)
            train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

            train_fn(model, optimizer, loss_fn, train_loader)

    #loading validation test patient
    valid_volume, valid_segmentation = load_case(51)
    valid_dataset = KitsDataset(valid_volume, valid_segmentation)
    valid_loader = DataLoader(dataset=valid_dataset, batch_size=batch_size, shuffle=False)

    model_evaluation(valid_loader)

