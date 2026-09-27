import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as T
from model import GeoSpecSRNet, SobelEdgeLoss

class SyntheticSatelliteDataset(Dataset):
    """
    Dataset loader for satellite super-resolution pairs.
    """
    def __init__(self, data_dir, scale_factor=4):
        self.data_dir = data_dir
        self.scale_factor = scale_factor
        self.transform = T.ToTensor()

    def __len__(self):
        return 50 # Paired dataset sample size

    def __getitem__(self, idx):
        # Generates multi-scale RGB synthetic satellite tensor pair
        hr = torch.rand(3, 256, 256)
        lr = F.interpolate(hr.unsqueeze(0), scale_factor=1.0/self.scale_factor, mode='bilinear').squeeze(0)
        return lr, hr

def train_geospecsr(epochs=2, batch_size=4, lr=1e-4):
    device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
    print(f"Training GeoSpecSR model on device: {device}")
    
    model = GeoSpecSRNet(in_channels=3, out_channels=3, num_features=56, num_blocks=5, scale_factor=4).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    l1_loss = nn.L1Loss()
    edge_loss_fn = SobelEdgeLoss().to(device)

    dataset = SyntheticSatelliteDataset("data/samples")
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        for lr_img, hr_img in loader:
            lr_img, hr_img = lr_img.to(device), hr_img.to(device)
            
            optimizer.zero_grad()
            sr_img = model(lr_img)
            
            # Reconstruction L1 Loss
            loss_pixel = l1_loss(sr_img, hr_img)
            
            # Sobel Geometry Loss
            loss_edge = l1_loss(edge_loss_fn(sr_img), edge_loss_fn(hr_img))
            
            # Combined Loss
            loss = loss_pixel + 0.2 * loss_edge
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        print(f"Epoch [{epoch+1}/{epochs}] Loss: {total_loss/len(loader):.4f}")

    torch.save(model.state_dict(), "geospecsr_trained.pth")
    print("Model training complete! Saved to geospecsr_trained.pth")

if __name__ == "__main__":
    train_geospecsr()
