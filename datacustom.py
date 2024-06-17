import os
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image


class Dataset(Dataset):
    def __init__(self, img_path, sigma, transforms=None):
        self.img_path = img_path
        self.transforms = transforms
        self.imgs = os.listdir(img_path)
        self.sigma = sigma

    def __len__(self):
        return len(self.imgs)
    
    def __getitem__(self, idx):
        path_file = os.path.join(self.img_path, self.imgs[idx])
        clean = Image.open(path_file).convert("RGB")
        
        transform = transforms.Compose([
            transforms.Resize((256, 256)), 
            transforms.ToTensor()  
        ])

        clean = transform(clean)
        noise = torch.randn(clean.shape) * self.sigma
        noisy = clean + noise
        
        return noisy, clean