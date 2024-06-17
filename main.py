from torch.utils.data import DataLoader
from torchvision import transforms
from datacustom import Sataset
from autoencoder import Autoencoder
from train import training


# Hyperparameters
num_epochs = 20
learning_rate = 0.001
batch_size = 16
sigma = 0.1
img_path = 'c:\\example'         


dataset = Dataset(
        img_path=img_path, sigma=sigma, transforms=transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor()
        ]))
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)


if __name__ == '__main__':
    training(Autoencoder, dataloader, learning_rate, num_epochs)

