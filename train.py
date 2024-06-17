import torch.nn as nn
import torch.optim as optim


def training(Autoencoder, dataloader, learning_rate, num_epochs):
    model = Autoencoder().cuda()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Training Loop
    for epoch in range(num_epochs):
        for data in dataloader:
            noisy_imgs, clean_imgs = data
            noisy_imgs = noisy_imgs.cuda()
            clean_imgs = clean_imgs.cuda()
            outputs = model(noisy_imgs)
            loss = criterion(outputs, clean_imgs)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

    print('Training complete.')