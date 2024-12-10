from torch.utils.data import Dataset, DataLoader

from model import *


def train():
    class CustomDataset(Dataset):
        def __init__(self):
            self.n_samples = len(X_train)
            self.x_data = X_train  # Features (bag of words)
            self.y_data = Y_train  # Labels (numerical indices for tags)

        def __getitem__(self, index):
            return self.x_data[index], self.y_data[index]

        def __len__(self):
            return self.n_samples

    print('Training...')

    dataset = CustomDataset()
    train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    loss: nn.CrossEntropyLoss() | None = None  # Placeholder to store the loss during training

    # Training loop over the specified number of epochs
    for epoch in range(EPOCHS):
        for batch, (words, labels) in enumerate(train_loader):
            # Convert features and labels to the correct type and move to the computation device
            words = words.float().to(device)  # Convert features to float tensor
            labels = labels.to(device)  # Move labels to the same device

            # Forward pass: compute predictions from the model
            output = model(words)

            # Compute the loss between predictions and true labels
            loss = loss_fn(output, labels)

            # Backpropagation: clear previous gradients, compute new ones, and update parameters
            optimizer.zero_grad()  # Clear gradients
            loss.backward()  # Compute gradients
            optimizer.step()  # Update model parameters

        if epoch % 5 == 4:
            print(f'Epoch [{epoch + 1}/{EPOCHS}] Loss: {loss.item():.10f}')

    # Save the trained model
    torch.save(model.state_dict(), 'model.pth')
    print(f'Finished training with final loss {loss.item():.10f}')


if __name__ == '__main__':
    # Start the training process when the script is executed
    train()
