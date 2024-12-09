from torch.utils.data import Dataset, DataLoader

from model import *


def train():
    class CustomDataset(Dataset):
        def __init__(self):
            self.n_samples = len(X_train)
            self.x_data = X_train
            self.y_data = Y_train

        def __getitem__(self, index):
            return self.x_data[index], self.y_data[index]

        def __len__(self):
            return self.n_samples

    print('Training...')
    dataset = CustomDataset()
    train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    loss = None

    for epoch in range(EPOCHS):
        for batch, (words, labels) in enumerate(train_loader):
            words = words.float().to(device)
            labels = labels.to(device)

            output = model(words)
            loss = loss_fn(output, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if epoch % 5 == 4:
            print(f'Epoch [{epoch + 1}/{EPOCHS}] Loss: {loss.item():.10f}')

    torch.save(model.state_dict(), 'model.pth')
    print(f'Finished training with final loss {loss.item():.10f}')


if __name__ == '__main__':
    train()
