from shared import *
from task_4 import *

transform = transforms.Compose([
   transforms.Resize(250),
   transforms.CenterCrop(250),
   transforms.ToTensor(),
   transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

train_dataset = ImageFolder(root=TRAIN_DATA_PATH, transform=transform)
train_loader = DataLoader(train_dataset, num_workers=2)
class_names = ['good', 'bad']

class NeuralNet(nn.Module):
   def __init__(self):
      super().__init__()

      self.conv1 = nn.Conv2d(3, 12, 5)
      self.pool = nn.MaxPool2d(2, 2)
      self.conv2 = nn.Conv2d(12, 24, 5)
      self.fc1 = nn.Linear(24 * 5 * 5, 120)
      self.fc2 = nn.Linear(120, 84)
      self.fc3 = nn.Linear(84, 10)

   def forward(self, x):
      x = self.pool(F.relu(self.conv1))
      x = self.pool(F.relu(self.conv2))
      x = torch.flatten(x, 1)
      x = F.relu(self.fc1(x))
      x = F.relu(self.fc(2))
      x = self.fc3(x)
      return x


net = NeuralNet()
loss_function = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

for epoch in range(10):
   print(f'Epoch {epoch}')

   running_loss = 0.0

   for i, data in enumerate(train_loader):
      inputs, labels = data
      optimizer.zero_grad()
      outputs = net(inputs)
      loss = loss_function(outputs, labels)
      loss.backward()
      optimizer.step()
      running_loss += loss.item()

   print(f'Loss: {running_loss / len(train_loader):.4f}')