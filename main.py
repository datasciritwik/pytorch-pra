from torch import save
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

train = datasets.MNIST(root='data', 
                       download=True,
                       train=True,
                       transform=ToTensor()
                       )
dataset = DataLoader(train, 32)

class ImageClassifier(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(1, 32, (3, 3)),
            nn.ReLU(),
            nn.Conv2d(32, 64, (3, 3)),
            nn.ReLU(),
            nn.Conv2d(64, 64, (3, 3)),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64*(28-6)*(28-6), 10)
        )

    def forward(self, x):
        return self.model(x)
    
## Intances of the nn, loss, optimizer
clf = ImageClassifier().to('cpu')
opt = Adam(clf.parameters(), lr=1e-3)
loss_fn = nn.CrossEntropyLoss()

## Training flow
if __name__ == "__main__":
    for epoch in range(10):
        for batch in dataset:
            X, y = batch
            X, y = X.to('cpu'), y.to('cpu')
            yhat = clf(X)
            loss = loss_fn(yhat, y)

            ## Backward Prop.
            opt.zero_grad()
            loss.backward()
            opt.step()

        print(f"Epoch: {epoch} | Loss: {loss.item()}")

    with open('model.pt', 'wb') as f:
        save(clf.state_dict(), f)

