from torch import load
import torch
from PIL import Image
from torchvision.transforms import ToTensor
import main

clf = main.ImageClassifier().to('cpu')


if __name__ == '__main__':
    with open('model.pt', 'rb') as f:
        clf.load_state_dict(load(f))

    img = Image.open('img_3.jpg')
    img_tensor = ToTensor()(img).unsqueeze(0).to('cpu')

    print('Predicted Number: ',torch.argmax(clf(img_tensor)))