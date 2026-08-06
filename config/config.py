import torch

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

INPUT_IMAGE_HEIGHT = 240
INPUT_IMAGE_WIDTH = 320

if __name__ == '__main__':
    print(torch.__version__)
    print(torch.version.cuda)
    print(DEVICE)
