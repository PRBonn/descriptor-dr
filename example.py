import cv2
from kornia.constants import T
import numpy as np
from sklearn.decomposition import PCA
import pickle as pk
import kornia
import torch 
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets
import argparse
from torch.utils.data import DataLoader
from tqdm import tqdm

parser = argparse.ArgumentParser(description='PyTorch dr')
parser.add_argument('--descriptor', type=str, default='SIFT', help='descriptor')
parser.add_argument('--dataset_names', type=str, default='liberty', help='dataset_names, notredame, yosemite, liberty')
parser.add_argument('--reduce_dim', type=int, default=64, help='reduce_dim')
args = parser.parse_args()

dataset = datasets.PhotoTour(
    root='./data', name=args.dataset_names, train=True, transform=None, download=True)


def load_model(model, filename, device):
    model.load_state_dict(torch.load(filename, map_location=lambda storage, loc: storage))
    print('Model loaded from %s.' % filename)
    model.to(device)
    model.eval()

class L2Norm(nn.Module):
    def __init__(self):
        super(L2Norm,self).__init__()
        self.eps = 1e-10
    def forward(self, x):
        norm = torch.sqrt(torch.sum(x * x, dim = 1) + self.eps)
        x= x / norm.unsqueeze(-1).expand_as(x)
        return x

class Encoder(nn.Module):
    def __init__(self, n_components,hidden=128):
        super(Encoder, self).__init__()
        self.enc_net = nn.Sequential(
          nn.Linear(128, hidden),
          nn.ReLU(),
          nn.BatchNorm1d(hidden),
          nn.Linear(hidden, hidden),
          nn.ReLU(),
          nn.BatchNorm1d(hidden),
          nn.Linear(hidden, n_components)
        )

    def forward(self, x):
        return L2Norm()(self.enc_net(x))

SIFT = kornia.feature.SIFTDescriptor(32, 8, 4, False).cuda()
device = torch.device('cuda:0')
n_components=64
encoder = Encoder(n_components=n_components,hidden=1024)
load_model(encoder, 'models/SIFT_sv_dim64.pth', device)

cv2_scale = lambda x: cv2.resize(x, dsize=(32, 32),
                                 interpolation=cv2.INTER_LINEAR)

dataloader = DataLoader(dataset, batch_size=128, shuffle=False)
descriptors = torch.empty((0,128), dtype=torch.float)

for batch_idx, patches in enumerate(tqdm(dataloader)):
    patches_32 = np.empty([0,1,32,32])
    patches = patches.cpu().detach().numpy()
    for i in range(patches.shape[0]):
        patch = cv2_scale(patches[i])
        patch = np.expand_dims(patch, axis=0)
        patch = np.expand_dims(patch, axis=0)
        patches_32 = np.concatenate((patches_32,patch),axis=0)
    descs = SIFT(torch.from_numpy(patches_32).float().cuda()) # original SIFT descriptor (128)
    print(descs.shape)
    descs_dr = encoder(descs) # reducted SIFT descriptor (64), to be used in downstream tasks
    print(descs_dr.shape)
    break

