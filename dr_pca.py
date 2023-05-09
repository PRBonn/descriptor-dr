import torch
from torch.utils.data import DataLoader
from torchvision import datasets
import kornia
import cv2
import numpy as np
import argparse
from tqdm import tqdm
from sklearn.decomposition import PCA
import pickle as pk
import os

parser = argparse.ArgumentParser(description='PyTorch dr')
parser.add_argument('--descriptor', type=str, default='SIFT', help='descriptor')
parser.add_argument('--dataset_names', type=str, default='liberty', help='dataset_names, notredame, yosemite, liberty')
parser.add_argument('--reduce_dim', type=int, default=64, help='reduce_dim')
args = parser.parse_args()

dataset = datasets.PhotoTour(
    root='./data', name=args.dataset_names, train=True, transform=None, download=True)

if args.descriptor == 'SIFT':
    des = kornia.feature.SIFTDescriptor(32, 8, 4, False).cuda()
elif args.descriptor == 'MKD':
    des = kornia.feature.MKDDescriptor().cuda()
elif args.descriptor == 'TFeat':
    des = kornia.feature.TFeat(pretrained=True).cuda()
elif args.descriptor == 'HardNet':
    des = kornia.feature.HardNet(pretrained=True).cuda()

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
    descs = des(torch.from_numpy(patches_32).float().cuda()).cpu().detach()
    descriptors = torch.vstack((descriptors,descs))

descriptors = descriptors.cpu().detach().numpy()
print(descriptors.shape)

descriptorsfile_name = args.descriptor + '-' + args.dataset_names + '.npz'
descriptorsfile = os.path.join('raw_descriptors', descriptorsfile_name)
np.savez(descriptorsfile, descriptors=descriptors)

pca = PCA(n_components=args.reduce_dim)
pca.fit(descriptors)

descriptors_pca = pca.transform(descriptors)
print(descriptors_pca.shape)

save_name = 'pca' + str(args.reduce_dim) + '-' + args.descriptor + '-' + args.dataset_names + '.pkl'
pk.dump(pca, open("models/"+save_name,"wb"))

pca_reload = pk.load(open("models/"+save_name,'rb'))
descriptors_pca_reload  = pca_reload .transform(descriptors)
print(descriptors_pca_reload.shape)