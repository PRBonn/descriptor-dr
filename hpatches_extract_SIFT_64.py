import sys
import glob
import os
import cv2
from kornia.constants import T
import numpy as np
from sklearn.decomposition import PCA
import pickle as pk
import kornia
import torch 
import torch.nn as nn
import torch.nn.functional as F
assert len(sys.argv)==2, "Usage python hpatches_extract.py hpatches_db_root_folder"
    
# all types of patches 
tps = ['ref','e1','e2','e3','e4','e5','h1','h2','h3','h4','h5',\
       't1','t2','t3','t4','t5']

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

class hpatches_sequence:
    """Class for loading an HPatches sequence from a sequence folder"""
    itr = tps
    def __init__(self,base):
        name = base.split('/')
        self.name = name[-1]
        self.base = base
        for t in self.itr:
            im_path = os.path.join(base, t+'.png')
            im = cv2.imread(im_path,0)
            self.N = im.shape[0]/65
            setattr(self, t, np.split(im, self.N))
            
    
seqs = glob.glob(sys.argv[1]+'/*')
seqs = [os.path.abspath(p) for p in seqs]     

hidden=1024
descr_name = 'SIFT_sv_dim64'
SIFT = kornia.feature.SIFTDescriptor(32, 8, 4, False).cuda()
device = torch.device('cuda:0')
n_components=64
encoder = Encoder(n_components=n_components,hidden=hidden)
load_model(encoder, 'models/SIFT_sv_dim64.pth', device)
cv2_scale = lambda x: cv2.resize(x, dsize=(32, 32),
                                 interpolation=cv2.INTER_LINEAR)
w = 65
for seq_path in seqs:
    seq = hpatches_sequence(seq_path)
    path = os.path.join(descr_name,seq.name)
    if not os.path.exists(path):
        os.makedirs(path)
    for tp in tps:
        print(seq.name+'/'+tp)
        if os.path.isfile(os.path.join(path,tp+'.csv')):
            continue
        n_patches = 0
        for i,patch in enumerate(getattr(seq, tp)):
            n_patches+=1
        patches_for_net = np.zeros((n_patches, 1, 32, 32))
        for i,patch in enumerate(getattr(seq, tp)):
            patches_for_net[i,0,:,:] = cv2.resize(patch[0:w,0:w],(32,32))
        encoder.eval()
        outs = []
        bs = 128
        n_batches = int(n_patches / bs) + 1
        for batch_idx in range(n_batches):
            st = batch_idx * bs
            if batch_idx == n_batches - 1:
                if (batch_idx + 1) * bs > n_patches:
                    end = n_patches
                else:
                    end = (batch_idx + 1) * bs            
            else:
                end = (batch_idx + 1) * bs
            if st >= end:
                continue
            data_a = patches_for_net[st: end, :, :, :].astype(np.float32)
            data_a = torch.from_numpy(data_a)

            data_a = data_a.to(device)

            out_a = SIFT(data_a)
            out_a = encoder(out_a)
            outs.append(out_a.data.cpu().numpy().reshape(-1, n_components))
        res_desc = np.concatenate(outs)
        res_desc = np.reshape(res_desc, (n_patches, -1))
        out = np.reshape(res_desc, (n_patches,-1))
        np.savetxt(os.path.join(path,tp+'.csv'), out, delimiter=',', fmt='%10.5f')   

