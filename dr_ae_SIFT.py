import torch
from torch.utils.data import Dataset
import os
import numpy as np
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import argparse
import random

parser = argparse.ArgumentParser(description='PyTorch dr')
parser.add_argument('--descriptor', type=str, default='SIFT', help='descriptor')
parser.add_argument('--dataset_names', type=str, default='liberty', help='dataset_names, notredame, yosemite, liberty')
parser.add_argument('--reduce_dim', type=int, default=64, help='reduce_dim')
parser.add_argument('--hidden', type=int, default=1024, help='hidden')
parser.add_argument('--bsz', type=int, default=1024, help='bsz')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
parser.add_argument('--seed', type=int, default=0, help='random seed (default: 0)')
args = parser.parse_args()

np.random.seed(args.seed)
torch.manual_seed(args.seed)
random.seed(args.seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


class DescriotorDataset(Dataset):
    def __init__(self, des_dir, descriptor):
        self.descriptorsfile = os.path.join(des_dir, descriptor + '-' + args.dataset_names + '.npz')
        self.descriptors = np.load(self.descriptorsfile)['descriptors']

    def __len__(self):
        return self.descriptors.shape[0]

    def __getitem__(self, idx):
        descriptor = self.descriptors[idx]
        return torch.from_numpy(descriptor)

descriptors = DescriotorDataset(
    des_dir="raw_descriptors",
    descriptor=args.descriptor
)


class Encoder(nn.Module):
    def __init__(self, n_components, hidden=1024):
        super(Encoder, self).__init__()
        self.enc_net = nn.Sequential(
          nn.Linear(128, hidden),
          nn.ReLU(inplace=True),
          nn.BatchNorm1d(hidden),
          nn.Linear(hidden, hidden),
          nn.ReLU(inplace=True),
          nn.BatchNorm1d(hidden),
          nn.Linear(hidden, n_components),
        )

    def forward(self, x):
        output = self.enc_net(x)
        output = F.normalize(output, dim=1)
        return output

class Decoder(nn.Module):
    def __init__(self, n_components, hidden=1024):
        super(Decoder, self).__init__()
        self.dec_net = nn.Sequential(
          nn.Linear(n_components, hidden),
          nn.ReLU(inplace=True),
          nn.BatchNorm1d(hidden),
          nn.Linear(hidden, hidden),
          nn.ReLU(inplace=True),
          nn.BatchNorm1d(hidden),
          nn.Linear(hidden, 128),
          nn.ReLU()
        )

    def forward(self, z):
        output = self.dec_net(z)
        output = F.normalize(output, dim=1)
        return output

device = torch.device('cuda:0')

encoder = Encoder(args.reduce_dim, hidden=args.hidden)
encoder.to(device)

decoder = Decoder(args.reduce_dim, hidden=args.hidden)
decoder.to(device)


encoder_optimizer = optim.Adam(encoder.parameters(), lr=args.lr)
decoder_optimizer = optim.Adam(decoder.parameters(), lr=args.lr)
criterion = nn.MSELoss()   
train_dataloader = DataLoader(descriptors, batch_size=args.bsz, shuffle=True)
print("start training") 
num_epochs = 5
for epoch in range(num_epochs):
    encoder.train()
    decoder.train()
    losses = []
    for descs in train_dataloader:
        encoder_optimizer.zero_grad()
        decoder_optimizer.zero_grad()

        descs = descs.to(device)
        output = encoder(descs)
        output = decoder(output)
        
        loss = criterion(output, descs)
        losses.append(loss.item())
        loss.backward()

        encoder_optimizer.step()
        decoder_optimizer.step()
        
    mean_loss = np.mean(np.array(losses))
    print ('Epoch {}, train_error: {:.4f}' 
            .format(epoch, mean_loss))

file_name = 'models/ae_' + args.descriptor + '_' + str(args.reduce_dim) + '_' + args.dataset_names + '.pth'
torch.save(encoder.state_dict(), file_name)
 