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
from tqdm import tqdm

parser = argparse.ArgumentParser(description='PyTorch dr')
parser.add_argument('--descriptor', type=str, default='HardNet', help='descriptor')
parser.add_argument('--dataset_names', type=str, default='liberty', help='dataset_names, notredame, yosemite, liberty')
parser.add_argument('--reduce_dim', type=int, default=64, help='reduce_dim')
parser.add_argument('--hidden', type=int, default=96, help='hidden')
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
          nn.Linear(hidden, 128),
          nn.ReLU()
        )

    def forward(self, z):
        output = self.dec_net(z)
        output = F.normalize(output, dim=1)
        return output


def distance_loss(encoders, decoders, batch, device, alpha=0.1):
    target_descriptors = batch
    embeddings = encoders(batch)
        
    t_loss = torch.tensor(0.).float().to(device)
    output_descriptors = decoders(embeddings)
    current_loss = torch.mean(
        torch.norm(output_descriptors - target_descriptors, dim=1)
    )
    t_loss += current_loss

    e_loss = torch.tensor(0.).float().to(device)
    
    sqdist_matrix_embeddings = 2 - 2 * embeddings @ embeddings.T
    sqdist_matrix_target = 2 - 2 * target_descriptors @ target_descriptors.T
    
    e_loss += torch.mean(
        torch.abs(sqdist_matrix_target - sqdist_matrix_embeddings)
    )

    if alpha > 0:
        loss = t_loss + alpha * e_loss
    else:
        loss = t_loss
    
    return loss, (t_loss.detach(), e_loss.detach())

class UpdatingMean():
    def __init__(self):
        self.sum = 0
        self.n = 0

    def mean(self):
        return self.sum / self.n

    def add(self, loss):
        self.sum += loss
        self.n += 1

device = torch.device('cuda:0')

encoder = Encoder(args.reduce_dim, hidden=args.hidden)
encoder.to(device)

decoder = Decoder(args.reduce_dim, hidden=args.hidden)
decoder.to(device)


encoder_optimizer = optim.Adam(encoder.parameters(), lr=args.lr)
decoder_optimizer = optim.Adam(decoder.parameters(), lr=args.lr)
  
loss_function = lambda encoders, decoders, batch, device: distance_loss(
        encoders, decoders, batch, device, 
        alpha=0.1
    ) 

train_dataloader = DataLoader(descriptors, batch_size=args.bsz, shuffle=True)
print("start training") 
num_epochs = 10
for epoch in range(num_epochs):
    encoder.train()
    decoder.train()
    epoch_loss = UpdatingMean()
    epoch_t_loss = UpdatingMean()
    epoch_e_loss = UpdatingMean()
    progress_bar = tqdm(enumerate(train_dataloader), total=len(train_dataloader))
        
    for batch_idx, batch in progress_bar:
        encoder_optimizer.zero_grad()
        decoder_optimizer.zero_grad()

        batch = batch.to(device)
        
        loss, (t_loss, e_loss) = loss_function(encoder, decoder, batch, device)
        
        epoch_loss.add(loss.data.cpu().numpy())
        epoch_t_loss.add(t_loss)
        epoch_e_loss.add(e_loss)
        progress_bar.set_postfix(
            loss=('%.4f' % epoch_loss.mean()),
            t_loss=('%.4f' % epoch_t_loss.mean()),
            e_loss=('%.4f' % epoch_e_loss.mean())
        )

        loss.backward()

        encoder_optimizer.step()
        decoder_optimizer.step()
        
    print ('Epoch {}, train_error: {:.4f}' 
            .format(epoch, epoch_loss.mean()))

file_name = 'models/ae_' + args.descriptor + '_' + str(args.reduce_dim) + '_' + args.dataset_names + '.pth'
torch.save(encoder.state_dict(), file_name)
 