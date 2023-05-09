import argparse
import os
import pickle
import time
import random
import faiss
import numpy as np
from copy import deepcopy
from sklearn.metrics.cluster import normalized_mutual_info_score
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.datasets as dset
from torch.utils.data import Dataset
from torch.autograd import Variable
import clustering
from util import AverageMeter, Logger, UnifLabelSampler
from tqdm import tqdm
import kornia
import copy
import PIL
import torch.nn.functional as F

HardNet = kornia.feature.HardNet(pretrained=True).cuda()
HardNet.eval()

def ErrorRateAt95Recall(labels, scores):
    distances = 1.0 / (scores + 1e-8)
    recall_point = 0.95
    labels = labels[np.argsort(distances)]
    threshold_index = np.argmax(np.cumsum(labels) >= recall_point * np.sum(labels)) 

    FP = np.sum(labels[:threshold_index] == 0) # Below threshold (i.e., labelled positive), but should be negative
    TN = np.sum(labels[threshold_index:] == 0) # Above threshold (i.e., labelled negative), and should be negative
    return float(FP) / float(FP + TN)

dataset_names = ['liberty', 'notredame', 'yosemite']
def parse_args():
    parser = argparse.ArgumentParser(description='PyTorch Implementation of DeepCluster')
    parser.add_argument('--arch', '-a', type=str, metavar='ARCH',
                        choices=['sift', 'hardnet'], default='sift',
                        help='architecture (default: sift)')
    parser.add_argument('--sobel', action='store_true', help='Sobel filtering')
    parser.add_argument('--clustering', type=str, choices=['Kmeans', 'PIC'],
                        default='Kmeans', help='clustering algorithm (default: Kmeans)')
    parser.add_argument('--nmb_cluster', '--k', type=int, default=100000,
                        help='number of cluster for k-means (default: 10000)')
    parser.add_argument('--lr', default=0.001, type=float,
                        help='learning rate (default: 0.05)')
    parser.add_argument('--wd', default=-5, type=float,
                        help='weight decay pow (default: -5)')
    parser.add_argument('--reassign', type=float, default=10.,
                        help="""how many epochs of training between two consecutive
                        reassignments of clusters (default: 1)""")
    parser.add_argument('--workers', default=4, type=int,
                        help='number of data loading workers (default: 4)')
    parser.add_argument('--epochs', type=int, default=20,
                        help='number of total epochs to run (default: 200)')
    parser.add_argument('--start_epoch', default=0, type=int,
                        help='manual epoch number (useful on restarts) (default: 0)')
    parser.add_argument('--batch', default=256, type=int,
                        help='mini-batch size (default: 256)')
    parser.add_argument('--momentum', default=0.9, type=float, help='momentum (default: 0.9)')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to checkpoint (default: None)')
    parser.add_argument('--checkpoints', type=int, default=25000,
                        help='how many iterations between two checkpoints (default: 25000)')
    parser.add_argument('--seed', type=int, default=31, help='random seed (default: 31)')
    parser.add_argument('--exp', type=str, default='data/logs/ss', help='path to exp folder')
    parser.add_argument('--verbose', action='store_true', default=True, help='chatty')
    parser.add_argument('--training-set', default= 'liberty',
                    help='Other options: liberty notredame, yosemite')
    parser.add_argument('--dataroot', type=str,
                    default='data/',
                    help='path to dataset')
    parser.add_argument('--reduce_dim', type=int, default=64, help='reduce_dim')
    parser.add_argument('--descriptor', type=str, default='SIFT', help='descriptor')
    parser.add_argument('--dataset_names', type=str, default='liberty', help='dataset_names, notredame, yosemite, liberty')
    return parser.parse_args()


class ArcClassifier(nn.Module):
    def __init__(self, dim, num_classes, margin=0.1, gamma=1.0,
                 trainable_gamma=True, eps=1e-7):
        super().__init__()
        self.weight = nn.parameter.Parameter(torch.empty([num_classes, dim]))
        nn.init.xavier_uniform_(self.weight)
        self.margin = margin
        self.eps = eps
        self.gamma = nn.parameter.Parameter(torch.ones(1) * gamma)
        if not trainable_gamma:
            self.gamma.requires_grad = False
    def forward(self, x, labels):
        raw_logits = F.linear(x, F.normalize(self.weight))
        theta = torch.acos(raw_logits.clamp(-1 + self.eps, 1 - self.eps))
        # Only apply margin if theta <= np.pi - self.margin.
        # mask = (theta <= np.pi - self.margin)
        # marginal_target_logits = torch.where(
        #         mask, torch.cos(theta + self.margin), raw_logits)
        # Only apply margin if it lowers the logit.
        marginal_target_logits = torch.min(torch.cos(theta + self.margin), raw_logits)
        one_hot = F.one_hot(labels, num_classes=raw_logits.size(1)).bool()
        final_logits = torch.where(one_hot, marginal_target_logits, raw_logits)
        final_logits *= self.gamma
        return final_logits

class MLP(nn.Module):
    def __init__(self, n_components = 64, hidden=96):
        super(MLP, self).__init__()
        self.classifier = nn.Linear(128, n_components)
        self._initialize_weights()

    def forward(self, x):
        x = self.classifier(x)
        x = F.normalize(x, dim=1)
        return x

    def _initialize_weights(self):
        for y, m in enumerate(self.modules()):
            if isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

def main(args, test_loaders=[]):
    # fix random seeds
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)

    # CNN
    if args.verbose:
        print('Architecture: {}'.format(args.arch))

    n_components=64
    model = MLP(n_components=n_components,hidden=96)
    model.cuda()

    cudnn.benchmark = True

    optimizer = torch.optim.Adam(
        filter(lambda x: x.requires_grad, model.parameters()),
    )

    # define loss function
    criterion = nn.CrossEntropyLoss().cuda()

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            # remove top_layer parameters from checkpoint
            for key in checkpoint['state_dict']:
                if 'top_layer' in key:
                    del checkpoint['state_dict'][key]
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    # creating checkpoint repo
    exp_check = os.path.join(args.exp, 'checkpoints')
    if not os.path.isdir(exp_check):
        os.makedirs(exp_check)

    # creating cluster assignments log
    cluster_log = Logger(os.path.join(args.exp, 'clusters'))

    # load the data
    end = time.time()

    # clustering algorithm to use
    deepcluster = clustering.__dict__[args.clustering](args.nmb_cluster)

    # training convnet with DeepCluster
    for epoch in range(args.start_epoch, args.epochs):
        end = time.time()
        dataloader = create_train_loaders()
        if epoch > 0:
            features, descriptors = compute_features(dataloader, model, 450092)
        else:
            features = compute_features_init(dataloader, 450092)
            descriptors = features

        # cluster the features
        if args.verbose:
            print('Cluster the features')
        clustering_loss = deepcluster.cluster(features, verbose=args.verbose)

        # assign pseudo-labels
        if args.verbose:
            print('Assign pseudo labels')
        train_dataset = clustering.cluster_assign(deepcluster.images_lists,
                                                  descriptors)

        # uniformly sample per target
        sampler = UnifLabelSampler(int(args.reassign * len(train_dataset)),
                                   deepcluster.images_lists)

        train_dataloader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=args.batch,
            num_workers=args.workers,
            sampler=sampler,
            pin_memory=True,
        )

        classifier = ArcClassifier(dim=n_components, num_classes=args.nmb_cluster)
        classifier.cuda()

        # train network with clusters as pseudo-labels
        end = time.time()
        loss = train(train_dataloader, model, classifier, criterion, optimizer, epoch)

        # print log
        if args.verbose:
            print('###### Epoch [{0}] ###### \n'
                  'Time: {1:.3f} s\n'
                  'Clustering loss: {2:.3f} \n'
                  'ConvNet loss: {3:.3f}'
                  .format(epoch, time.time() - end, clustering_loss, loss))
            try:
                nmi = normalized_mutual_info_score(
                    clustering.arrange_clustering(deepcluster.images_lists),
                    clustering.arrange_clustering(cluster_log.data[-1])
                )
                print('NMI against previous assignment: {0:.3f}'.format(nmi))
            except IndexError:
                pass
            print('####################### \n')
        # save running checkpoint
        torch.save({'epoch': epoch + 1,
                    'arch': args.arch,
                    'state_dict': model.state_dict(),
                    'optimizer' : optimizer.state_dict()},
                   os.path.join(args.exp, 'checkpoint_20000_adam_200.pth.tar'))

        # save cluster assignments
        cluster_log.log(deepcluster.images_lists)

        file_name = 'models/ss_' + args.descriptor + '_' + str(args.reduce_dim) + '_' + args.dataset_names + '.pth'
        torch.save(model.state_dict(), file_name)
        for test_loader in test_loaders:
            test(test_loader['dataloader'], model, epoch, test_loader['name'])
        # test(test_loaders[0]['dataloader'], model, epoch, test_loaders[0]['name'])


def train(loader, model, classifier, crit, opt, epoch):
    batch_time = AverageMeter()
    losses = AverageMeter()
    data_time = AverageMeter()
    forward_time = AverageMeter()
    backward_time = AverageMeter()

    # switch to train mode
    model.train()
    classifier.train()
    optimizer_tl = torch.optim.Adam(
        classifier.parameters(),
    )

    end = time.time()
    for i, (input_tensor, target) in enumerate(loader):
        data_time.update(time.time() - end)

        # save checkpoint
        n = len(loader) * epoch + i
        if n % args.checkpoints == 0:
            path = os.path.join(
                args.exp,
                'checkpoints',
                'checkpoint_' + str(n / args.checkpoints) + '.pth.tar',
            )
            if args.verbose:
                print('Save checkpoint at: {0}'.format(path))
            torch.save({
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'optimizer' : opt.state_dict()
            }, path)

        target = target.cuda()
        input_var = torch.autograd.Variable(input_tensor.cuda())
        target_var = torch.autograd.Variable(target)

        output = model(input_var)
        output = classifier(output,target_var)
        loss = crit(output, target_var)

        # record loss
        losses.update(loss.item(), input_tensor.size(0))

        # compute gradient and do SGD step
        opt.zero_grad()
        optimizer_tl.zero_grad()
        loss.backward()
        opt.step()
        optimizer_tl.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if args.verbose and (i % 200) == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time: {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data: {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss: {loss.val:.4f} ({loss.avg:.4f})'
                  .format(epoch, i, len(loader), batch_time=batch_time,
                          data_time=data_time, loss=losses))

    return losses.avg

class TripletPhotoTour(dset.PhotoTour):
    """
    From the PhotoTour Dataset it generates triplet samples
    note: a triplet is composed by a pair of matching images and one of
    different class.
    """
    def __init__(self, train=True, transform=None, batch_size = None,load_random_triplets = False,  *arg, **kw):
        super(TripletPhotoTour, self).__init__(*arg, **kw)
        self.transform = transform
        self.out_triplets = load_random_triplets
        self.train = train
        self.n_triplets = 5000000
        self.batch_size = batch_size

        if self.train:
            print('Generating {} triplets'.format(self.n_triplets))
            self.triplets = self.generate_triplets(self.labels, self.n_triplets)

    @staticmethod
    def generate_triplets(labels, num_triplets):
        def create_indices(_labels):
            inds = dict()
            for idx, ind in enumerate(_labels):
                if ind not in inds:
                    inds[ind] = []
                inds[ind].append(idx)
            return inds

        triplets = []
        indices = create_indices(labels.numpy())
        unique_labels = np.unique(labels.numpy())
        n_classes = unique_labels.shape[0]
        # add only unique indices in batch
        already_idxs = set()

        for x in tqdm(range(num_triplets)):
            if len(already_idxs) >= 1024:
                already_idxs = set()
            c1 = np.random.randint(0, n_classes)
            while c1 in already_idxs:
                c1 = np.random.randint(0, n_classes)
            already_idxs.add(c1)
            c2 = np.random.randint(0, n_classes)
            while c1 == c2:
                c2 = np.random.randint(0, n_classes)
            if len(indices[c1]) == 2:  # hack to speed up process
                n1, n2 = 0, 1
            else:
                n1 = np.random.randint(0, len(indices[c1]))
                n2 = np.random.randint(0, len(indices[c1]))
                while n1 == n2:
                    n2 = np.random.randint(0, len(indices[c1]))
            n3 = np.random.randint(0, len(indices[c2]))
            triplets.append([indices[c1][n1], indices[c1][n2], indices[c2][n3]])
        return torch.LongTensor(np.array(triplets))

    def __getitem__(self, index):
        def transform_img(img):
            if self.transform is not None:
                img = self.transform(img.numpy())
            return img

        if not self.train:
            m = self.matches[index]
            img1 = transform_img(self.data[m[0]])
            img2 = transform_img(self.data[m[1]])
            return img1, img2, m[2]

        t = self.triplets[index]
        a, p, n = self.data[t[0]], self.data[t[1]], self.data[t[2]]

        img_a = transform_img(a)
        img_p = transform_img(p)
        img_n = None
        if self.out_triplets:
            img_n = transform_img(n)
        # transform images if required
        if True:
            do_flip = random.random() > 0.5
            do_rot = random.random() > 0.5
            if do_rot:
                img_a = img_a.permute(0,2,1)
                img_p = img_p.permute(0,2,1)
                if self.out_triplets:
                    img_n = img_n.permute(0,2,1)
            if do_flip:
                img_a = torch.from_numpy(deepcopy(img_a.numpy()[:,:,::-1]))
                img_p = torch.from_numpy(deepcopy(img_p.numpy()[:,:,::-1]))
                if self.out_triplets:
                    img_n = torch.from_numpy(deepcopy(img_n.numpy()[:,:,::-1]))
        if self.out_triplets:
            return (img_a, img_p, img_n)
        else:
            return (img_a, img_p)

    def __len__(self):
        if self.train:
            return self.triplets.size(0)
        else:
            return self.matches.size(0)

def create_loaders():

    test_dataset_names = copy.copy(dataset_names)
    test_dataset_names.remove(args.training_set)

    kwargs = {'num_workers': args.workers, 'pin_memory': True}

    np_reshape64 = lambda x: np.reshape(x, (64, 64, 1))
    transform_test = transforms.Compose([
            transforms.Lambda(np_reshape64),
            transforms.ToPILImage(),
            transforms.Resize(32),
            transforms.ToTensor()])

    test_loaders = [{'name': name,
                     'dataloader': torch.utils.data.DataLoader(
             TripletPhotoTour(train=False,
                     batch_size=1024,
                     root=args.dataroot,
                     name=name,
                     download=True,
                     transform=transform_test),
                        batch_size=1024,
                        shuffle=False, **kwargs)}
                    for name in test_dataset_names]

    return test_loaders

class NewPhotoTour(dset.PhotoTour):
    """
    From the PhotoTour Dataset it generates triplet samples
    note: a triplet is composed by a pair of matching images and one of
    different class.
    """
    def __init__(self,  *arg, **kw):
        super(NewPhotoTour, self).__init__(*arg, **kw)

    def __getitem__(self, index):
        if self.train:
            data = self.data[index]
            if self.transform is not None:
                data = self.transform(data.numpy())
            return data

def create_train_loaders():
    kwargs = {'num_workers': args.workers, 'pin_memory': True}
    np_reshape64 = lambda x: np.reshape(x, (64, 64, 1))
    transform_train = transforms.Compose([
            transforms.Lambda(np_reshape64),
            transforms.ToPILImage(),
            transforms.RandomRotation(5,PIL.Image.BILINEAR),
            transforms.RandomResizedCrop(32, scale = (0.9,1.0),ratio = (0.9,1.1)),
            transforms.Resize(32),
            transforms.ToTensor()])
    
    train_loader = torch.utils.data.DataLoader(
            NewPhotoTour(
                             root='./data', 
                             name=args.training_set, 
                             train=True, 
                             transform=transform_train, 
                             download=True),
                             batch_size=args.batch,
                             shuffle=False, **kwargs)

    return train_loader

def test(test_loader, model, epoch, logger_test_name):
    # switch to evaluate mode
    model.eval()

    labels, distances = [], []

    pbar = tqdm(enumerate(test_loader))
    with torch.no_grad():
        for batch_idx, (data_a, data_p, label) in pbar:
            data_a, data_p = data_a.cuda(), data_p.cuda()
            out_a = model(HardNet(data_a))
            out_p = model(HardNet(data_p))
            dists = torch.sqrt(torch.sum((out_a - out_p) ** 2, 1))  # euclidean distance
            distances.append(dists.data.cpu().numpy().reshape(-1,1))
            ll = label.data.cpu().numpy().reshape(-1, 1)
            labels.append(ll)

            if batch_idx % 10 == 0:
                pbar.set_description(logger_test_name+' Test Epoch: {} [{}/{} ({:.0f}%)]'.format(
                    epoch, batch_idx * len(data_a), len(test_loader.dataset),
                        100. * batch_idx / len(test_loader)))

    num_tests = test_loader.dataset.matches.size(0)
    labels = np.vstack(labels).reshape(num_tests)
    distances = np.vstack(distances).reshape(num_tests)

    fpr95 = ErrorRateAt95Recall(labels, 1.0 / (distances + 1e-8))
    print('\33[91mTest set: Accuracy(FPR95): {:.8f}\n\33[0m'.format(fpr95))


def compute_features(dataloader, model, N):
    if args.verbose:
        print('Compute features')
    batch_time = AverageMeter()
    end = time.time()
    model.eval()
    # discard the label information in the dataloader
    with torch.no_grad():
        for i, input_tensor in enumerate(dataloader):
            input_var = input_tensor.cuda()
            des = HardNet(input_var)
            aux = model(des).cpu().detach().numpy()
            descr = des.cpu().detach().numpy()

            if i == 0:
                features = np.zeros((N, aux.shape[1]), dtype='float32')
                descrs = np.zeros((N, descr.shape[1]), dtype='float32')

            aux = aux.astype('float32')
            if i < len(dataloader) - 1:
                features[i * args.batch: (i + 1) * args.batch] = aux
                descrs[i * args.batch: (i + 1) * args.batch] = descr
            else:
                # special treatment for final batch
                features[i * args.batch:] = aux
                descrs[i * args.batch:] = descr

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if args.verbose and (i % 200) == 0:
                print('{0} / {1}\t'
                    'Time: {batch_time.val:.3f} ({batch_time.avg:.3f})'
                    .format(i, len(dataloader), batch_time=batch_time))
    return features, descrs

def compute_features_init(dataloader, N):
    if args.verbose:
        print('Compute features init')
    batch_time = AverageMeter()
    end = time.time()
    # discard the label information in the dataloader
    for i, input_tensor in enumerate(dataloader):
        aux = HardNet(input_tensor.cuda()).cpu().detach().numpy()

        if i == 0:
            features = np.zeros((N, aux.shape[1]), dtype='float32')

        aux = aux.astype('float32')
        if i < len(dataloader) - 1:
            features[i * args.batch: (i + 1) * args.batch] = aux
        else:
            # special treatment for final batch
            features[i * args.batch:] = aux

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if args.verbose and (i % 200) == 0:
            print('{0} / {1}\t'
                  'Time: {batch_time.val:.3f} ({batch_time.avg:.3f})'
                  .format(i, len(dataloader), batch_time=batch_time))
    return features


if __name__ == '__main__':
    args = parse_args()
    test_loader = create_loaders()
    #test_loader = []
    main(args,test_loader)
