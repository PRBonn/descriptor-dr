import torch
import torch.nn as nn
import sys

def distance_matrix_vector(anchor, positive):
    """Given batch of anchor descriptors and positive descriptors calculate distance matrix"""

    d1_sq = torch.sum(anchor * anchor, dim=1).unsqueeze(-1)
    d2_sq = torch.sum(positive * positive, dim=1).unsqueeze(-1)

    eps = 1e-6
    return torch.sqrt((d1_sq.repeat(1, positive.size(0)) + torch.t(d2_sq.repeat(1, anchor.size(0)))
                      - 2.0 * torch.bmm(anchor.unsqueeze(0), torch.t(positive).unsqueeze(0)).squeeze(0))+eps)

def inner_dot_matrix(anchor, postive):
    inner = torch.mm(anchor, torch.t(postive))
    mask = torch.eye(inner.size(1)).cuda() 
    inner = inner - 1e-6*mask
    dist_m = torch.sqrt( 2.0*(1.0-inner) + 1e-8)
    return dist_m

def distance_vectors_pairwise(anchor, positive, negative = None):
    """Given batch of anchor descriptors and positive descriptors calculate distance matrix"""

    a_sq = torch.sum(anchor * anchor, dim=1)
    p_sq = torch.sum(positive * positive, dim=1)

    eps = 1e-8
    d_a_p = torch.sqrt(a_sq + p_sq - 2*torch.sum(anchor * positive, dim = 1) + eps)
    if negative is not None:
        n_sq = torch.sum(negative * negative, dim=1)
        d_a_n = torch.sqrt(a_sq + n_sq - 2*torch.sum(anchor * negative, dim = 1) + eps)
        d_p_n = torch.sqrt(p_sq + n_sq - 2*torch.sum(positive * negative, dim = 1) + eps)
        return d_a_p, d_a_n, d_p_n
    return d_a_p
def loss_random_sampling(anchor, positive, negative, anchor_swap = False, margin = 1.0, loss_type = "triplet_margin"):
    """Loss with random sampling (no hard in batch).
    """

    assert anchor.size() == positive.size(), "Input sizes between positive and negative must be equal."
    assert anchor.size() == negative.size(), "Input sizes between positive and negative must be equal."
    assert anchor.dim() == 2, "Inputd must be a 2D matrix."
    eps = 1e-8
    (pos, d_a_n, d_p_n) = distance_vectors_pairwise(anchor, positive, negative)
    if anchor_swap:
       min_neg = torch.min(d_a_n, d_p_n)
    else:
       min_neg = d_a_n

    if loss_type == "triplet_margin":
        loss = torch.clamp(margin + pos - min_neg, min=0.0)
    elif loss_type == 'softmax':
        exp_pos = torch.exp(2.0 - pos);
        exp_den = exp_pos + torch.exp(2.0 - min_neg) + eps;
        loss = - torch.log( exp_pos / exp_den )
    elif loss_type == 'contrastive':
        loss = torch.clamp(margin - min_neg, min=0.0) + pos;
    else: 
        print ('Unknown loss type. Try triplet_margin, softmax or contrastive')
        sys.exit(1)
    loss = torch.mean(loss)
    return loss

def loss_L2Net(anchor, positive, anchor_swap = False,  margin = 1.0, loss_type = "triplet_margin"):
    """L2Net losses: using whole batch as negatives, not only hardest.
    """

    assert anchor.size() == positive.size(), "Input sizes between positive and negative must be equal."
    assert anchor.dim() == 2, "Inputd must be a 2D matrix."
    eps = 1e-8
    dist_matrix = distance_matrix_vector(anchor, positive)
    eye = torch.autograd.Variable(torch.eye(dist_matrix.size(1))).cuda()

    # steps to filter out same patches that occur in distance matrix as negatives
    pos1 = torch.diag(dist_matrix)
    dist_without_min_on_diag = dist_matrix+eye*10
    mask = (dist_without_min_on_diag.ge(0.008).float()-1.0)*(-1)
    mask = mask.type_as(dist_without_min_on_diag)*10
    dist_without_min_on_diag = dist_without_min_on_diag+mask
    
    if loss_type == 'softmax':
        exp_pos = torch.exp(2.0 - pos1)
        exp_den = torch.sum(torch.exp(2.0 - dist_matrix),1) + eps
        loss = -torch.log( exp_pos / exp_den )
        if anchor_swap:
            exp_den1 = torch.sum(torch.exp(2.0 - dist_matrix),0) + eps
            loss += -torch.log( exp_pos / exp_den1 )
    else: 
        print ('Only softmax loss works with L2Net sampling')
        sys.exit(1)
    loss = torch.mean(loss)
    return loss

def loss_HardNet(anchor, positive, anchor_swap = False, anchor_ave = False,\
        margin = 1.0, batch_reduce = 'min', loss_type = "triplet_margin"):
    """HardNet margin loss - calculates loss based on distance matrix based on positive distance and closest negative distance.
    """

    assert anchor.size() == positive.size(), "Input sizes between positive and negative must be equal."
    assert anchor.dim() == 2, "Inputd must be a 2D matrix."
    eps = 1e-8
    dist_matrix = distance_matrix_vector(anchor, positive) +eps
    eye = torch.autograd.Variable(torch.eye(dist_matrix.size(1))).cuda()

    # steps to filter out same patches that occur in distance matrix as negatives
    pos1 = torch.diag(dist_matrix)
    dist_without_min_on_diag = dist_matrix+eye*10
    mask = (dist_without_min_on_diag.ge(0.008).float()-1.0)*(-1)
    mask = mask.type_as(dist_without_min_on_diag)*10
    dist_without_min_on_diag = dist_without_min_on_diag+mask
    if batch_reduce == 'min':
        min_neg = torch.min(dist_without_min_on_diag,1)[0]
        if anchor_swap:
            min_neg2 = torch.min(dist_without_min_on_diag,0)[0]
            min_neg = torch.min(min_neg,min_neg2)
        min_neg = min_neg
        pos = pos1
    elif batch_reduce == 'average':
        pos = pos1.repeat(anchor.size(0)).view(-1,1).squeeze(0)
        min_neg = dist_without_min_on_diag.view(-1,1)
        if anchor_swap:
            min_neg2 = torch.t(dist_without_min_on_diag).contiguous().view(-1,1)
            min_neg = torch.min(min_neg,min_neg2)
        min_neg = min_neg.squeeze(0)
    elif batch_reduce == 'random':
        idxs = torch.autograd.Variable(torch.randperm(anchor.size()[0]).long()).cuda()
        min_neg = dist_without_min_on_diag.gather(1,idxs.view(-1,1))
        if anchor_swap:
            min_neg2 = torch.t(dist_without_min_on_diag).gather(1,idxs.view(-1,1)) 
            min_neg = torch.min(min_neg,min_neg2)
        min_neg = torch.t(min_neg).squeeze(0)
        pos = pos1
    else: 
        print ('Unknown batch reduce mode. Try min, average or random')
        sys.exit(1)
    if loss_type == "triplet_margin":
        loss = torch.clamp(margin + pos - min_neg, min=0.0)
        #loss = nn.ReLU()(margin + pos - min_neg)
    elif loss_type == "triplet_margin_QHT":
        loss = torch.square(torch.clamp(margin + pos - min_neg, min=0.0))
    elif loss_type == 'softmax':
        exp_pos = torch.exp(2.0 - pos)
        exp_den = exp_pos + torch.exp(2.0 - min_neg) + eps
        loss = - torch.log( exp_pos / exp_den )
    elif loss_type == 'contrastive':
        loss = torch.clamp(margin - min_neg, min=0.0) + pos
    else: 
        print ('Unknown loss type. Try triplet_margin, softmax or contrastive')
        sys.exit(1)
    loss = torch.mean(loss)
    return loss

def loss_HardNet_metric(anchor, positive,out_a_raw,out_p_raw, anchor_swap = False, anchor_ave = False,\
        margin = 1.0, batch_reduce = 'min', loss_type = "triplet_margin",alpha=0.0):
    """HardNet margin loss - calculates loss based on distance matrix based on positive distance and closest negative distance.
    """

    assert anchor.size() == positive.size(), "Input sizes between positive and negative must be equal."
    assert anchor.dim() == 2, "Inputd must be a 2D matrix."
    eps = 1e-8
    dist_matrix = distance_matrix_vector(anchor, positive) +eps
    eye = torch.autograd.Variable(torch.eye(dist_matrix.size(1))).cuda()

    # steps to filter out same patches that occur in distance matrix as negatives
    pos1 = torch.diag(dist_matrix)
    dist_without_min_on_diag = dist_matrix+eye*10
    mask = (dist_without_min_on_diag.ge(0.008).float()-1.0)*(-1)
    mask = mask.type_as(dist_without_min_on_diag)*10
    dist_without_min_on_diag = dist_without_min_on_diag+mask
    if batch_reduce == 'min':
        min_neg = torch.min(dist_without_min_on_diag,1)[0]
        if anchor_swap:
            min_neg2 = torch.min(dist_without_min_on_diag,0)[0]
            min_neg = torch.min(min_neg,min_neg2)
        min_neg = min_neg
        pos = pos1
    elif batch_reduce == 'average':
        pos = pos1.repeat(anchor.size(0)).view(-1,1).squeeze(0)
        min_neg = dist_without_min_on_diag.view(-1,1)
        if anchor_swap:
            min_neg2 = torch.t(dist_without_min_on_diag).contiguous().view(-1,1)
            min_neg = torch.min(min_neg,min_neg2)
        min_neg = min_neg.squeeze(0)
    elif batch_reduce == 'random':
        idxs = torch.autograd.Variable(torch.randperm(anchor.size()[0]).long()).cuda()
        min_neg = dist_without_min_on_diag.gather(1,idxs.view(-1,1))
        if anchor_swap:
            min_neg2 = torch.t(dist_without_min_on_diag).gather(1,idxs.view(-1,1)) 
            min_neg = torch.min(min_neg,min_neg2)
        min_neg = torch.t(min_neg).squeeze(0)
        pos = pos1
    else: 
        print ('Unknown batch reduce mode. Try min, average or random')
        sys.exit(1)
    if loss_type == "triplet_margin":
        loss = torch.clamp(margin + pos - min_neg, min=0.0)
        #loss = nn.ReLU()(margin + pos - min_neg)
    elif loss_type == "triplet_margin_QHT":
        loss = torch.square(torch.clamp(margin + pos - min_neg, min=0.0))
    elif loss_type == 'softmax':
        exp_pos = torch.exp(2.0 - pos)
        exp_den = exp_pos + torch.exp(2.0 - min_neg) + eps
        loss = - torch.log( exp_pos / exp_den )
    elif loss_type == 'contrastive':
        loss = torch.clamp(margin - min_neg, min=0.0) + pos
    else: 
        print ('Unknown loss type. Try triplet_margin, softmax or contrastive')
        sys.exit(1)
    loss = torch.mean(loss)

    e_loss = torch.tensor(0.).float().cuda()
    
    sqdist_matrix_anchor_embeddings = 2 - 2 * anchor @ anchor.T
    sqdist_matrix_anchor = 2 - 2 * out_a_raw @ out_a_raw.T

    sqdist_matrix_positive_embeddings = 2 - 2 * positive @ positive.T
    sqdist_matrix_positive = 2 - 2 * out_p_raw @ out_p_raw.T

    sqdist_matrix_anchor_positive_embeddings = 2 - 2 * anchor @ positive.T
    sqdist_matrix_anchor_positive = 2 - 2 * out_a_raw @ out_p_raw.T

    e_loss += torch.mean(
        torch.abs(sqdist_matrix_anchor - sqdist_matrix_anchor_embeddings)
    )

    e_loss += torch.mean(
        torch.abs(sqdist_matrix_positive - sqdist_matrix_positive_embeddings)
    )

    e_loss += torch.mean(
        torch.abs(sqdist_matrix_anchor_positive - sqdist_matrix_anchor_positive_embeddings)
    )

    if alpha > 0:
        loss_sum = loss + alpha * e_loss
    elif alpha < 0:
        loss_sum = e_loss
    else:
        loss_sum = loss

    return loss_sum

def global_orthogonal_regularization(anchor, negative):

    neg_dis = torch.sum(torch.mul(anchor,negative),1)
    dim = anchor.size(1)
    gor = torch.pow(torch.mean(neg_dis),2) + torch.clamp(torch.mean(torch.pow(neg_dis,2))-1.0/dim, min=0.0)
    
    return gor

class Loss_HyNet():

    def __init__(self, device, dim_desc, margin, alpha, is_sosr, knn_sos=8):
        self.device = device
        self.margin = margin
        self.alpha = alpha
        self.is_sosr = is_sosr
        self.dim_desc = dim_desc
        self.knn_sos = knn_sos
        self.index_dim = torch.LongTensor(range(0, dim_desc))

    def sort_distance(self):
        L = self.L.clone().detach()
        L = L + 2 * self.mask_pos_pair
        L = L + 2 * L.le(dist_th).float()

        R = self.R.clone().detach()
        R = R + 2 * self.mask_pos_pair
        R = R + 2 * R.le(dist_th).float()

        LR = self.LR.clone().detach()
        LR = LR + 2 * self.mask_pos_pair
        LR = LR + 2 * LR.le(dist_th).float()

        self.indice_L = torch.argsort(L, dim=1)
        self.indice_R = torch.argsort(R, dim=0)
        self.indice_LR = torch.argsort(LR, dim=1)
        self.indice_RL = torch.argsort(LR, dim=0)
        return

    def triplet_loss_hybrid(self):
        L = self.L
        R = self.R
        LR = self.LR
        indice_L = self.indice_L[:, 0]
        indice_R = self.indice_R[0, :]
        indice_LR = self.indice_LR[:, 0]
        indice_RL = self.indice_RL[0, :]
        index_desc = self.index_desc

        dist_pos = LR[self.mask_pos_pair.bool()]
        dist_neg_LL = L[index_desc, indice_L]
        dist_neg_RR = R[indice_R, index_desc]
        dist_neg_LR = LR[index_desc, indice_LR]
        dist_neg_RL = LR[indice_RL, index_desc]
        dist_neg = torch.cat((dist_neg_LL.unsqueeze(0),
                              dist_neg_RR.unsqueeze(0),
                              dist_neg_LR.unsqueeze(0),
                              dist_neg_RL.unsqueeze(0)), dim=0)
        dist_neg_hard, index_neg_hard = torch.sort(dist_neg, dim=0)
        dist_neg_hard = dist_neg_hard[0, :]
        # scipy.io.savemat('dist.mat', dict(dist_pos=dist_pos.cpu().detach().numpy(), dist_neg=dist_neg_hard.cpu().detach().numpy()))

        loss_triplet = torch.clamp(self.margin + (dist_pos + dist_pos.pow(2)/2*self.alpha) - (dist_neg_hard + dist_neg_hard.pow(2)/2*self.alpha), min=0.0)

        self.num_triplet_display = loss_triplet.gt(0).sum()

        self.loss = self.loss + loss_triplet.sum()
        self.dist_pos_display = dist_pos.detach().mean()
        self.dist_neg_display = dist_neg_hard.detach().mean()

        return

    def norm_loss_pos(self):
        diff_norm = self.norm_L - self.norm_R
        self.loss += diff_norm.pow(2).sum().mul(0.1)

    def sos_loss(self):
        L = self.L
        R = self.R
        knn = self.knn_sos
        indice_L = self.indice_L[:, 0:knn]
        indice_R = self.indice_R[0:knn, :]
        indice_LR = self.indice_LR[:, 0:knn]
        indice_RL = self.indice_RL[0:knn, :]
        index_desc = self.index_desc
        num_pt_per_batch = self.num_pt_per_batch
        index_row = index_desc.unsqueeze(1).expand(-1, knn)
        index_col = index_desc.unsqueeze(0).expand(knn, -1)

        A_L = torch.zeros(num_pt_per_batch, num_pt_per_batch).to(self.device)
        A_R = torch.zeros(num_pt_per_batch, num_pt_per_batch).to(self.device)
        A_LR = torch.zeros(num_pt_per_batch, num_pt_per_batch).to(self.device)

        A_L[index_row, indice_L] = 1
        A_R[indice_R, index_col] = 1
        A_LR[index_row, indice_LR] = 1
        A_LR[indice_RL, index_col] = 1

        A_L = A_L + A_L.t()
        A_L = A_L.gt(0).float()
        A_R = A_R + A_R.t()
        A_R = A_R.gt(0).float()
        A_LR = A_LR + A_LR.t()
        A_LR = A_LR.gt(0).float()
        A = A_L + A_R + A_LR
        A = A.gt(0).float() * self.mask_neg_pair

        sturcture_dif = (L - R) * A
        self.loss = self.loss + sturcture_dif.pow(2).sum(dim=1).add(eps_sqrt).sqrt().sum()

        return

    def compute(self, desc_L, desc_R, desc_raw_L, desc_raw_R):
        num_pt_per_batch = desc_L.shape[0]
        self.num_pt_per_batch = num_pt_per_batch
        self.index_desc = torch.LongTensor(range(0, num_pt_per_batch))
        diagnal = torch.eye(num_pt_per_batch)
        self.mask_pos_pair = diagnal.eq(1).float().to(self.device)
        self.mask_neg_pair = diagnal.eq(0).float().to(self.device)
        self.desc_L = desc_L
        self.desc_R = desc_R
        self.desc_raw_L = desc_raw_L
        self.desc_raw_R = desc_raw_R
        self.norm_L = self.desc_raw_L.pow(2).sum(1).add(eps_sqrt).sqrt()
        self.norm_R = self.desc_raw_R.pow(2).sum(1).add(eps_sqrt).sqrt()
        self.L = cal_l2_distance_matrix(desc_L, desc_L)
        self.R = cal_l2_distance_matrix(desc_R, desc_R)
        self.LR = cal_l2_distance_matrix(desc_L, desc_R)

        self.loss = torch.Tensor([0]).to(self.device)

        self.sort_distance()
        self.triplet_loss_hybrid()
        self.norm_loss_pos()
        if self.is_sosr:
            self.sos_loss()

        return self.loss, self.dist_pos_display, self.dist_neg_display

class Loss_SOSNet():

    def __init__(self, device, dim_desc, margin, knn_sos=8):
        self.device = device
        self.margin = margin
        self.dim_desc = dim_desc
        self.knn_sos = knn_sos
        self.index_dim = torch.LongTensor(range(0, dim_desc))

    def sort_distance(self):
        L = self.L.clone().detach()
        L = L + 2 * self.mask_pos_pair
        L = L + 2 * L.le(dist_th).float()

        R = self.R.clone().detach()
        R = R + 2 * self.mask_pos_pair
        R = R + 2 * R.le(dist_th).float()

        LR = self.LR.clone().detach()
        LR = LR + 2 * self.mask_pos_pair
        LR = LR + 2 * LR.le(dist_th).float()

        self.indice_L = torch.argsort(L, dim=1)
        self.indice_R = torch.argsort(R, dim=0)
        self.indice_LR = torch.argsort(LR, dim=1)
        self.indice_RL = torch.argsort(LR, dim=0)
        return

    def triplet_loss(self):
        L = self.L
        R = self.R
        LR = self.LR
        indice_L = self.indice_L[:, 0]
        indice_R = self.indice_R[0, :]
        indice_LR = self.indice_LR[:, 0]
        indice_RL = self.indice_RL[0, :]
        index_desc = self.index_desc

        dist_neg_hard_L = torch.min(LR[index_desc, indice_LR], L[index_desc, indice_L])
        dist_neg_hard_R = torch.min(LR[indice_RL, index_desc], R[indice_R, index_desc])
        dist_neg_hard = torch.min(dist_neg_hard_L, dist_neg_hard_R)
        dist_pos = LR[self.mask_pos_pair.bool()]
        loss = torch.clamp(self.margin + dist_pos - dist_neg_hard, min=0.0)

        loss = loss.pow(2)

        self.loss = self.loss + loss.sum()
        self.dist_pos_display = dist_pos.detach().mean()
        self.dist_neg_display = dist_neg_hard.detach().mean()

        return

    def sos_loss(self):
        L = self.L
        R = self.R
        knn = self.knn_sos
        indice_L = self.indice_L[:, 0:knn]
        indice_R = self.indice_R[0:knn, :]
        indice_LR = self.indice_LR[:, 0:knn]
        indice_RL = self.indice_RL[0:knn, :]
        index_desc = self.index_desc
        num_pt_per_batch = self.num_pt_per_batch
        index_row = index_desc.unsqueeze(1).expand(-1, knn)
        index_col = index_desc.unsqueeze(0).expand(knn, -1)

        A_L = torch.zeros(num_pt_per_batch, num_pt_per_batch).to(self.device)
        A_R = torch.zeros(num_pt_per_batch, num_pt_per_batch).to(self.device)
        A_LR = torch.zeros(num_pt_per_batch, num_pt_per_batch).to(self.device)

        A_L[index_row, indice_L] = 1
        A_R[indice_R, index_col] = 1
        A_LR[index_row, indice_LR] = 1
        A_LR[indice_RL, index_col] = 1

        A_L = A_L + A_L.t()
        A_L = A_L.gt(0).float()
        A_R = A_R + A_R.t()
        A_R = A_R.gt(0).float()
        A_LR = A_LR + A_LR.t()
        A_LR = A_LR.gt(0).float()
        A = A_L + A_R + A_LR
        A = A.gt(0).float() * self.mask_neg_pair

        sturcture_dif = (L - R) * A
        self.loss = self.loss + sturcture_dif.pow(2).sum(dim=1).add(eps_sqrt).sqrt().sum()

        return

    def compute(self, desc_l, desc_r):
        num_pt_per_batch = desc_l.shape[0]
        self.num_pt_per_batch = num_pt_per_batch
        self.index_desc = torch.LongTensor(range(0, num_pt_per_batch))
        diagnal = torch.eye(num_pt_per_batch)
        self.mask_pos_pair = diagnal.eq(1).float().to(self.device)
        self.mask_neg_pair = diagnal.eq(0).float().to(self.device)
        self.loss = torch.Tensor([0]).to(self.device)
        self.L = cal_l2_distance_matrix(desc_l, desc_l)
        self.R = cal_l2_distance_matrix(desc_r, desc_r)
        self.LR = cal_l2_distance_matrix(desc_l, desc_r)
        self.sort_distance()
        self.triplet_loss()
        self.sos_loss()

        return self.loss, self.dist_pos_display, self.dist_neg_display

dist_th = 8e-3
eps_sqrt = 1e-6

def cal_l2_distance_matrix(x, y, flag_sqrt=True):
    ''''distance matrix of x with respect to y, d_ij is the distance between x_i and y_j'''
    D = torch.abs(2 * (1 - torch.mm(x, y.t())))
    if flag_sqrt:
        D = torch.sqrt(D + eps_sqrt)
    return D
