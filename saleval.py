import torch
import numpy as np

# adapted from https://github.com/shelhamer/fcn.berkeleyvision.org/blob/master/score.py
MAX_IMG_PER_BATCH = 256
import torch.nn.functional as F

class SalEval:
    def __init__(self, nthresh=50):
        self.nthresh = nthresh
        self.thresh = torch.linspace(1./(nthresh + 1), 1. - 1./(nthresh + 1), nthresh).cuda()
        self.EPSILON = np.finfo(np.float).eps

        self.gt_sum = torch.zeros((nthresh,)).cuda()
        self.pred_sum = torch.zeros((nthresh,)).cuda()
        self.num_images = 0
        self.mae = 0
        self.prec = torch.zeros(self.nthresh).cuda()
        self.recall = torch.zeros(self.nthresh).cuda()


    def add_batch(self, predict, gth):
        bs = predict.shape[0]
        assert(predict.shape[0] < MAX_IMG_PER_BATCH)
        predict = predict.detach()
        gth = gth.detach()
        gth.requires_grad = False
        predict.requires_grad = False
        #predict = F.interpolate(predict.unsqueeze(dim=0), size=(384,384), mode='bilinear', align_corners=False).squeeze(dim=0)
        #gth = F.interpolate(gth.unsqueeze(dim=0), size=(384,384), mode='nearest').squeeze(dim=0)
        recall = torch.zeros(self.nthresh).cuda()
        prec = torch.zeros(self.nthresh).cuda()
        mae = 0
        predict = predict.view(bs, -1)
        gth = gth.view(bs, -1)
        length = predict.shape[1]
        thres_mat = self.thresh.expand(bs, length, self.nthresh).permute(2, 0, 1)
        predict_ = predict.expand(self.nthresh, bs, length)
        gth_ = gth.expand(self.nthresh, bs, length)
        # nthres, n, length
        bi_res = (predict > thres_mat).float()
        intersect = (gth_ * bi_res).sum(dim=2) # nthres, n
        recall = (intersect / (gth_.sum(dim=2) + self.EPSILON)).sum(dim=1)
        prec = (intersect / (bi_res.sum(dim=2) + self.EPSILON)).sum(dim=1)
        mae = (predict_[0] - gth_[0]).abs().sum() / length

        """
        for i in range(predict.shape[0]):
            for t in range(self.nthresh):
                bi_res = (predict[i] > self.thresh[t]).float()
                intersect = (gth[i] * bi_res).sum()
                recall[t] += intersect * 1. / (gth[i].sum() + self.EPSILON)
                prec[t] += intersect * 1. / (bi_res.sum() + self.EPSILON)
            mae += (gth[i] - predict[i]).abs().sum() * 1. / gth[i].numel()
        """
        self.prec += prec
        self.recall += recall
        self.mae += mae
        self.num_images += bs

    def get_metric(self):
        prec = self.prec / self.num_images
        recall = self.recall / self.num_images
        F_beta = (1 + 0.3) * prec * recall / (0.3 * prec + recall + self.EPSILON)
        MAE = self.mae / self.num_images
        return F_beta.max().item(),  MAE.item()

