from __future__ import print_function, absolute_import
from audioop import cross
import time
from .utils.meters import AverageMeter
import torch.nn as nn
import torch
from torch.nn import functional as F
import math




def pdist_torch(emb1, emb2):
    '''
    compute the eucilidean distance matrix between embeddings1 and embeddings2
    using gpu
    '''
    m, n = emb1.shape[0], emb2.shape[0]
    emb1_pow = torch.pow(emb1, 2).sum(dim = 1, keepdim = True).expand(m, n)
    emb2_pow = torch.pow(emb2, 2).sum(dim = 1, keepdim = True).expand(n, m).t()
    dist_mtx = emb1_pow + emb2_pow
    dist_mtx = dist_mtx.addmm_(1, -2, emb1, emb2.t())
    # dist_mtx = dist_mtx.clamp(min = 1e-12)
    dist_mtx = dist_mtx.clamp(min = 1e-12).sqrt()
    return dist_mtx 
def softmax_weights(dist, mask):
    max_v = torch.max(dist * mask, dim=1, keepdim=True)[0]
    diff = dist - max_v
    Z = torch.sum(torch.exp(diff) * mask, dim=1, keepdim=True) + 1e-6 # avoid division by zero
    W = torch.exp(diff) * mask / Z
    return W
def normalize(x, axis=-1):
    """Normalizing to unit length along the specified dimension.
    Args:
      x: pytorch Variable
    Returns:
      x: pytorch Variable, same shape as input
    """
    x = 1. * x / (torch.norm(x, 2, axis, keepdim=True).expand_as(x) + 1e-12)
    return x

class ClusterContrastTrainer(object):
    def __init__(self, encoder, memory=None):
        super(ClusterContrastTrainer, self).__init__()
        self.encoder = encoder
        self.memory_ir = memory
        self.memory_rgb = memory
        # self.tri = TripletLoss_ADP(alpha = 1, gamma = 1, square = 1)

    def train_s1(self, epoch, data_loader_ir,data_loader_rgb, optimizer, print_freq=10, train_iters=400, i2r=None, r2i=None):
        self.encoder.train()

        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        criterion_tri = OriTripletLoss(256, 0.3) # (batchsize, margin)


        end = time.time()

        for i in range(train_iters):

            # load data
            inputs_ir = data_loader_ir.next()
            inputs_rgb = data_loader_rgb.next()
            data_time.update(time.time() - end)

            # process inputs
            inputs_ir, labels_ir, indexes_ir = self._parse_data_ir(inputs_ir)
            inputs_rgb,inputs_rgb1, labels_rgb, indexes_rgb = self._parse_data_rgb(inputs_rgb)
            # KL any?

            #targets_soft_rgb = torch.zeros(log_probs_ir.size()).scatter_(1, labels_ir.unsqueeze(1).data.cpu(), 1)

            # forward
            inputs_rgb = torch.cat((inputs_rgb,inputs_rgb1),0)
            labels_rgb = torch.cat((labels_rgb,labels_rgb),-1)
            


            _,f_out_rgb,f_out_ir,labels_rgb,labels_ir,pool_rgb,pool_ir = self._forward(inputs_rgb,inputs_ir,label_1=labels_rgb,label_2=labels_ir,modal=0)


            loss_ir = self.memory_ir(f_out_ir, labels_ir)
            loss_rgb = self.memory_rgb(f_out_rgb, labels_rgb)

            #cross_loss = torch.tensor(0.0)
            new_loss_rgb = loss_rgb
            #new_cross_loss = cross_loss


            loss = loss_ir+new_loss_rgb #+ self.lambda1 * new_cross_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            losses.update(loss.item())

            # print log
            batch_time.update(time.time() - end)
            end = time.time()

            if (i + 1) % print_freq == 0:
                print('Epoch: [{}][{}/{}]\t'
                      'Time {:.3f} ({:.3f})\t'
                      'Data {:.3f} ({:.3f})\t'
                      'Loss {:.3f} ({:.3f})\t'
                      'Loss ir {:.3f}\t'
                      'Loss rgb {:.3f}\t'
                    #   'Loss tri rgb {:.3f}\t'
                    #   'Loss tri ir {:.3f}\t'
                      .format(epoch, i + 1, len(data_loader_rgb),
                              batch_time.val, batch_time.avg,
                              data_time.val, data_time.avg,
                              losses.val, losses.avg,loss_ir,new_loss_rgb
                            #   , loss_tri_rgb
                            # , loss_tri_ir
                              ))



    def train(self, epoch, data_loader_ir,data_loader_rgb, optimizer, print_freq=10, train_iters=400, i2r=None, r2i=None):
        self.encoder.train()

        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        criterion_tri = OriTripletLoss(256, 0.3) # (batchsize, margin)


        end = time.time()

        for i in range(train_iters):

            # load data
            inputs_ir = data_loader_ir.next()
            inputs_rgb = data_loader_rgb.next()
            data_time.update(time.time() - end)

            # process inputs
            inputs_ir, labels_ir, indexes_ir = self._parse_data_ir(inputs_ir)
            inputs_rgb,inputs_rgb1, labels_rgb, indexes_rgb = self._parse_data_rgb(inputs_rgb)
            # KL any?
            targets_soft_rgb = []
            targets_soft_ir = []
            weights_rgb = []
            weights_ir = []
            

            for idx in indexes_rgb:
                targets_soft_rgb.append(self.correlations_rgb[idx.item()])
                weights_rgb.append(self.weights_rgb[idx.item()])
            for idx in indexes_ir:
                targets_soft_ir.append(self.correlations_ir[idx.item()])
                weights_ir.append(self.weights_ir[idx.item()])

            weights_rgb = torch.tensor(weights_rgb).cuda()
            weights_ir = torch.tensor(weights_ir).cuda()




            targets_soft_rgb = torch.stack(targets_soft_rgb, dim=0)
            targets_soft_ir = torch.stack(targets_soft_ir, dim=0)

            #targets_soft_rgb = torch.zeros(log_probs_ir.size()).scatter_(1, labels_ir.unsqueeze(1).data.cpu(), 1)

            # forward
            inputs_rgb = torch.cat((inputs_rgb,inputs_rgb1),0)
            labels_rgb = torch.cat((labels_rgb,labels_rgb),-1)
            targets_soft_rgb = torch.cat((targets_soft_rgb, targets_soft_rgb), 0)
            weights_rgb = torch.cat((weights_rgb, weights_rgb), -1)
            
 

            _,f_out_rgb,f_out_ir,labels_rgb,labels_ir,pool_rgb,pool_ir = self._forward(inputs_rgb,inputs_ir,label_1=labels_rgb,label_2=labels_ir,modal=0)


            feats_ir = self.memory_ir(f_out_ir, labels_ir, return_out=True)
            feats_rgb = self.memory_rgb(f_out_rgb, labels_rgb, return_out=True)

            targets_soft_rgb = (self.balance * torch.zeros(feats_rgb.size()).scatter_(1, labels_rgb.unsqueeze(1).data.cpu(), 1) + (1.0 - self.balance) * targets_soft_rgb).cuda()
            targets_soft_ir = (self.balance * torch.zeros(feats_ir.size()).scatter_(1, labels_ir.unsqueeze(1).data.cpu(), 1) + (1.0 - self.balance) * targets_soft_ir).cuda()

            #targets_soft_rgb = torch.zeros(feats_rgb.size()).scatter_(1, labels_rgb.unsqueeze(1).data.cpu(), 1).cuda() 
            #targets_soft_ir = torch.zeros(feats_ir.size()).scatter_(1, labels_ir.unsqueeze(1).data.cpu(), 1).cuda()
            log_probs_ir = nn.LogSoftmax(dim=1)(feats_ir)
            log_probs_rgb = nn.LogSoftmax(dim=1)(feats_rgb)

            loss_ir = ((- targets_soft_ir * log_probs_ir).sum(1) * weights_ir).mean()
            loss_rgb = ((- targets_soft_rgb * log_probs_rgb).sum(1) * weights_rgb).mean()

            #loss_ir = ((- targets_soft_ir * log_probs_ir).sum(1)).mean()
            #loss_rgb = ((- targets_soft_rgb * log_probs_rgb).sum(1)).mean()


            # cross contrastive learning
            if r2i:
                rgb2ir_labels = []
                ir2rgb_labels = []
                f_out_rgb_select = []
                f_out_ir_select = []

                rgb_weights_cross = []
                ir_weights_cross = []

                targets_soft_rgb_cross = []
                targets_soft_ir_cross = []
                for idx in indexes_rgb:
                    targets_soft_rgb_cross.append(self.correlations_rgb_cross[idx.item()])
                    rgb_weights_cross.append(self.weights_rgb_cross[idx.item()])
                for idx in indexes_ir:
                    targets_soft_ir_cross.append(self.correlations_ir_cross[idx.item()])
                    ir_weights_cross.append(self.weights_ir_cross[idx.item()])

                targets_soft_rgb_cross = torch.stack(targets_soft_rgb_cross, dim=0)
                targets_soft_ir_cross = torch.stack(targets_soft_ir_cross, dim=0)

                rgb_weights_cross = torch.tensor(rgb_weights_cross).cuda()
                ir_weights_cross = torch.tensor(ir_weights_cross).cuda()

                rgb_weights_cross = torch.cat((rgb_weights_cross, rgb_weights_cross), -1)

                targets_soft_rgb_cross = torch.cat((targets_soft_rgb_cross, targets_soft_rgb_cross), 0)

                assert f_out_rgb.shape[0] == labels_rgb.shape[0]
                assert f_out_ir.shape[0] == labels_ir.shape[0]

                for idx in range(f_out_rgb.shape[0]):
                    assert labels_rgb[idx].item() in r2i.keys()
                    f_out_rgb_select.append(f_out_rgb[idx])
                    rgb2ir_labels.append(r2i[labels_rgb[idx].item()])

                for idx in range(f_out_ir.shape[0]):
                    assert labels_ir[idx].item() in i2r.keys()
                    f_out_ir_select.append(f_out_ir[idx])
                    ir2rgb_labels.append(i2r[labels_ir[idx].item()])

                
                f_out_rgb_select = torch.stack(f_out_rgb_select, dim=0)
                rgb2ir_labels = torch.tensor(rgb2ir_labels).cuda().long()
                

                
                feats_rgb_cross = self.memory_ir(f_out_rgb_select, rgb2ir_labels.long(), return_out=True)
            
                
                f_out_ir_select = torch.stack(f_out_ir_select, dim=0)
                ir2rgb_labels = torch.tensor(ir2rgb_labels).cuda().long()

                feats_ir_cross = self.memory_rgb(f_out_ir_select, ir2rgb_labels.long(), return_out=True)

                targets_soft_rgb_cross = (self.balance * torch.zeros(feats_rgb_cross.size()).scatter_(1, rgb2ir_labels.unsqueeze(1).data.cpu(), 1) + (1.0 - self.balance) * targets_soft_rgb_cross).cuda()
                targets_soft_ir_cross = (self.balance * torch.zeros(feats_ir_cross.size()).scatter_(1, ir2rgb_labels.unsqueeze(1).data.cpu(), 1) + (1.0 - self.balance) * targets_soft_ir_cross).cuda()

                log_probs_ir_cross = nn.LogSoftmax(dim=1)(feats_ir_cross)
                log_probs_rgb_cross = nn.LogSoftmax(dim=1)(feats_rgb_cross)
                
                cross_loss_ir = ((- targets_soft_ir_cross * log_probs_ir_cross).sum(1) * ir_weights_cross).mean()
                cross_loss_rgb = ((- targets_soft_rgb_cross * log_probs_rgb_cross).sum(1) * rgb_weights_cross).mean()
                

                alternate = True
                if alternate:
                    # accl
                    if epoch % 2 == 1:
                        cross_loss = cross_loss_ir   #1 * self.memory_rgb(f_out_ir_select, ir2rgb_labels.long())
                    else:
                        cross_loss = cross_loss_rgb   #1 * self.memory_ir(f_out_rgb_select, rgb2ir_labels.long())
                else:
                    cross_loss = cross_loss_ir + cross_loss_rgb   
                
            else:
                cross_loss = torch.tensor(0.0)

            new_loss_rgb = loss_rgb
            new_cross_loss = cross_loss

            
            loss = loss_ir+new_loss_rgb + self.lambda1 * new_cross_loss 
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            losses.update(loss.item())

            # print log
            batch_time.update(time.time() - end)
            end = time.time()

            if (i + 1) % print_freq == 0:
                print('Epoch: [{}][{}/{}]\t'
                      'Time {:.3f} ({:.3f})\t'
                      'Data {:.3f} ({:.3f})\t'
                      'Loss {:.3f} ({:.3f})\t'
                      'Loss ir {:.3f}\t'
                      'Loss rgb {:.3f}\t'
                      'Loss cross {:.3f}\t'
                    #   'Loss tri rgb {:.3f}\t'
                    #   'Loss tri ir {:.3f}\t'
                      .format(epoch, i + 1, len(data_loader_rgb),
                              batch_time.val, batch_time.avg,
                              data_time.val, data_time.avg,
                              losses.val, losses.avg,loss_ir,new_loss_rgb,new_cross_loss
                            #   , loss_tri_rgb
                            # , loss_tri_ir
                              ))

    def _parse_data_rgb(self, inputs):
        imgs,imgs1, _, pids, _, indexes = inputs
        return imgs.cuda(),imgs1.cuda(), pids.cuda(), indexes.cuda()

    def _parse_data_ir(self, inputs):
        imgs, _, pids, _, indexes = inputs
        return imgs.cuda(), pids.cuda(), indexes.cuda()

    def _forward(self, x1, x2, label_1=None,label_2=None,modal=0):
        return self.encoder(x1, x2, modal=modal,label_1=label_1,label_2=label_2)


class OriTripletLoss(nn.Module):
    """Triplet loss with hard positive/negative mining.
    
    Args:
    - margin (float): margin for triplet.
    """
    
    def __init__(self, batch_size, margin=0.3):
        super(OriTripletLoss, self).__init__()
        self.margin = margin
        self.ranking_loss = nn.MarginRankingLoss(margin=margin)

    def forward(self, inputs, targets):
        """
        Args:
        - inputs: feature matrix with shape (batch_size, feat_dim)
        - targets: ground truth labels with shape (num_classes)
        """
        n = inputs.size(0)
        
        # Compute pairwise distance, replace by the official when merged
        dist = torch.pow(inputs, 2).sum(dim=1, keepdim=True).expand(n, n)
        dist = dist + dist.t()
        dist.addmm_(1, -2, inputs, inputs.t())
        dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
        
        # For each anchor, find the hardest positive and negative
        mask = targets.expand(n, n).eq(targets.expand(n, n).t())
        dist_ap, dist_an = [], []
        for i in range(n):
            dist_ap.append(dist[i][mask[i]].max().unsqueeze(0))
            dist_an.append(dist[i][mask[i] == 0].min().unsqueeze(0))
        dist_ap = torch.cat(dist_ap)
        dist_an = torch.cat(dist_an)
        
        # Compute ranking hinge loss
        y = torch.ones_like(dist_an)
        loss = self.ranking_loss(dist_an, dist_ap, y)
        
        # compute accuracy
        correct = torch.ge(dist_an, dist_ap).sum().item()
        return loss, correct
