# -*- coding: utf-8 -*-
from __future__ import print_function, absolute_import
import argparse
import os.path as osp
import random
import numpy as np
import sys
import collections
import time
from datetime import timedelta

from sklearn.cluster import DBSCAN
from PIL import Image
import torch
from torch import nn
from torch.backends import cudnn
from torch.utils.data import DataLoader
import torch.nn.functional as F

from clustercontrast import datasets
from clustercontrast import models
from clustercontrast.models.cm import ClusterMemory
from clustercontrast.trainers import ClusterContrastTrainer
from clustercontrast.evaluators import Evaluator, extract_features
from clustercontrast.utils.data import IterLoader
from clustercontrast.utils.data import transforms as T
from clustercontrast.utils.data.preprocessor import Preprocessor,Preprocessor_color
from clustercontrast.utils.logging import Logger
from clustercontrast.utils.serialization import load_checkpoint, save_checkpoint
from clustercontrast.utils.faiss_rerank import compute_jaccard_distance
from clustercontrast.utils.data.sampler import RandomMultipleGallerySampler, RandomMultipleGallerySamplerNoCam
import os
import torch.utils.data as data
from torch.autograd import Variable
from scipy.optimize import linear_sum_assignment

import math
from ChannelAug import ChannelAdap, ChannelAdapGray, ChannelRandomErasing,ChannelExchange,Gray
from collections import Counter

import copy

from sklearn.metrics.cluster import adjusted_rand_score
from sklearn.metrics.cluster import normalized_mutual_info_score
from sklearn.metrics.cluster import adjusted_mutual_info_score
from sklearn.metrics.cluster import fowlkes_mallows_score


def get_data(name, data_dir):
    root = osp.join(data_dir, name)
    dataset = datasets.create(name, root)
    return dataset

def get_train_loader_ir(args, dataset, height, width, batch_size, workers,
                     num_instances, iters, trainset=None, no_cam=False,train_transformer=None):



    train_set = sorted(dataset.train) if trainset is None else sorted(trainset)
    rmgs_flag = num_instances > 0
    if rmgs_flag:
        if no_cam:
            sampler = RandomMultipleGallerySamplerNoCam(train_set, num_instances)
        else:
            sampler = RandomMultipleGallerySampler(train_set, num_instances)
    else:
        sampler = None
    train_loader = IterLoader(
        DataLoader(Preprocessor(train_set, root=dataset.images_dir, transform=train_transformer),
                   batch_size=batch_size, num_workers=workers, sampler=sampler,
                   shuffle=not rmgs_flag, pin_memory=True, drop_last=True), length=iters)

    return train_loader

def get_train_loader_color(args, dataset, height, width, batch_size, workers,
                     num_instances, iters, trainset=None, no_cam=False,train_transformer=None,train_transformer1=None):



    train_set = sorted(dataset.train) if trainset is None else sorted(trainset)
    rmgs_flag = num_instances > 0
    if rmgs_flag:
        if no_cam:
            sampler = RandomMultipleGallerySamplerNoCam(train_set, num_instances)
        else:
            sampler = RandomMultipleGallerySampler(train_set, num_instances)
    else:
        sampler = None
    if train_transformer1 is None:
        train_loader = IterLoader(
            DataLoader(Preprocessor(train_set, root=dataset.images_dir, transform=train_transformer),
                       batch_size=batch_size, num_workers=workers, sampler=sampler,
                       shuffle=not rmgs_flag, pin_memory=True, drop_last=True), length=iters)
    else:
        train_loader = IterLoader(
            DataLoader(Preprocessor_color(train_set, root=dataset.images_dir, transform=train_transformer,transform1=train_transformer1),
                       batch_size=batch_size, num_workers=workers, sampler=sampler,
                       shuffle=not rmgs_flag, pin_memory=True, drop_last=True), length=iters)

    return train_loader


def get_test_loader(dataset, height, width, batch_size, workers, testset=None,test_transformer=None):
    normalizer = T.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    if test_transformer is None:
        test_transformer = T.Compose([
            T.Resize((height, width), interpolation=3),
            T.ToTensor(),
            normalizer
        ])

    if testset is None:
        testset = list(set(dataset.query) | set(dataset.gallery))

    test_loader = DataLoader(
        Preprocessor(testset, root=dataset.images_dir, transform=test_transformer),
        batch_size=batch_size, num_workers=workers,
        shuffle=False, pin_memory=True)

    return test_loader


def create_model(args):
    model = models.create(args.arch, num_features=args.features, norm=True, dropout=args.dropout,
                          num_classes=0, pooling_type=args.pooling_type)
    # use CUDA
    model.cuda()
    model = nn.DataParallel(model)#,output_device=1)
    return model



class TestData(data.Dataset):
    def __init__(self, test_img_file, test_label, transform=None, img_size = (144,288)):

        test_image = []
        for i in range(len(test_img_file)):
            img = Image.open(test_img_file[i])
            img = img.resize((img_size[0], img_size[1]), Image.ANTIALIAS)
            pix_array = np.array(img)
            test_image.append(pix_array)
        test_image = np.array(test_image)
        self.test_image = test_image
        self.test_label = test_label
        self.transform = transform

    def __getitem__(self, index):
        img1,  target1 = self.test_image[index],  self.test_label[index]
        img1 = self.transform(img1)
        return img1, target1

    def __len__(self):
        return len(self.test_image)

def process_query_sysu(data_path, mode = 'all', relabel=False):
    if mode== 'all':
        ir_cameras = ['cam3','cam6']
    elif mode =='indoor':
        ir_cameras = ['cam3','cam6']
    
    file_path = os.path.join(data_path,'exp/test_id.txt')
    files_rgb = []
    files_ir = []

    with open(file_path, 'r') as file:
        ids = file.read().splitlines()
        ids = [int(y) for y in ids[0].split(',')]
        ids = ["%04d" % x for x in ids]

    for id in sorted(ids):
        for cam in ir_cameras:
            img_dir = os.path.join(data_path,cam,id)
            if os.path.isdir(img_dir):
                new_files = sorted([img_dir+'/'+i for i in os.listdir(img_dir)])
                files_ir.extend(new_files)
    query_img = []
    query_id = []
    query_cam = []
    for img_path in files_ir:
        camid, pid = int(img_path[-15]), int(img_path[-13:-9])
        query_img.append(img_path)
        query_id.append(pid)
        query_cam.append(camid)
    return query_img, np.array(query_id), np.array(query_cam)

def process_gallery_sysu(data_path, mode = 'all', trial = 0, relabel=False):
    
    random.seed(trial)
    
    if mode== 'all':
        rgb_cameras = ['cam1','cam2','cam4','cam5']
    elif mode =='indoor':
        rgb_cameras = ['cam1','cam2']
        
    file_path = os.path.join(data_path,'exp/test_id.txt')
    files_rgb = []
    with open(file_path, 'r') as file:
        ids = file.read().splitlines()
        ids = [int(y) for y in ids[0].split(',')]
        ids = ["%04d" % x for x in ids]

    for id in sorted(ids):
        for cam in rgb_cameras:
            img_dir = os.path.join(data_path,cam,id)
            if os.path.isdir(img_dir):
                new_files = sorted([img_dir+'/'+i for i in os.listdir(img_dir)])
                files_rgb.append(random.choice(new_files))
    gall_img = []
    gall_id = []
    gall_cam = []
    for img_path in files_rgb:
        camid, pid = int(img_path[-15]), int(img_path[-13:-9])
        gall_img.append(img_path)
        gall_id.append(pid)
        gall_cam.append(camid)
    return gall_img, np.array(gall_id), np.array(gall_cam)
    

def select_max(sim, idx):
    minus_tensor = torch.zeros(sim.size())
    assert len(minus_tensor.size()) == 1
    assert len(sim.size()) == 1
    minus_tensor[idx] = 2
    next_val, next_idx = torch.max((sim - minus_tensor), dim=-1)

    return next_val, next_idx

def generate_correspondence(cluster_features_rgb, cluster_features_ir, radio):
    r2i = {}
    print('cluster_features_rgb', cluster_features_rgb.shape)
    print('cluster_features_ir', cluster_features_ir.shape)
    sim_rgb_ir = cluster_features_rgb.mm(cluster_features_ir.t())
    sim_ir_rgb = cluster_features_ir.mm(cluster_features_rgb.t())
    print('sim_rgb_ir', sim_rgb_ir.shape)
    print('sim_ir_rgb', sim_ir_rgb.shape)

    sim_thres = []

    for i in range(sim_rgb_ir.shape[0]):
        max_val, max_idx = torch.max(sim_rgb_ir[i], dim=-1)
        next_val, next_idx = select_max(sim_ir_rgb[max_idx.item()], i)

        assert next_idx.item() != i

        sim_thres.append(max_val.item())

    sim_thres = torch.tensor(sim_thres)
    num_radio = int(sim_thres.shape[0] * radio)
    top_vals, top_inds = torch.topk(sim_thres, num_radio, dim=-1, largest=True, sorted=True)

    for i in range(top_inds.shape[0]):
        select_idx = top_inds[i].item()
        val, correspondence = torch.max(sim_rgb_ir[select_idx], dim=-1)

        r2i[select_idx] = correspondence.item()


    return r2i


def fliplr(img):
    '''flip horizontal'''
    inv_idx = torch.arange(img.size(3)-1,-1,-1).long()  # N x C x H x W
    img_flip = img.index_select(3,inv_idx)
    return img_flip
def extract_gall_feat(model,gall_loader,ngall):
    pool_dim=2048
    net = model
    net.eval()
    print ('Extracting Gallery Feature...')
    start = time.time()
    ptr = 0
    gall_feat_pool = np.zeros((ngall, pool_dim))
    gall_feat_fc = np.zeros((ngall, pool_dim))
    with torch.no_grad():
        for batch_idx, (input, label ) in enumerate(gall_loader):
            batch_num = input.size(0)
            flip_input = fliplr(input)
            input = Variable(input.cuda())
            feat_fc = net( input,input, 1)
            flip_input = Variable(flip_input.cuda())
            feat_fc_1 = net( flip_input,flip_input, 1)
            feature_fc = (feat_fc.detach() + feat_fc_1.detach())/2
            fnorm_fc = torch.norm(feature_fc, p=2, dim=1, keepdim=True)
            feature_fc = feature_fc.div(fnorm_fc.expand_as(feature_fc))
            gall_feat_fc[ptr:ptr+batch_num,: ]   = feature_fc.cpu().numpy()
            ptr = ptr + batch_num
    print('Extracting Time:\t {:.3f}'.format(time.time()-start))
    return gall_feat_fc
    
def extract_query_feat(model,query_loader,nquery):
    pool_dim=2048
    net = model
    net.eval()
    print ('Extracting Query Feature...')
    start = time.time()
    ptr = 0
    query_feat_pool = np.zeros((nquery, pool_dim))
    query_feat_fc = np.zeros((nquery, pool_dim))
    with torch.no_grad():
        for batch_idx, (input, label ) in enumerate(query_loader):
            batch_num = input.size(0)
            flip_input = fliplr(input)
            input = Variable(input.cuda())
            feat_fc = net( input, input,2)
            flip_input = Variable(flip_input.cuda())
            feat_fc_1 = net( flip_input,flip_input, 2)
            feature_fc = (feat_fc.detach() + feat_fc_1.detach())/2
            fnorm_fc = torch.norm(feature_fc, p=2, dim=1, keepdim=True)
            feature_fc = feature_fc.div(fnorm_fc.expand_as(feature_fc))
            query_feat_fc[ptr:ptr+batch_num,: ]   = feature_fc.cpu().numpy()
            
            ptr = ptr + batch_num         
    print('Extracting Time:\t {:.3f}'.format(time.time()-start))
    return query_feat_fc



def eval_sysu(distmat, q_pids, g_pids, q_camids, g_camids, max_rank = 20):
    """Evaluation with sysu metric
    Key: for each query identity, its gallery images from the same camera view are discarded. "Following the original setting in ite dataset"
    """
    num_q, num_g = distmat.shape
    if num_g < max_rank:
        max_rank = num_g
        print("Note: number of gallery samples is quite small, got {}".format(num_g))
    indices = np.argsort(distmat, axis=1)
    pred_label = g_pids[indices]
    matches = (g_pids[indices] == q_pids[:, np.newaxis]).astype(np.int32)
    
    # compute cmc curve for each query
    new_all_cmc = []
    all_cmc = []
    all_AP = []
    all_INP = []
    num_valid_q = 0. # number of valid query
    for q_idx in range(num_q):
        # get query pid and camid
        q_pid = q_pids[q_idx]
        q_camid = q_camids[q_idx]

        # remove gallery samples that have the same pid and camid with query
        order = indices[q_idx]
        remove = (q_camid == 3) & (g_camids[order] == 2)
        keep = np.invert(remove)
        
        # compute cmc curve
        # the cmc calculation is different from standard protocol
        # we follow the protocol of the author's released code
        new_cmc = pred_label[q_idx][keep]
        new_index = np.unique(new_cmc, return_index=True)[1]
        new_cmc = [new_cmc[index] for index in sorted(new_index)]
        
        new_match = (new_cmc == q_pid).astype(np.int32)
        new_cmc = new_match.cumsum()
        new_all_cmc.append(new_cmc[:max_rank])
        
        orig_cmc = matches[q_idx][keep] # binary vector, positions with value 1 are correct matches
        if not np.any(orig_cmc):
            # this condition is true when query identity does not appear in gallery
            continue

        cmc = orig_cmc.cumsum()

        # compute mINP
        # refernece Deep Learning for Person Re-identification: A Survey and Outlook
        pos_idx = np.where(orig_cmc == 1)
        pos_max_idx = np.max(pos_idx)
        inp = cmc[pos_max_idx]/ (pos_max_idx + 1.0)
        all_INP.append(inp)

        cmc[cmc > 1] = 1

        all_cmc.append(cmc[:max_rank])
        num_valid_q += 1.

        # compute average precision
        # reference: https://en.wikipedia.org/wiki/Evaluation_measures_(information_retrieval)#Average_precision
        num_rel = orig_cmc.sum()
        tmp_cmc = orig_cmc.cumsum()
        tmp_cmc = [x / (i+1.) for i, x in enumerate(tmp_cmc)]
        tmp_cmc = np.asarray(tmp_cmc) * orig_cmc
        AP = tmp_cmc.sum() / num_rel
        all_AP.append(AP)

    assert num_valid_q > 0, "Error: all query identities do not appear in gallery"
    
    all_cmc = np.asarray(all_cmc).astype(np.float32)
    all_cmc = all_cmc.sum(0) / num_valid_q   # standard CMC
    
    new_all_cmc = np.asarray(new_all_cmc).astype(np.float32)
    new_all_cmc = new_all_cmc.sum(0) / num_valid_q
    mAP = np.mean(all_AP)
    mINP = np.mean(all_INP)
    return new_all_cmc, mAP, mINP

def pairwise_distance(features_q, features_g):
    x = torch.from_numpy(features_q)
    y = torch.from_numpy(features_g)
    m, n = x.size(0), y.size(0)
    x = x.view(m, -1)
    y = y.view(n, -1)
    dist_m = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(m, n) + \
           torch.pow(y, 2).sum(dim=1, keepdim=True).expand(n, m).t()
    dist_m.addmm_(1, -2, x, y.t())
    return dist_m.numpy()


def generate_correlations(features, pseudo_labels, cluster_nums, neighbour_num, coe):
    features = F.normalize(features, dim=1)
    sim = features.mm(features.t())
    assert sim.shape[0] == pseudo_labels.shape[0]
    _, inds = torch.topk(sim, neighbour_num, dim=1, largest=True, sorted=True)
    correlations_all = []
    weights_all = []

    cluster_nums = torch.tensor(cluster_nums)


    for i in range(inds.shape[0]):
        correlation = torch.zeros(cluster_nums.shape[0])
        for j in range(inds.shape[1]):
            pid = pseudo_labels[inds[i, j].item()]
            if pid.item() != -1:
                correlation[pid.item()] += 1


        correlation = correlation / (neighbour_num - correlation + cluster_nums)

        correlation /= correlation.sum()
        correlations_all.append(correlation)
        #weight = torch.log(correlation[pseudo_labels[i].item()]+1.0)
        if pseudo_labels[i].item() == -1:
            weights_all.append(0)
        else:
            weight = torch.exp(-coe * (1.0 - correlation[pseudo_labels[i].item()]) * (1.0 - correlation[pseudo_labels[i].item()]))
            weights_all.append(weight)
    correlations_all = torch.stack(correlations_all, dim=0)
    weights_all = torch.tensor(weights_all)
    return correlations_all, weights_all

def generate_correlations_cross(features, features_cross, pseudo_labels, cluster_nums, neighbour_num, pseudo_labels_own, pid_transform, coe):
    features = F.normalize(features, dim=1)
    features_cross = F.normalize(features_cross, dim=1)
    sim = features.mm(features_cross.t())
    assert sim.shape[1] == pseudo_labels.shape[0]
    assert sim.shape[0] == pseudo_labels_own.shape[0]
    _, inds = torch.topk(sim, neighbour_num, dim=1, largest=True, sorted=True)
    correlations_all = []
    weights_all = []

    cluster_nums = torch.tensor(cluster_nums)

    for i in range(inds.shape[0]):
        correlation = torch.zeros(cluster_nums.shape[0])
        for j in range(inds.shape[1]):
            pid = pseudo_labels[inds[i, j].item()]
            if pid.item() != -1:
                correlation[pid.item()] += 1

        correlation = correlation / (neighbour_num - correlation + cluster_nums)

        correlation /= correlation.sum()
        correlations_all.append(correlation)
        if pseudo_labels_own[i].item() == -1:
            weights_all.append(0)
        else:
            weight = torch.exp(-coe * (1.0 - correlation[pid_transform[pseudo_labels_own[i].item()]]) * (1.0 - correlation[pid_transform[pseudo_labels_own[i].item()]]))
            weights_all.append(weight)
    correlations_all = torch.stack(correlations_all, dim=0)
    weights_all = torch.tensor(weights_all)
    return correlations_all, weights_all


def main():
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        cudnn.benchmark = False
    log_s1_name = 'sysu_s1'
    log_s2_name = 'sysu_s2'
    assert args.stage in ['s1', 's2']
    if args.stage == 's1':
        main_worker_stage1(args,log_s1_name) # Stage 1
    else:
        main_worker_stage2(args,log_s1_name,log_s2_name) # Stage 2
    #main_worker_stage1(args,log_s1_name) # Stage 1
    #main_worker_stage2(args,log_s1_name,log_s2_name) # Stage 2

def main_worker_stage1(args,log_s1_name):
    start_epoch=0
    best_mAP=0
    stage1_logs_dir = osp.join(args.logs_dir+'/'+log_s1_name)
    start_time = time.monotonic()
    # cudnn.benchmark = True
    sys.stdout = Logger(osp.join(stage1_logs_dir, 'log.txt'))
    print("==========\nArgs:{}\n==========".format(args))
    # Create datasets
    iters = args.iters if (args.iters > 0) else None
    print("==> Load unlabeled dataset")
    dataset_ir = get_data('sysu_ir', args.data_dir)
    dataset_rgb = get_data('sysu_rgb', args.data_dir)

    # Create model
    model = create_model(args)
    # Optimizer
    params = [{"params": [value]} for _, value in model.named_parameters() if value.requires_grad]
    optimizer = torch.optim.Adam(params, lr=args.lr, weight_decay=args.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=0.1)
    # Trainer
    trainer = ClusterContrastTrainer(model)
    # ########################
    normalizer = T.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
    height=args.height
    width=args.width
    train_transformer_rgb = T.Compose([
        T.Resize((height, width), interpolation=3),
        T.Pad(10),
        T.RandomCrop((height, width)),
        T.RandomHorizontalFlip(p=0.5),
        # T.RandomGrayscale(p=0.1),
        T.ToTensor(),
        normalizer,
        ChannelRandomErasing(probability = 0.5)
    ])
    
    train_transformer_rgb1 = T.Compose([
        T.Resize((height, width), interpolation=3),
        T.Pad(10),
        T.RandomCrop((height, width)),
        T.RandomHorizontalFlip(p=0.5),
        T.ColorJitter(brightness=0.5,contrast=0.5,saturation=0.5,hue=0.5),
        T.ToTensor(),
        normalizer,
        ChannelRandomErasing(probability = 0.5),
        ChannelExchange(gray = 2)#2, TODO
    ])

    transform_thermal = T.Compose( [
        T.Resize((height, width), interpolation=3),
        T.Pad(10),
        T.RandomCrop((288, 144)),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        normalizer,
        ChannelRandomErasing(probability = 0.5),
        ChannelAdapGray(probability =0.5)])
    
    for epoch in range(args.epochs):
        with torch.no_grad():
            if epoch == 0:
                # DBSCAN cluster
                ir_eps = 0.6
                print('IR Clustering criterion: eps: {:.3f}'.format(ir_eps))
                cluster_ir = DBSCAN(eps=ir_eps, min_samples=4, metric='precomputed', n_jobs=-1)
                rgb_eps = 0.6
                print('RGB Clustering criterion: eps: {:.3f}'.format(rgb_eps))
                cluster_rgb = DBSCAN(eps=rgb_eps, min_samples=4, metric='precomputed', n_jobs=-1)

            print('==> Create pseudo labels for unlabeled RGB data')

            cluster_loader_rgb = get_test_loader(dataset_rgb, args.height, args.width,
                                             256, args.workers, 
                                             testset=sorted(dataset_rgb.train))
            features_rgb, labels_rgb = extract_features(model, cluster_loader_rgb, print_freq=50,mode=1)
            del cluster_loader_rgb
            features_rgb = torch.cat([features_rgb[f].unsqueeze(0) for f, _, _ in sorted(dataset_rgb.train)], 0)
            labels_rgb = [labels_rgb[f].item() for f, _, _ in sorted(dataset_rgb.train)]
            
            print('==> Create pseudo labels for unlabeled IR data')
            cluster_loader_ir = get_test_loader(dataset_ir, args.height, args.width,
                                             256, args.workers, 
                                             testset=sorted(dataset_ir.train))
            features_ir, _ = extract_features(model, cluster_loader_ir, print_freq=50,mode=2)
            del cluster_loader_ir
            features_ir = torch.cat([features_ir[f].unsqueeze(0) for f, _, _ in sorted(dataset_ir.train)], 0)


            rerank_dist_ir = compute_jaccard_distance(features_ir, k1=args.k1, k2=args.k2,search_option=3)#rerank_dist_all_jacard[features_rgb.size(0):,features_rgb.size(0):]#
            pseudo_labels_ir = cluster_ir.fit_predict(rerank_dist_ir)
            rerank_dist_rgb = compute_jaccard_distance(features_rgb, k1=args.k1, k2=args.k2,search_option=3)#rerank_dist_all_jacard[:features_rgb.size(0),:features_rgb.size(0)]#
            pseudo_labels_rgb = cluster_rgb.fit_predict(rerank_dist_rgb)
            del rerank_dist_rgb
            del rerank_dist_ir
            num_cluster_ir = len(set(pseudo_labels_ir)) - (1 if -1 in pseudo_labels_ir else 0)
            num_cluster_rgb = len(set(pseudo_labels_rgb)) - (1 if -1 in pseudo_labels_rgb else 0)

        

        # generate new dataset and calculate cluster centers
        @torch.no_grad()
        def generate_cluster_features(labels, features):
            centers = collections.defaultdict(list)
            for i, label in enumerate(labels):
                if label == -1:
                    continue
                centers[labels[i]].append(features[i])

            centers = [
                torch.stack(centers[idx], dim=0).mean(0) for idx in sorted(centers.keys())
            ]

            centers = torch.stack(centers, dim=0)
            return centers

        cluster_features_ir = generate_cluster_features(pseudo_labels_ir, features_ir)
        cluster_features_rgb = generate_cluster_features(pseudo_labels_rgb, features_rgb)
        memory_ir = ClusterMemory(model.module.num_features, num_cluster_ir, temp=args.temp,
                               momentum=args.momentum, use_hard=args.use_hard).cuda()
        memory_rgb = ClusterMemory(model.module.num_features, num_cluster_rgb, temp=args.temp,
                               momentum=args.momentum, use_hard=args.use_hard).cuda()
        memory_ir.features = F.normalize(cluster_features_ir, dim=1).cuda()
        memory_rgb.features = F.normalize(cluster_features_rgb, dim=1).cuda()
        del cluster_features_rgb, cluster_features_ir

        trainer.memory_ir = memory_ir
        trainer.memory_rgb = memory_rgb

        
        

        

        pseudo_labeled_dataset_ir = []
    
        ir_label=[]
        pseudo_real_ir = {}
        for i, ((fname, _, cid), label) in enumerate(zip(sorted(dataset_ir.train), pseudo_labels_ir)):
            if label != -1:
                pseudo_labeled_dataset_ir.append((fname, label.item(), cid))
                ir_label.append(label.item())
                
        print('==> Statistics for IR epoch {}: {} clusters'.format(epoch, num_cluster_ir))
        pseudo_labeled_dataset_rgb = []
        
        rgb_label=[]
        pseudo_real_rgb = {}
        for i, ((fname, _, cid), label) in enumerate(zip(sorted(dataset_rgb.train), pseudo_labels_rgb)):
            if label != -1:
                pseudo_labeled_dataset_rgb.append((fname, label.item(), cid))
                rgb_label.append(label.item())
            
        print('==> Statistics for RGB epoch {}: {} clusters'.format(epoch, num_cluster_rgb))



        train_loader_ir = get_train_loader_ir(args, dataset_ir, args.height, args.width,
                                        args.batch_size, args.workers, args.num_instances, iters,
                                        trainset=pseudo_labeled_dataset_ir, no_cam=args.no_cam,train_transformer=transform_thermal)

        train_loader_rgb = get_train_loader_color(args, dataset_rgb, args.height, args.width,
                                        (args.batch_size//2), args.workers, args.num_instances, iters,
                                        trainset=pseudo_labeled_dataset_rgb, no_cam=args.no_cam,train_transformer=train_transformer_rgb,train_transformer1=train_transformer_rgb1)

        train_loader_ir.new_epoch()
        train_loader_rgb.new_epoch()

        trainer.train_s1(epoch, train_loader_ir,train_loader_rgb, optimizer,
                      print_freq=args.print_freq, train_iters=len(train_loader_ir))

        if epoch>=6 and ( (epoch + 1) % args.eval_step == 0 or (epoch == args.epochs - 1)):
##############################
            # args.test_batch=64
            args.img_w=args.width
            args.img_h=args.height
            normalize = T.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            transform_test = T.Compose([
                T.ToPILImage(),
                T.Resize((args.img_h,args.img_w)),
                T.ToTensor(),
                normalize,
            ])
            mode='all'
            data_path='data/USL-VI-ReID/SYSU-MM01'
            query_img, query_label, query_cam = process_query_sysu(data_path, mode=mode)
            nquery = len(query_label)
            queryset = TestData(query_img, query_label, transform=transform_test, img_size=(args.img_w, args.img_h))
            query_loader = data.DataLoader(queryset, batch_size=args.test_batch, shuffle=False, num_workers=4)
            query_feat_fc = extract_query_feat(model,query_loader,nquery)
            for trial in range(10):
                gall_img, gall_label, gall_cam = process_gallery_sysu(data_path, mode=mode, trial=trial)
                ngall = len(gall_label)
                trial_gallset = TestData(gall_img, gall_label, transform=transform_test, img_size=(args.img_w, args.img_h))
                trial_gall_loader = data.DataLoader(trial_gallset, batch_size=args.test_batch, shuffle=False, num_workers=4)

                gall_feat_fc = extract_gall_feat(model,trial_gall_loader,ngall)

                # fc feature
                distmat = np.matmul(query_feat_fc, np.transpose(gall_feat_fc))
                cmc, mAP, mINP = eval_sysu(-distmat, query_label, gall_label, query_cam, gall_cam)

                if trial == 0:
                    all_cmc = cmc
                    all_mAP = mAP
                    all_mINP = mINP

                else:
                    all_cmc = all_cmc + cmc
                    all_mAP = all_mAP + mAP
                    all_mINP = all_mINP + mINP

                print('Test Trial: {}'.format(trial))
                print(
                    'FC:   Rank-1: {:.2%} | Rank-5: {:.2%} | Rank-10: {:.2%}| Rank-20: {:.2%}| mAP: {:.2%}| mINP: {:.2%}'.format(
                        cmc[0], cmc[4], cmc[9], cmc[19], mAP, mINP))

            cmc = all_cmc / 10
            mAP = all_mAP / 10
            mINP = all_mINP / 10
            print('All Average:')
            print('FC:     Rank-1: {:.2%} | Rank-5: {:.2%} | Rank-10: {:.2%}| Rank-20: {:.2%}| mAP: {:.2%}| mINP: {:.2%}'.format(
                    cmc[0], cmc[4], cmc[9], cmc[19], mAP, mINP))
#################################
            is_best = (mAP > best_mAP)
            best_mAP = max(mAP, best_mAP)
            save_checkpoint({
                'state_dict': model.state_dict(),
                'epoch': epoch + 1,
                'best_mAP': best_mAP,
            }, is_best, fpath=osp.join(stage1_logs_dir, 'checkpoint.pth.tar'))

            print('\n * Finished epoch {:3d}  model mAP: {:5.1%}  best: {:5.1%}{}\n'.
                  format(epoch, mAP, best_mAP, ' *' if is_best else ''))
############################
        lr_scheduler.step()

    print('==> Test with the best model all search:')
    checkpoint = load_checkpoint(osp.join(stage1_logs_dir, 'model_best.pth.tar'))
    model.load_state_dict(checkpoint['state_dict'])

    mode='all'
    data_path='data/USL-VI-ReID/SYSU-MM01'
    query_img, query_label, query_cam = process_query_sysu(data_path, mode=mode)
    nquery = len(query_label)
    queryset = TestData(query_img, query_label, transform=transform_test, img_size=(args.img_w, args.img_h))
    query_loader = data.DataLoader(queryset, batch_size=args.test_batch, shuffle=False, num_workers=4)
    query_feat_fc = extract_query_feat(model,query_loader,nquery)
    for trial in range(10):
        gall_img, gall_label, gall_cam = process_gallery_sysu(data_path, mode=mode, trial=trial)
        ngall = len(gall_label)
        trial_gallset = TestData(gall_img, gall_label, transform=transform_test, img_size=(args.img_w, args.img_h))
        trial_gall_loader = data.DataLoader(trial_gallset, batch_size=args.test_batch, shuffle=False, num_workers=4)

        gall_feat_fc = extract_gall_feat(model,trial_gall_loader,ngall)
        # fc feature
        distmat = np.matmul(query_feat_fc, np.transpose(gall_feat_fc))

        cmc, mAP, mINP = eval_sysu(-distmat, query_label, gall_label, query_cam, gall_cam)
        if trial == 0:
            all_cmc = cmc
            all_mAP = mAP
            all_mINP = mINP

        else:
            all_cmc = all_cmc + cmc
            all_mAP = all_mAP + mAP
            all_mINP = all_mINP + mINP


        print('Test Trial: {}'.format(trial))
        print(
            'FC:   Rank-1: {:.2%} | Rank-5: {:.2%} | Rank-10: {:.2%}| Rank-20: {:.2%}| mAP: {:.2%}| mINP: {:.2%}'.format(
                cmc[0], cmc[4], cmc[9], cmc[19], mAP, mINP))
    cmc = all_cmc / 10
    mAP = all_mAP / 10
    mINP = all_mINP / 10
    print('All Average:')
    print('FC:     Rank-1: {:.2%} | Rank-5: {:.2%} | Rank-10: {:.2%}| Rank-20: {:.2%}| mAP: {:.2%}| mINP: {:.2%}'.format(
            cmc[0], cmc[4], cmc[9], cmc[19], mAP, mINP))
    end_time = time.monotonic()
    print('Total running time: ', timedelta(seconds=end_time - start_time))

    print('==> Test with the best model indoor search:')
    checkpoint = load_checkpoint(osp.join(stage1_logs_dir, 'model_best.pth.tar'))
    model.load_state_dict(checkpoint['state_dict'])
    mode='indoor'
    data_path='data/USL-VI-ReID/SYSU-MM01'
    query_img, query_label, query_cam = process_query_sysu(data_path, mode=mode)
    nquery = len(query_label)
    queryset = TestData(query_img, query_label, transform=transform_test, img_size=(args.img_w, args.img_h))
    query_loader = data.DataLoader(queryset, batch_size=args.test_batch, shuffle=False, num_workers=4)
    query_feat_fc = extract_query_feat(model,query_loader,nquery)
    for trial in range(10):
        gall_img, gall_label, gall_cam = process_gallery_sysu(data_path, mode=mode, trial=trial)
        ngall = len(gall_label)
        trial_gallset = TestData(gall_img, gall_label, transform=transform_test, img_size=(args.img_w, args.img_h))
        trial_gall_loader = data.DataLoader(trial_gallset, batch_size=args.test_batch, shuffle=False, num_workers=4)

        gall_feat_fc = extract_gall_feat(model,trial_gall_loader,ngall)
        # fc feature
        distmat = np.matmul(query_feat_fc, np.transpose(gall_feat_fc))

        cmc, mAP, mINP = eval_sysu(-distmat, query_label, gall_label, query_cam, gall_cam)
        if trial == 0:
            all_cmc = cmc
            all_mAP = mAP
            all_mINP = mINP

        else:
            all_cmc = all_cmc + cmc
            all_mAP = all_mAP + mAP
            all_mINP = all_mINP + mINP


        print('Test Trial: {}'.format(trial))
        print(
            'FC:   Rank-1: {:.2%} | Rank-5: {:.2%} | Rank-10: {:.2%}| Rank-20: {:.2%}| mAP: {:.2%}| mINP: {:.2%}'.format(
                cmc[0], cmc[4], cmc[9], cmc[19], mAP, mINP))
    cmc = all_cmc / 10
    mAP = all_mAP / 10
    mINP = all_mINP / 10
    print('All Average:')
    print('FC:     Rank-1: {:.2%} | Rank-5: {:.2%} | Rank-10: {:.2%}| Rank-20: {:.2%}| mAP: {:.2%}| mINP: {:.2%}'.format(
            cmc[0], cmc[4], cmc[9], cmc[19], mAP, mINP))
    end_time = time.monotonic()
    print('Total running time: ', timedelta(seconds=end_time - start_time))



def cal_match_acc(r2i, i2r, gts_rgb, gts_ir):
    
    acc_num = 0
    total_num = 0

    for k in r2i.keys():
        if gts_rgb[k].item() == gts_ir[r2i[k]].item():
            acc_num += 1
        total_num += 1

    
    for k in i2r.keys():
        if gts_ir[k].item() == gts_rgb[i2r[k]].item():
            acc_num += 1
        total_num += 1
    
    return float(acc_num) / float(total_num)


def main_worker_stage2(args,log_s1_name,log_s2_name):
    start_epoch=0
    best_mAP=0
    #checkpoint = load_checkpoint(osp.join(args.logs_dir + '/' + log_s1_name, 'model_best.pth.tar'))
    checkpoint = load_checkpoint(osp.join(args.logs_dir + '/' + log_s1_name, 'model_best.pth.tar'))
    stage2_logs_dir = osp.join(args.logs_dir+'/'+log_s2_name)
    start_time = time.monotonic()

    acc_list = []
    radio_list = []

    # cudnn.benchmark = True

    sys.stdout = Logger(osp.join(stage2_logs_dir, 'log.txt'))
    print("==========\nArgs:{}\n==========".format(args))

    # Create datasets
    iters = args.iters if (args.iters > 0) else None
    print("==> Load unlabeled dataset")
    dataset_ir = get_data('sysu_ir', args.data_dir)
    dataset_rgb = get_data('sysu_rgb', args.data_dir)

    # Create model
    model = create_model(args)
    model.load_state_dict(checkpoint['state_dict'])


    # Optimizer
    params = [{"params": [value]} for _, value in model.named_parameters() if value.requires_grad]
    optimizer = torch.optim.Adam(params, lr=args.lr, weight_decay=args.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=0.1)

    # Trainer
    trainer = ClusterContrastTrainer(model)
    for epoch in range(args.epochs):
        @torch.no_grad()
        def generate_cluster_features(labels, features):
            centers = collections.defaultdict(list)
        
            for i, label in enumerate(labels):
                if label == -1:
                    continue
                centers[labels[i]].append(features[i])

            cluster_nums = [len(centers[idx]) for idx in sorted(centers.keys())]

            centers = [
                torch.stack(centers[idx], dim=0).mean(0) for idx in sorted(centers.keys())
            ]

            centers = torch.stack(centers, dim=0)
            return centers, cluster_nums
        
        with torch.no_grad():
            if epoch == 0:
                # DBSCAN cluster
                ir_eps = 0.6
                print('IR Clustering criterion: eps: {:.3f}'.format(ir_eps))
                cluster_ir = DBSCAN(eps=ir_eps, min_samples=4, metric='precomputed', n_jobs=-1)
                rgb_eps = 0.6
                print('RGB Clustering criterion: eps: {:.3f}'.format(rgb_eps))
                cluster_rgb = DBSCAN(eps=rgb_eps, min_samples=4, metric='precomputed', n_jobs=-1)

            print('==> Create pseudo labels for unlabeled RGB data')

            cluster_loader_rgb = get_test_loader(dataset_rgb, args.height, args.width,
                                             256, args.workers, 
                                             testset=sorted(dataset_rgb.train))
            features_rgb, _ = extract_features(model, cluster_loader_rgb, print_freq=50,mode=1)
            del cluster_loader_rgb,
            features_rgb = torch.cat([features_rgb[f].unsqueeze(0) for f, _, _ in sorted(dataset_rgb.train)], 0)
            
            print('==> Create pseudo labels for unlabeled IR data')
            cluster_loader_ir = get_test_loader(dataset_ir, args.height, args.width,
                                             256, args.workers, 
                                             testset=sorted(dataset_ir.train))
            features_ir, _ = extract_features(model, cluster_loader_ir, print_freq=50,mode=2)
            del cluster_loader_ir
            features_ir = torch.cat([features_ir[f].unsqueeze(0) for f, _, _ in sorted(dataset_ir.train)], 0)


            rerank_dist_ir = compute_jaccard_distance(features_ir, k1=args.k1, k2=args.k2,search_option=3)#rerank_dist_all_jacard[features_rgb.size(0):,features_rgb.size(0):]#
            pseudo_labels_ir = cluster_ir.fit_predict(rerank_dist_ir)


            num_cluster_ir = len(set(pseudo_labels_ir)) - (1 if -1 in pseudo_labels_ir else 0)
            print('num_cluster_ir', num_cluster_ir)


            #cluster_features_ir = generate_cluster_features(pseudo_labels_ir, features_ir)
            cluster_features_ir, cluster_nums_ir = generate_cluster_features(pseudo_labels_ir, features_ir)
            
            rerank_dist_rgb = compute_jaccard_distance(features_rgb, k1=args.k1, k2=args.k2,search_option=3)#rerank_dist_all_jacard[:features_rgb.size(0),:features_rgb.size(0)]#
            pseudo_labels_rgb = cluster_rgb.fit_predict(rerank_dist_rgb)

            num_cluster_rgb = len(set(pseudo_labels_rgb)) - (1 if -1 in pseudo_labels_rgb else 0)
            print('num_cluster_rgb', num_cluster_rgb)

            

            

            del rerank_dist_rgb
            del rerank_dist_ir

    
        
        #cluster_features_rgb = generate_cluster_features(pseudo_labels_rgb, features_rgb)
        cluster_features_rgb, cluster_nums_rgb = generate_cluster_features(pseudo_labels_rgb, features_rgb)
        # Create hybrid memory
        memory_ir = ClusterMemory(model.module.num_features, num_cluster_ir, temp=args.temp,
                               momentum=args.momentum, use_hard=args.use_hard).cuda()
        memory_rgb = ClusterMemory(model.module.num_features, num_cluster_rgb, temp=args.temp,
                               momentum=args.momentum, use_hard=args.use_hard).cuda()
        memory_ir.features = F.normalize(cluster_features_ir, dim=1).cuda()
        memory_rgb.features = F.normalize(cluster_features_rgb, dim=1).cuda()


        trainer.memory_ir = memory_ir
        trainer.memory_rgb = memory_rgb
        trainer.lambda1 = args.lambda1
        trainer.lambda2 = args.lambda2
        trainer.balance = args.balance


        
        cluster_features_rgb = F.normalize(cluster_features_rgb, dim=1)
        cluster_features_ir = F.normalize(cluster_features_ir, dim=1)




        print("Progressive Graph Matching")
        i2r = {}
        r2i = {}
        R = []
        bgm = False
        
        assert num_cluster_rgb >= num_cluster_ir
        if num_cluster_rgb >= num_cluster_ir:
            print(1)
            # clusternorm
            cluster_features_rgb = F.normalize(cluster_features_rgb, dim=1)
            cluster_features_ir = F.normalize(cluster_features_ir, dim=1)
            # [-1, 1] torch.mm(cluster_features_rgb, cluster_features_ir.T) #CostMatrix
            similarity = ((torch.mm(cluster_features_rgb, cluster_features_ir.T))/1).exp().cpu() #.exp().cpu()
            dis_similarity = (1 / (similarity))
            cost = dis_similarity / 1
            tmp = torch.zeros(dis_similarity.shape[0], dis_similarity.shape[0] - dis_similarity.shape[1])
            cost = (torch.cat((cost, tmp), 1))
            unmatched_row = []
            row_ind, col_ind = linear_sum_assignment(cost)

            
            for idx, item in enumerate(row_ind):
                if col_ind[idx] < similarity.shape[1]:
                    R.append((row_ind[idx], col_ind[idx]))
                    r2i[row_ind[idx]] = col_ind[idx]
                    i2r[col_ind[idx]] = row_ind[idx]
                else:
                    unmatched_row.append(row_ind[idx])

            
            assert len(unmatched_row) > 0
            if len(unmatched_row) > num_cluster_ir:
                print(3)
                unmatched_row_new = []
                
                unmatched_cost = cost[unmatched_row][:,:dis_similarity.shape[1]]
                

                tmp = torch.zeros(unmatched_cost.shape[0], unmatched_cost.shape[0] - unmatched_cost.shape[1])
                unmatched_cost = (torch.cat((unmatched_cost, tmp), 1))
                unmatched_row_ind, unmatched_col_ind = linear_sum_assignment(unmatched_cost)

                
                for idx, item in enumerate(unmatched_row_ind):
                    if unmatched_col_ind[idx] < similarity.shape[1]:
                        R.append((unmatched_row[idx], unmatched_col_ind[idx]))
                        r2i[unmatched_row[idx]] = unmatched_col_ind[idx]
                    else:
                        unmatched_row_new.append(unmatched_row[idx])
                
                unmatched_row = copy.deepcopy(unmatched_row_new)


            
            assert len(unmatched_row) > 0
            print(2)
                
            unmatched_cost = cost[unmatched_row][:,:dis_similarity.shape[1]]
            unmatched_row_ind, unmatched_col_ind = linear_sum_assignment(unmatched_cost)
            for idx, item in enumerate(unmatched_row_ind):
                if unmatched_col_ind[idx] < similarity.shape[1]:
                    R.append((unmatched_row[idx], unmatched_col_ind[idx]))
                    r2i[unmatched_row[idx]] = unmatched_col_ind[idx]
                
        

            del cluster_features_ir, cluster_features_rgb

        

        



        correlations_rgb_all, weights_rgb_all = generate_correlations(features_rgb, pseudo_labels_rgb, cluster_nums_rgb, args.neighbour, args.coe)
        correlations_ir_all, weights_ir_all = generate_correlations(features_ir, pseudo_labels_ir, cluster_nums_ir, args.neighbour, args.coe)

        correlations_rgb_all_cross, weights_rgb_all_cross = generate_correlations_cross(features_rgb, features_ir, pseudo_labels_ir, cluster_nums_ir, args.neighbour, pseudo_labels_rgb, r2i, args.coe)
        correlations_ir_all_cross, weights_ir_all_cross = generate_correlations_cross(features_ir, features_rgb, pseudo_labels_rgb, cluster_nums_rgb, args.neighbour, pseudo_labels_ir, i2r, args.coe)

        assert correlations_ir_all.shape[0] == pseudo_labels_ir.shape[0]
        assert correlations_rgb_all.shape[0] == pseudo_labels_rgb.shape[0]

        assert len(dataset_ir.train) == correlations_ir_all.shape[0]
        assert len(dataset_rgb.train) == correlations_rgb_all.shape[0]


        pseudo_labeled_dataset_ir = []
        correlations_ir = []
        correlations_ir_cross = []
        weights_ir = []
        weights_ir_cross = []
        ir_label=[]
        for i, ((fname, _, cid), label, correlation, correlation_cross, weight, weight_cross) in enumerate(zip(sorted(dataset_ir.train), pseudo_labels_ir, correlations_ir_all, correlations_ir_all_cross, weights_ir_all, weights_ir_all_cross)):
            if label != -1:
                pseudo_labeled_dataset_ir.append((fname, label.item(), cid))
                ir_label.append(label.item())
                correlations_ir.append(correlation)
                correlations_ir_cross.append(correlation_cross)
                weights_ir.append(weight)
                weights_ir_cross.append(weight_cross)
        print('==> Statistics for IR epoch {}: {} clusters'.format(epoch, num_cluster_ir))

        
        pseudo_labeled_dataset_rgb = []
        correlations_rgb = []
        correlations_rgb_cross = []
        weights_rgb = []
        weights_rgb_cross = []
        rgb_label=[]
        for i, ((fname, _, cid), label, correlation, correlation_cross, weight, weight_cross) in enumerate(zip(sorted(dataset_rgb.train), pseudo_labels_rgb, correlations_rgb_all, correlations_rgb_all_cross, weights_rgb_all, weights_rgb_all_cross)):
            if label != -1:
                pseudo_labeled_dataset_rgb.append((fname, label.item(), cid))
                rgb_label.append(label.item())
                correlations_rgb.append(correlation)
                correlations_rgb_cross.append(correlation_cross)
                weights_rgb.append(weight)
                weights_rgb_cross.append(weight_cross)
        print('==> Statistics for RGB epoch {}: {} clusters'.format(epoch, num_cluster_rgb))
        
        ######################## PGM

        assert len(pseudo_labeled_dataset_ir) == len(correlations_ir)
        assert len(pseudo_labeled_dataset_rgb) == len(correlations_rgb)


        trainer.correlations_rgb = correlations_rgb
        trainer.correlations_ir = correlations_ir
        trainer.correlations_ir_cross = correlations_ir_cross
        trainer.correlations_rgb_cross = correlations_rgb_cross
        trainer.weights_ir = weights_ir
        trainer.weights_ir_cross = weights_ir_cross
        trainer.weights_rgb = weights_rgb
        trainer.weights_rgb_cross = weights_rgb_cross

        

        ########################
        normalizer = T.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
        height=args.height
        width=args.width
        train_transformer_rgb = T.Compose([
            T.Resize((height, width), interpolation=3),
            T.Pad(10),
            T.RandomCrop((height, width)),
            T.RandomHorizontalFlip(p=0.5),
            # T.RandomGrayscale(p=0.1),
            T.ToTensor(),
            normalizer,
            ChannelRandomErasing(probability = 0.5)
        ])
        
        train_transformer_rgb1 = T.Compose([
            T.Resize((height, width), interpolation=3),
            T.Pad(10),
            T.RandomCrop((height, width)),
            T.RandomHorizontalFlip(p=0.5),
            T.ColorJitter(brightness=0.5,contrast=0.5,saturation=0.5,hue=0.5),
            T.ToTensor(),
            normalizer,
            ChannelRandomErasing(probability = 0.5),
            ChannelExchange(gray = 2),
        ])

        transform_thermal = T.Compose( [
            T.Resize((height, width), interpolation=3),
            T.Pad(10),
            T.RandomCrop((288, 144)),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            normalizer,
            ChannelRandomErasing(probability = 0.5),
            ChannelAdapGray(probability =0.5)])

        train_loader_ir = get_train_loader_ir(args, dataset_ir, args.height, args.width,
                                        args.batch_size, args.workers, args.num_instances, iters,
                                        trainset=pseudo_labeled_dataset_ir, no_cam=args.no_cam,train_transformer=transform_thermal)

        train_loader_rgb = get_train_loader_color(args, dataset_rgb, args.height, args.width,
                                        (args.batch_size//2), args.workers, args.num_instances, iters,
                                        trainset=pseudo_labeled_dataset_rgb, no_cam=args.no_cam,train_transformer=train_transformer_rgb,train_transformer1=train_transformer_rgb1)

        train_loader_ir.new_epoch()
        train_loader_rgb.new_epoch()

        trainer.train(epoch, train_loader_ir,train_loader_rgb, optimizer,
                      print_freq=args.print_freq, train_iters=len(train_loader_ir), i2r=i2r, r2i=r2i)
        # del train_loader_ir, train_loader_rgb

        if epoch>=6 and ( (epoch + 1) % args.eval_step == 0 or (epoch == args.epochs - 1)):
            args.img_w=args.width
            args.img_h=args.height
            normalize = T.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            transform_test = T.Compose([
                T.ToPILImage(),
                T.Resize((args.img_h,args.img_w)),
                T.ToTensor(),
                normalize,
            ])
            mode='all'
            data_path='data/USL-VI-ReID/SYSU-MM01'
            query_img, query_label, query_cam = process_query_sysu(data_path, mode=mode)
            nquery = len(query_label)
            queryset = TestData(query_img, query_label, transform=transform_test, img_size=(args.img_w, args.img_h))
            query_loader = data.DataLoader(queryset, batch_size=args.test_batch, shuffle=False, num_workers=4)
            query_feat_fc = extract_query_feat(model,query_loader,nquery)
            for trial in range(10):
                gall_img, gall_label, gall_cam = process_gallery_sysu(data_path, mode=mode, trial=trial)
                ngall = len(gall_label)
                trial_gallset = TestData(gall_img, gall_label, transform=transform_test, img_size=(args.img_w, args.img_h))
                trial_gall_loader = data.DataLoader(trial_gallset, batch_size=args.test_batch, shuffle=False, num_workers=4)

                gall_feat_fc = extract_gall_feat(model,trial_gall_loader,ngall)

                # fc feature
                distmat = np.matmul(query_feat_fc, np.transpose(gall_feat_fc))
                cmc, mAP, mINP = eval_sysu(-distmat, query_label, gall_label, query_cam, gall_cam)

                if trial == 0:
                    all_cmc = cmc
                    all_mAP = mAP
                    all_mINP = mINP

                else:
                    all_cmc = all_cmc + cmc
                    all_mAP = all_mAP + mAP
                    all_mINP = all_mINP + mINP

                print('Test Trial: {}'.format(trial))
                print(
                    'FC:   Rank-1: {:.2%} | Rank-5: {:.2%} | Rank-10: {:.2%}| Rank-20: {:.2%}| mAP: {:.2%}| mINP: {:.2%}'.format(
                        cmc[0], cmc[4], cmc[9], cmc[19], mAP, mINP))

            cmc = all_cmc / 10
            mAP = all_mAP / 10
            mINP = all_mINP / 10
            print('All Average:')
            print('FC:     Rank-1: {:.2%} | Rank-5: {:.2%} | Rank-10: {:.2%}| Rank-20: {:.2%}| mAP: {:.2%}| mINP: {:.2%}'.format(
                    cmc[0], cmc[4], cmc[9], cmc[19], mAP, mINP))

            is_best = (mAP > best_mAP)
            best_mAP = max(mAP, best_mAP)
            save_checkpoint({
                'state_dict': model.state_dict(),
                'epoch': epoch + 1,
                'best_mAP': best_mAP,
            }, is_best, fpath=osp.join(stage2_logs_dir, 'checkpoint.pth.tar'))

            print('\n * Finished epoch {:3d}  model mAP: {:5.1%}  best: {:5.1%}{}\n'.
                  format(epoch, mAP, best_mAP, ' *' if is_best else ''))
############################
        lr_scheduler.step()

    print('==> Test with the best model:')
    checkpoint = load_checkpoint(osp.join(stage2_logs_dir, 'model_best.pth.tar'))
    model.load_state_dict(checkpoint['state_dict'])
    mode='all'
    print(mode)
    data_path='data/USL-VI-ReID/SYSU-MM01'
    query_img, query_label, query_cam = process_query_sysu(data_path, mode=mode)
    nquery = len(query_label)
    queryset = TestData(query_img, query_label, transform=transform_test, img_size=(args.img_w, args.img_h))
    query_loader = data.DataLoader(queryset, batch_size=args.test_batch, shuffle=False, num_workers=4)
    query_feat_fc = extract_query_feat(model,query_loader,nquery)
    for trial in range(10):
        gall_img, gall_label, gall_cam = process_gallery_sysu(data_path, mode=mode, trial=trial)
        ngall = len(gall_label)
        trial_gallset = TestData(gall_img, gall_label, transform=transform_test, img_size=(args.img_w, args.img_h))
        trial_gall_loader = data.DataLoader(trial_gallset, batch_size=args.test_batch, shuffle=False, num_workers=4)

        gall_feat_fc = extract_gall_feat(model,trial_gall_loader,ngall)
        # fc feature
        distmat = np.matmul(query_feat_fc, np.transpose(gall_feat_fc))

        cmc, mAP, mINP = eval_sysu(-distmat, query_label, gall_label, query_cam, gall_cam)
        if trial == 0:
            all_cmc = cmc
            all_mAP = mAP
            all_mINP = mINP

        else:
            all_cmc = all_cmc + cmc
            all_mAP = all_mAP + mAP
            all_mINP = all_mINP + mINP


        print('Test Trial: {}'.format(trial))
        print(
            'FC:   Rank-1: {:.2%} | Rank-5: {:.2%} | Rank-10: {:.2%}| Rank-20: {:.2%}| mAP: {:.2%}| mINP: {:.2%}'.format(
                cmc[0], cmc[4], cmc[9], cmc[19], mAP, mINP))
    cmc = all_cmc / 10
    mAP = all_mAP / 10
    mINP = all_mINP / 10
    print('All Average:')
    print('FC:     Rank-1: {:.2%} | Rank-5: {:.2%} | Rank-10: {:.2%}| Rank-20: {:.2%}| mAP: {:.2%}| mINP: {:.2%}'.format(
            cmc[0], cmc[4], cmc[9], cmc[19], mAP, mINP))
#################################

    mode='indoor'
    print(mode)
    data_path='data/USL-VI-ReID/SYSU-MM01'
    query_img, query_label, query_cam = process_query_sysu(data_path, mode=mode)
    nquery = len(query_label)
    queryset = TestData(query_img, query_label, transform=transform_test, img_size=(args.img_w, args.img_h))
    query_loader = data.DataLoader(queryset, batch_size=args.test_batch, shuffle=False, num_workers=4)
    query_feat_fc = extract_query_feat(model,query_loader,nquery)
    for trial in range(10):
        gall_img, gall_label, gall_cam = process_gallery_sysu(data_path, mode=mode, trial=trial)
        ngall = len(gall_label)
        trial_gallset = TestData(gall_img, gall_label, transform=transform_test, img_size=(args.img_w, args.img_h))
        trial_gall_loader = data.DataLoader(trial_gallset, batch_size=args.test_batch, shuffle=False, num_workers=4)

        gall_feat_fc = extract_gall_feat(model,trial_gall_loader,ngall)
        # fc feature
        distmat = np.matmul(query_feat_fc, np.transpose(gall_feat_fc))

        cmc, mAP, mINP = eval_sysu(-distmat, query_label, gall_label, query_cam, gall_cam)
        if trial == 0:
            all_cmc = cmc
            all_mAP = mAP
            all_mINP = mINP

        else:
            all_cmc = all_cmc + cmc
            all_mAP = all_mAP + mAP
            all_mINP = all_mINP + mINP


        print('Test Trial: {}'.format(trial))
        print(
            'FC:   Rank-1: {:.2%} | Rank-5: {:.2%} | Rank-10: {:.2%}| Rank-20: {:.2%}| mAP: {:.2%}| mINP: {:.2%}'.format(
                cmc[0], cmc[4], cmc[9], cmc[19], mAP, mINP))
    cmc = all_cmc / 10
    mAP = all_mAP / 10
    mINP = all_mINP / 10
    print('All Average:')
    print('FC:     Rank-1: {:.2%} | Rank-5: {:.2%} | Rank-10: {:.2%}| Rank-20: {:.2%}| mAP: {:.2%}| mINP: {:.2%}'.format(
            cmc[0], cmc[4], cmc[9], cmc[19], mAP, mINP))

    end_time = time.monotonic()
    print('Total running time: ', timedelta(seconds=end_time - start_time))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Augmented Dual-Contrastive Aggregation Learning for USL-VI-ReID")
    # data
    parser.add_argument('-d', '--dataset', type=str, default='sysu')
    parser.add_argument('-b', '--batch-size', type=int, default=2)
    parser.add_argument('--test-batch', type=int, default=64)
    parser.add_argument('-j', '--workers', type=int, default=8)
    parser.add_argument('--height', type=int, default=288, help="input height")
    parser.add_argument('--width', type=int, default=144, help="input width")
    parser.add_argument('--num-instances', type=int, default=4,
                        help="each minibatch consist of "
                             "(batch_size // num_instances) identities, and "
                             "each identity has num_instances instances, "
                             "default: 0 (NOT USE)")
    # cluster
    parser.add_argument('--eps', type=float, default=0.6,
                        help="max neighbor distance for DBSCAN")
    parser.add_argument('--eps-gap', type=float, default=0.02,
                        help="multi-scale criterion for measuring cluster reliability")
    parser.add_argument('--k1', type=int, default=30,
                        help="hyperparameter for jaccard distance")
    parser.add_argument('--k2', type=int, default=6,
                        help="hyperparameter for jaccard distance")

    # model
    parser.add_argument('-a', '--arch', type=str, default='resnet50',
                        choices=models.names())
    parser.add_argument('--features', type=int, default=0)
    parser.add_argument('--dropout', type=float, default=0)
    parser.add_argument('--momentum', type=float, default=0.2,
                        help="update momentum for the hybrid memory")
    # optimizer
    parser.add_argument('--lr', type=float, default=0.00035,
                        help="learning rate")
    parser.add_argument('--weight-decay', type=float, default=5e-4)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--iters', type=int, default=200)
    parser.add_argument('--step-size', type=int, default=20)
    # training configs
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--print-freq', type=int, default=10)
    parser.add_argument('--eval-step', type=int, default=1)
    parser.add_argument('--temp', type=float, default=0.05,
                        help="temperature for scaling contrastive loss")
    # path
    working_dir = osp.dirname(osp.abspath(__file__))
    parser.add_argument('--data-dir', type=str, metavar='PATH',
                        default=osp.join(working_dir, 'data'))
    parser.add_argument('--logs-dir', type=str, metavar='PATH',
                        default=osp.join(working_dir, 'logs'))
    parser.add_argument('--pooling-type', type=str, default='gem')
    parser.add_argument('--use-hard', action="store_true")
    parser.add_argument('--no-cam',  action="store_true")
    parser.add_argument('--neighbour', type=int, default=30)
    parser.add_argument('--lambda1', type=float, default=0.1)
    parser.add_argument('--lambda2', type=float, default=0.1)
    parser.add_argument('--balance', type=float, default=0.5)
    parser.add_argument('--coe', type=float, default=5.0)
    parser.add_argument('--stage', type=str, default='s1')

    main()
