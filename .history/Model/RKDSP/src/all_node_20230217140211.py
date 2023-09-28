import sys

print(sys.getdefaultencoding())
from datetime import datetime
import math
import argparse
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import auc

import networkx as nx
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.utils import shuffle
from utils import *
from model import *
from numpy import *
import torch.utils.data as Data


torch.backends.cudnn.benchmark = True
from sklearn.metrics import roc_auc_score, average_precision_score
from functools import reduce
from tqdm import tqdm
from torch.autograd import Variable

lnc_dis = np.loadtxt('lnc_dis.txt')


def parse_args():
    parser = argparse.ArgumentParser(description='ckd')

    parser.add_argument('--output', type=str, required=False, help='emb file', default='emb.dat')
    parser.add_argument('--ltype', type=str, default="1,4")

    parser.add_argument('--seed', type=int, default=5)
    parser.add_argument('--device', type=str, default='cuda')

    parser.add_argument('--size', type=int, default=50)  # feature size
    parser.add_argument('--dim', type=int, default=50)  # output emb size
    parser.add_argument('--layers', type=int, default=2)
    parser.add_argument('--dropout', type=float, default=0.0)
    parser.add_argument('--negative_cnt', type=int, default=5)
    parser.add_argument('--sample_times', type=int, default=1)
    parser.add_argument('--topk', type=int, default=20)

    parser.add_argument('--supervised', type=str, default="False")
    parser.add_argument('--neigh-por', type=float, default=0.6)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--batch-size', type=int, default=6)
    parser.add_argument('--epochs', type=int, default=150)
    parser.add_argument('--stop_cnt', type=int, default=5)
    parser.add_argument('--global-weight', type=float, default=0.05)

    parser.add_argument('--attributed', type=str, default="True")

    parser.add_argument('--dataset', type=str, help='', default='lnc')
    args = parser.parse_args()

    if args.dataset == 'ACM2':
        args.dim = 100
        args.ltype = '0,1'
        args.lr = 0.00002
        args.topk = 15
        args.epochs = 60
        args.stop_cnt = 100
        args.global_weight = 0.1
        args.batch_size = 4
    elif args.dataset == 'ACM':
        args.dim = 100
        args.ltype = '0,2,4,6'
        args.lr = 0.001
        args.topk = 20  # 25
        args.epochs = 20
        args.stop_cnt = 100
        args.global_weight = 0.05
        args.batch_size = 6
        args.seed = 7
    elif args.dataset == 'lnc':
        args.dim = 100
        args.ltype_lnc = '0,1,2'
        args.ltype_dis = '3,4,5'
        args.lr = 0.0005
        args.topk = 20  # 25
        args.epochs = 200
        args.stop_cnt = 100
        args.global_weight = 0.05
        args.batch_size = 6
        args.seed = 7
    elif args.dataset == 'DBLP2':
        args.dim = 300
        args.ltype = '1,3,4,5'
        # args.lr = 0.00001
        args.lr = 0.00002
        args.topk = 35
        args.epochs = 60
        args.stop_cnt = 100
        args.global_weight = 0.12
        args.batch_size = 2
        args.seed = 11
    elif args.dataset == 'DBLP':
        args.ltype = '0,1,2'
        args.lr = 0.00005
        args.topk = 30  # 25
        args.epochs = 200
        args.stop_cnt = 100
        args.global_weight = 0.1
    elif args.dataset == 'Freebase':
        args.dim = 200
        args.ltype = '0,1,2,3,4'
        args.lr = 0.00002
        args.topk = 35  # 25
        args.epochs = 150
        args.stop_cnt = 100
        args.global_weight = 0.15
        args.batch_size = 2
    elif args.dataset == 'PubMed':
        args.dim = 200
        args.ltype = '1,2,4'
        args.lr = 0.0002
        args.topk = 25
        args.epochs = 60
        args.stop_cnt = 100
        args.global_weight = 0.1
        args.batch_size = 2
    return args
    """
    if args.dataset=='PubMed':
        args.ltype='1,2,4'
        args.lr=0.0002
        args.topk=25 #25
        args.epochs=100
        args.stop_cnt=100
        args.global_weight=0.1
    elif args.dataset=='acm3':
        args.ltype = '0,1'
        args.lr = 0.00002
        args.topk = 20
        args.epochs = 40
        args.stop_cnt = 100
        args.global_weight = 0.05
    elif args.dataset=='DBLP2':
        args.ltype='1,3,4,5'
        args.lr=0.00005
        args.topk=25 #25
        args.epochs=150
        args.stop_cnt=100
        args.global_weight=0.1
    return args
    """

def Count_Value_1(matrix: ndarray, k: int):
    A = array(nonzero(matrix))
    A = A.T
    np.random.shuffle(A)
    B = array_split(A, k, 0)

    return B

def Count_Value_0(matrix: ndarray, k: int):
    A = array(np.where(matrix == 0))
    A = A.T
    np.random.shuffle(A)
    B = []
    for i in range(2685):
        B.append(A[i])
    C = np.array(B)
    D = array_split(C, k, 0)

    return D, A

def Make_Train_Test_Set(train_1: ndarray, train_0: ndarray, all_train_0: ndarray):
    matrix1 = []
    matrix0 = []

    for i in range(len(train_1) - 1):
        for j in range(train_1[i].shape[0]):
            matrix1.append(train_1[i][j])
    for m in range(len(train_0) - 1):
        for n in range(train_0[m].shape[0]):
            matrix1.append(train_0[m][n])

    for p in range(train_1[len(train_1) - 1].shape[0]):
        matrix0.append(train_1[len(train_1) - 1][p])
    for q in range(len(all_train_0)):
        matrix0.append(all_train_0[q])
    train_five_1 = train_1[4]

    matrix_train = np.array(matrix1)
    matrix_test = np.array(matrix0)

    return matrix_train, matrix_test, train_five_1

def fold_5(TPR, FPR, PR):
    fold = len(TPR)
    le = []
    for i in range(fold):
        le.append(len(TPR[i]))
    min_f = min(le)
    F_TPR = np.zeros((fold, min_f))
    F_FPR = np.zeros((fold, min_f))
    F_P = np.zeros((fold, min_f))
    for i in range(fold):
        k = len(TPR[i]) / min_f  # 这行数据的有效数据数目是最小数目的多少倍
        for j in range(min_f):
            F_TPR[i][j] = TPR[i][int(round(((j + 1) * k))) - 1]
            F_FPR[i][j] = FPR[i][int(round(((j + 1) * k))) - 1]
            F_P[i][j] = PR[i][int(round(((j + 1) * k))) - 1]
    TPR_5 = F_TPR.sum(0) / fold
    FPR_5 = F_FPR.sum(0) / fold
    PR_5 = F_P.sum(0) / fold
    return TPR_5, FPR_5, PR_5

def calculate_TPR_FPR(RD, f, B):
    old_id = np.argsort(-RD)
    min_f = int(min(f))
    max_f = int(max(f))

    TP_FN = np.zeros((RD.shape[0], 1), dtype=np.float64)
    FP_TN = np.zeros((RD.shape[0], 1), dtype=np.float64)
    TP = np.zeros((RD.shape[0], max_f), dtype=np.float64)
    TP2 = np.zeros((RD.shape[0], min_f), dtype=np.float64)
    FP = np.zeros((RD.shape[0], max_f), dtype=np.float64)
    FP2 = np.zeros((RD.shape[0], min_f), dtype=np.float64)
    P = np.zeros((RD.shape[0], max_f), dtype=np.float64)
    P2 = np.zeros((RD.shape[0], min_f), dtype=np.float64)

    for i in range(RD.shape[0]):
        TP_FN[i] = sum(B[i] == 1)
        FP_TN[i] = sum(B[i] == 0)

    for i in range(RD.shape[0]):
        for j in range(int(f[i])):
            if j == 0:
                if B[i][old_id[i][j]] == 1:
                    FP[i][j] = 0
                    TP[i][j] = 1
                    P[i][j] = TP[i][j] / (j + 1)
                else:
                    TP[i][j] = 0
                    FP[i][j] = 1
                    P[i][j] = TP[i][j] / (j + 1)
            else:
                if B[i][old_id[i][j]] == 1:
                    FP[i][j] = FP[i][j - 1]
                    TP[i][j] = TP[i][j - 1] + 1
                    P[i][j] = TP[i][j] / (j + 1)
                else:
                    TP[i][j] = TP[i][j - 1]
                    FP[i][j] = FP[i][j - 1] + 1
                    P[i][j] = TP[i][j] / (j + 1)

    ki = 0
    for i in range(RD.shape[0]):
        if TP_FN[i] == 0:
            TP[i] = 0
            FP[i] = 0
            ki = ki + 1
        else:
            TP[i] = TP[i] / TP_FN[i]
            FP[i] = FP[i] / FP_TN[i]

    for i in range(RD.shape[0]):
        kk = f[i] / min_f
        for j in range(min_f):
            TP2[i][j] = TP[i][int(np.round_(((j + 1) * kk))) - 1]
            FP2[i][j] = FP[i][int(np.round_(((j + 1) * kk))) - 1]
            P2[i][j] = P[i][int(np.round_(((j + 1) * kk))) - 1]
    TPR = TP2.sum(0) / (TP.shape[0] - ki)
    FPR = FP2.sum(0) / (FP.shape[0] - ki)
    P = P2.sum(0) / (P.shape[0] - ki)
    return TPR, FPR, P


def curve(FPR, TPR, P):
    plt.figure()

    plt.subplot(121)
    plt.xlim(0.0, 1.0)
    plt.ylim(0.0, 1.0)

    plt.title("ROC curve  (AUC = %.4f)" % (auc(FPR, TPR)))
    plt.xlabel('FPR')
    plt.ylabel('TPR')

    plt.plot(FPR, TPR)
    plt.subplot(122)
    plt.xlim(0.0, 1.0)
    plt.ylim(0.0, 1.0)

    plt.title("PR curve  (AUPR = %.4f)" % (auc(TPR, P) + TPR[0] * P[0]))

    plt.xlabel('R')
    plt.ylabel('P')

    plt.plot(TPR, P)
    plt.show()

def load_data(id, BATCH_SIZE):
    x = []
    y = []
    for j in range(id.shape[0]):
        temp_save = []
        x_A = int(id[j][0])  # 取横坐标
        y_A = int(id[j][1])  # 取纵坐标
        temp_save.append([x_A, y_A])  # 存坐标值 [1,2]
        # print(np.array(temp_save).shape)
        label = lnc_dis[[x_A], [y_A]]  # 取坐标对应的标签
        x.append([temp_save])
        y.append(label)
    x = torch.FloatTensor(np.array(x))
    print(x.size())  # [12648,1,1,2]
    y = torch.LongTensor(np.array(y))
    print(y.size())
    torch_dataset = Data.TensorDataset(x, y)  # 用来对坐标进行打包,如[1,1,2,2],[2,1,2,5]
    data2_loader = Data.DataLoader(
        dataset=torch_dataset,
        batch_size=BATCH_SIZE,  # 批大小
        shuffle=False,
        num_workers=0,
        drop_last=False
    )  # 用来包装所使用的成批数据
    return data2_loader

def score(criterion, emb_list, graph_emb_list, status_list):
    index = torch.Tensor([0]).long().cuda()
    loss = None
    for idx in range(len(emb_list)):  # 0, 1, 2, 3
        emb_list[idx] = emb_list[idx].index_select(dim=1, index=index).squeeze()  # 6, 21, 100变成6, 100
    for idx in range(len(emb_list)):  # 0, 1, 2, 3
        for idy in range(len(emb_list)):  # 0, 1, 2, 3
            node_emb = emb_list[idx]
            graph_emb = graph_emb_list[idy]
            mask = torch.Tensor([i[idy] for i in status_list]).bool().cuda()
            pos = torch.sum(node_emb * graph_emb, dim=1).squeeze().masked_select(mask)
            matrix = torch.mm(node_emb, graph_emb.T)  # 6, 6
            mask_idx = torch.Tensor([i for i in range(len(status_list)) if
                                     status_list[i][idy] == 0]).long().cuda()  # 找到六个节点中的在第一个元路径中没有邻居的节点索引
            neg_mask = np.ones(shape=(node_emb.shape[0], node_emb.shape[0]))  # 生成全是1的6, 6矩阵
            row, col = np.diag_indices_from(neg_mask)  # 获取主对角线元素的索引
            neg_mask[row, col] = 0
            neg_mask = torch.from_numpy(neg_mask).bool().cuda()
            neg_mask[mask_idx,] = 0
            neg = matrix.masked_select(neg_mask)

            if pos.shape[0] == 0:
                continue
            if loss is None:
                loss = criterion(pos, torch.from_numpy(np.array([1] * pos.shape[0])).float().cuda())
                loss += criterion(neg, torch.from_numpy(np.array([0] * neg.shape[0])).float().cuda())
            else:
                loss += criterion(pos, torch.from_numpy(np.array([1] * pos.shape[0])).float().cuda())
                loss += criterion(neg, torch.from_numpy(np.array([0] * neg.shape[0])).float().cuda())
    return loss


def global_score(criterion, emb_list, graph_emb_list, neg_graph_emb_list, status_list):
    loss = None

    for idx in range(len(emb_list)):
        for idy in range(len(emb_list)):
            node_emb = emb_list[idx]
            global_emb = graph_emb_list[idy]
            neg_global_emb = neg_graph_emb_list[idy]
            mask = torch.Tensor([i[idx] for i in status_list]).bool().cuda()
            pos = torch.sum(node_emb * global_emb, dim=1).squeeze().masked_select(mask)
            neg = torch.sum(node_emb * neg_global_emb, dim=1).squeeze().masked_select(mask)
            if pos.shape[0] == 0:
                continue

            if loss is None:
                loss = criterion(pos, torch.from_numpy(np.array([1] * pos.shape[0])).float().cuda())
                loss += criterion(neg, torch.from_numpy(np.array([0] * neg.shape[0])).float().cuda())
            else:
                loss += criterion(pos, torch.from_numpy(np.array([1] * pos.shape[0])).float().cuda())
                loss += criterion(neg, torch.from_numpy(np.array([0] * neg.shape[0])).float().cuda())
    return loss


def main():
    print(f'start time:{datetime.now()}')
    cuda_device = 0
    torch.cuda.set_device(cuda_device)
    print('cuda:', cuda_device)

    args = parse_args()
    print(f'emb size:{args.dim}')

    set_seed(args.seed, args.device)
    print(f'seed:{args.seed}')

    print(
        f'dataset:{args.dataset},attributed:{args.attributed},ltypes:{args.ltype},topk:{args.topk},lr:{args.lr},batch-size:{args.batch_size},stop_cnt:{args.stop_cnt},epochs:{args.epochs}')
    print(f'global weight:{args.global_weight}')

    base_path = f'../data/{args.dataset}/'

    name2id_lnc, id2name_lnc, features_lnc, node2neigh_list_lnc, _ = load_data_lnc(ltypes=[int(i) for i in args.ltype_lnc.strip().split(',')],
                                                               base_path=base_path,
                                                               use_features=True if args.attributed == 'True' else False)
    name2id_dis, id2name_dis, features_dis, node2neigh_list_dis, _ = load_data_dis(ltypes=[int(i) for i in args.ltype_dis.strip().split(',')],
                                                               base_path=base_path,
                                                               use_features=True if args.attributed == 'True' else False)
    # print(node2neigh_list[0][3024])
    # print(node2neigh_list[1][3024])
    # print(node2neigh_list[2][3024])
    # print(node2neigh_list[3][3024])

    print(f'load lnc dis data finish:{datetime.now()}')

    print('lnc graph num:', len(node2neigh_list_lnc))  # 3
    print('dis graph num:', len(node2neigh_list_dis))  # 3

    print('lnc node num:', len(name2id_lnc))
    print('dis node num:', len(name2id_dis))

    target_nodes_lnc = np.array(list(id2name_lnc.keys()))
    target_nodes_dis = np.array(list(id2name_dis.keys()))

    embeddings_lnc = torch.from_numpy(features_lnc).float().to(args.device)
    embeddings_dis = torch.from_numpy(features_dis).float().to(args.device)

    shuffle_embeddings_lnc = torch.from_numpy(shuffle(features_lnc)).to(args.device)
    shuffle_embeddings_dis = torch.from_numpy(shuffle(features_dis)).to(args.device)

    dim_lnc = embeddings_lnc.shape[-1]
    dim_dis = embeddings_dis.shape[-1]

    adjs_lnc, sim_matrix_list_lnc = PPR_lnc(node2neigh_list_lnc)  # Am Sm
    adjs_dis, sim_matrix_list_dis = PPR_dis(node2neigh_list_dis)  # Am Sm

    print('load adj finish', datetime.now())
    total_train_views_lnc = get_topk_neigh_multi_lnc(target_nodes_lnc, node2neigh_list_lnc, args.topk, adjs_lnc, sim_matrix_list_lnc)
    total_train_views_dis = get_topk_neigh_multi_dis(target_nodes_dis, node2neigh_list_dis, args.topk, adjs_dis, sim_matrix_list_dis)

    print(f'sample finish:{datetime.now()}')
    for node, status, view in total_train_views_lnc:
        for channel_data in view:
            channel_data[0] = torch.from_numpy(channel_data[0]).to(args.device).type(torch.long)  # topk_result
            channel_data[1] = torch.from_numpy(channel_data[1]).to(args.device).type(torch.float32)  # adj_result
            data = embeddings_lnc[channel_data[0]]
            channel_data.append(data.reshape(1, data.shape[0], data.shape[1]))
            shuffle_data = shuffle_embeddings_lnc[channel_data[0]]

            channel_data.append(shuffle_data.reshape(1, shuffle_data.shape[0], shuffle_data.shape[1]))

    for node, status, view in total_train_views_dis:
        for channel_data in view:
            channel_data[0] = list(map(lambda x: x - 240, channel_data[0]))
            channel_data[0] = torch.tensor(channel_data[0]).to(args.device).type(torch.int64)
            # channel_data[0] = torch.from_numpy(channel_data[0]).to(args.device).type(torch.long)  # topk_result
            channel_data[1] = torch.from_numpy(channel_data[1]).to(args.device).type(torch.float32)  # adj_result
            # channel_data[0] = list(map(lambda x: x - 240, channel_data[0]))
            data = embeddings_dis[channel_data[0]]
            channel_data.append(data.reshape(1, data.shape[0], data.shape[1]))
            shuffle_data = shuffle_embeddings_dis[channel_data[0]]

            channel_data.append(shuffle_data.reshape(1, shuffle_data.shape[0], shuffle_data.shape[1]))

    sample_train_views_lnc = [i for i in total_train_views_lnc if sum(i[1]) >= 1]
    sample_train_views_dis = [i for i in total_train_views_dis if sum(i[1]) >= 1]

    print(f'lnc context subgraph num:{len(sample_train_views_lnc)}')
    print(f'dis context subgraph num:{len(sample_train_views_dis)}')

    print(f'sample finish:{datetime.now()}')
    out_dim = args.dim
    # model = CKD(dim_lnc, out_dim, layers=args.layers)
    model = nn.Sequential(
        CKD_lnc(dim_lnc, out_dim, layers=args.layers),
        CKD_dis(dim_dis, out_dim, layers=args.layers),
        FC())
    model = model.cuda()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    criterion = torch.nn.BCEWithLogitsLoss()
    loss_func = nn.CrossEntropyLoss()  # 定义损失函数

    stop_cnt = args.stop_cnt
    min_loss = 100000
    TPR_ALL = []
    FPR_ALL = []
    P_ALL = []

    model = model.train()
    for test in range(5):
        print("目前是第", test+1, "交叉验证")

        positive_sample = Count_Value_1(lnc_dis, 5)
        negative_sample, all_negative_sample = Count_Value_0(lnc_dis, 5)

        Coordinate_Matrix_Train, Coordinate_Matrix_Test, train_1 = Make_Train_Test_Set(positive_sample, negative_sample,
                                                                                       all_negative_sample)

        np.savetxt('../data/save/Coordinate_Matrix_Train_1122night_%d.txt' % test, Coordinate_Matrix_Train)
        np.savetxt("../data/save/Coordinate_Matrix_Test_1122night_%d.txt" % test, Coordinate_Matrix_Test)
        np.savetxt("../data/save/train_1122night_%d.txt" % test, train_1)

        train_loader = load_data(Coordinate_Matrix_Train, 140)
        test_loader = load_data(Coordinate_Matrix_Test, 500)

        for epoch in range(args.epochs):
            if stop_cnt <= 0:
                break

            print(f'run epoch{epoch}')
            losses = []
            local_losses = []
            global_losses = []
            train_views_lnc = shuffle(sample_train_views_lnc)
            train_views_dis = shuffle(sample_train_views_dis)

            steps = (len(train_loader) // args.batch_size) + (0 if len(train_loader) % args.batch_size == 0 else 1)

            # get global emb
            global_graph_emb_list_lnc = []
            neg_global_graph_emb_list_lnc = []

            for channel in range(len(node2neigh_list_lnc)):  # 0, 1, 2, 3
                train_features = torch.cat([i[2][channel][2] for i in total_train_views_lnc], dim=0)
                neg_features = torch.cat([i[2][channel][3] for i in total_train_views_lnc], dim=0)
                train_adj = torch.cat([i[2][channel][1] for i in total_train_views_lnc], dim=0)
                emb, graph_emb = model[0](train_features,
                                       train_adj)  # input: 3025, 21, 100    3025, 21, 21   output: 3025, 21, 100 hi   3025, 100 li
                neg_emb, neg_graph_emb = model[0](neg_features, train_adj)
                index = torch.Tensor([0]).long().cuda()
                emb = emb.index_select(dim=1, index=index).squeeze()  # 3025 100 hi
                global_emb = torch.mean(emb, dim=0).detach()  # 100 pm

                neg_emb = neg_emb.index_select(dim=1, index=index).squeeze()
                global_neg_emb = torch.mean(neg_emb, dim=0).detach()  # 100 pm

                global_graph_emb_list_lnc.append(global_emb)
                neg_global_graph_emb_list_lnc.append(global_neg_emb)

            global_graph_emb_list_dis = []
            neg_global_graph_emb_list_dis = []

            for channel in range(len(node2neigh_list_dis)):  # 0, 1, 2, 3
                train_features = torch.cat([i[2][channel][2] for i in total_train_views_dis], dim=0)
                neg_features = torch.cat([i[2][channel][3] for i in total_train_views_dis], dim=0)
                train_adj = torch.cat([i[2][channel][1] for i in total_train_views_dis], dim=0)
                emb, graph_emb = model[1](train_features,
                                       train_adj)  # input: 3025, 21, 100    3025, 21, 21   output: 3025, 21, 100 hi   3025, 100 li
                neg_emb, neg_graph_emb = model[1](neg_features, train_adj)
                index = torch.Tensor([0]).long().cuda()
                emb = emb.index_select(dim=1, index=index).squeeze()  # 3025 100 hi
                global_emb = torch.mean(emb, dim=0).detach()  # 100 pm

                neg_emb = neg_emb.index_select(dim=1, index=index).squeeze()
                global_neg_emb = torch.mean(neg_emb, dim=0).detach()  # 100 pm

                global_graph_emb_list_dis.append(global_emb)
                neg_global_graph_emb_list_dis.append(global_neg_emb)

            train_loss = 0  # 计算损失是每轮更新
            train_acc = 0
            num_correct_num = 0

            for step, (x, train_label) in tqdm(enumerate(train_loader)):
                start = step * args.batch_size
                end = min((step + 1) * args.batch_size, len(train_views_lnc))
                if end - start <= 1:
                    continue
                step_train_views_lnc = train_views_lnc[start:end]
                step_train_views_dis = train_views_dis[start:end]
                # x_A = int(x[:][0][0][start:end])
                # y_A = int(x[:][0][0][1])

                emb_list_lnc = []
                graph_emb_list_lnc = []

                train_label1 = []
                # emb_np = []
                x = x.squeeze()
                # a = x[0][0]
                # b = x[0][1]
                # c = x[1][0]
                # d = x[1][1]

                x_A = list(x[i][0] for i in range(x.shape[0]))  # lnc id 0-239 6, 1, 1, 1
                y_A = list(x[i][1] for i in range(x.shape[0]))  # dis id 0-404 6, 1, 1, 1

                x_A = torch.tensor(x_A).int()
                y_A = torch.tensor(y_A).int()

                # a = x_A[0]
                # x_A = x_A.unsqueeze()

                step_train_views_lnc_sample = []
                step_train_views_dis_sample = []

                for i in range(x_A.shape[0]):
                    # total_train_views_lnc = torch.tensor(total_train_views_lnc)
                    step_train_views_lnc_sample.append(total_train_views_lnc[x_A[i]])

                for i in range(y_A.shape[0]):
                    step_train_views_dis_sample.append(total_train_views_dis[y_A[i]])

                # step_train_views_lnc_sample = [i for i in total_train_views_lnc if sum(i[0]) >= 1]

                for channel in range(len(node2neigh_list_lnc)):  # 0, 1, 2, 3
                    train_features_lnc = torch.cat([i[2][channel][2] for i in step_train_views_lnc_sample], dim=0)  # 6, 21, 100
                    train_adj_lnc = torch.cat([i[2][channel][1] for i in step_train_views_lnc_sample], dim=0)  # 6, 21, 21
                    emb_lnc, graph_emb_lnc = model[0](train_features_lnc, train_adj_lnc)
                    emb_list_lnc.append(emb_lnc)
                    graph_emb_list_lnc.append(graph_emb_lnc)
                    # lnc_label.append([i[0] for i in step_train_views_lnc_sample])

                emb_list_dis = []
                graph_emb_list_dis = []
                for channel in range(len(node2neigh_list_dis)):  # 0, 1, 2, 3
                    train_features_dis = torch.cat([i[2][channel][2] for i in step_train_views_dis_sample], dim=0)  # 6, 21, 100
                    train_adj_dis = torch.cat([i[2][channel][1] for i in step_train_views_dis_sample], dim=0)  # 6, 21, 21
                    emb_dis, graph_emb_dis = model[1](train_features_dis, train_adj_dis)
                    emb_list_dis.append(emb_dis)
                    graph_emb_list_dis.append(graph_emb_dis)
                    # dis_label.append([i[0] for i in step_train_views_dis])

                local_loss_lnc = score(criterion, emb_list_lnc, graph_emb_list_lnc, [i[1] for i in step_train_views_lnc_sample])
                global_loss_lnc = global_score(criterion, emb_list_lnc, global_graph_emb_list_lnc, neg_global_graph_emb_list_lnc,
                                           [i[1] for i in step_train_views_lnc_sample])

                local_loss_dis = score(criterion, emb_list_dis, graph_emb_list_dis, [i[1] for i in step_train_views_dis_sample])
                global_loss_dis = global_score(criterion, emb_list_dis, global_graph_emb_list_dis, neg_global_graph_emb_list_dis,
                                           [i[1] for i in step_train_views_dis_sample])
                emb_list_lnc = sum(emb_list_lnc)
                emb_list_dis = sum(emb_list_dis)

                emb_np = torch.cat((emb_list_lnc, emb_list_dis), dim=1)

                pre = model[2](emb_np)

                train_label1.extend(train_label)  # 存储每数据批的标签
                y = torch.LongTensor(np.array(train_label1).astype(int32))  # 转化为Tensor
                y = Variable(y).to(device)  # tensor变成variable之后才能进行反向传播求梯度
                pre = torch.cat(pre, 0)
                loss_c = loss_func(pre, y)  # 计算损失函数
                train_loss += loss_c.item()  # 取tensor的元素值
                # 计算准确率
                _, pred = pre.max(1)  # 取最大值的索引，得到预测标签
                num_correct = (pred == y).sum().item()  # 计算每批预测正确的数目
                print(num_correct)
                num_correct_num = num_correct + num_correct_num

                acc = num_correct / x.shape[0]  # 批预测正确率
                train_acc += acc  # 总正确率

                loss = local_loss_lnc + local_loss_dis + global_loss_lnc * args.global_weight + global_loss_dis * args.global_weight + loss_c
                losses.append(loss.item())
                local_losses.append(local_loss_lnc.item())
                global_losses.append(global_loss_lnc.item())

                loss.backward()

                optimizer.step()
                optimizer.zero_grad()
            train_acc = num_correct_num/4298

            epoch_loss = np.mean(losses)
            print(f'epoch:{epoch},loss:{np.mean(losses)},{np.mean(local_losses)},{np.mean(global_losses)}')
            print(f'min_loss:{min_loss},epoch_loss:{epoch_loss}', epoch_loss < min_loss)
            print('Epoch: {}, Train Loss: {:.8f}, Train Acc: {:.6f}, Num_cor: {}'
                  .format(epoch, loss, train_acc, num_correct_num))
        torch.save(model.state_dict(), "../data/save/ckd_1122night_%d.pth" % test)
        print("ckd模型参数保存成功")

        #####################测试CNN模型###############################
        model.load_state_dict(torch.load("../data/save/ckd_1122night_%d.pth" % test))

        model.eval()
        test_acc = 0
        num_cor = 0
        o = np.zeros((0, 2))

        for test_x, test_label in test_loader:
            test_label1 = []
            lnc_dis_attr = []

            test_x = test_x.squeeze()

            x_A = list(test_x[i][0] for i in range(test_x.shape[0]))  # lnc id 0-239 6, 1, 1, 1
            y_A = list(test_x[i][1] for i in range(test_x.shape[0]))

            x_A = torch.tensor(x_A).int()
            y_A = torch.tensor(y_A).int()

            step_test_views_lnc_sample = []
            step_test_views_dis_sample = []

            emb_list_lnc = []
            emb_list_dis = []
            # eval_size = args.batch_size
            # eval_steps = (len(total_train_views_lnc) // args.batch_size) + (
            #     0 if len(total_train_views_lnc) % args.batch_size == 0 else 1)
            for i in range(x_A.shape[0]):
                step_test_views_lnc_sample.append(total_train_views_lnc[x_A[i]])
            for i in range(y_A.shape[0]):
                step_test_views_dis_sample.append(total_train_views_dis[y_A[i]])
            for channel in range(len(node2neigh_list_lnc)):
                temp_emb_list_lnc = []
                train_features_lnc = torch.cat([i[2][channel][2] for i in step_test_views_lnc_sample], dim=0)
                train_adj_lnc = torch.cat([i[2][channel][1] for i in step_test_views_lnc_sample], dim=0)
                emb_lnc, graph_emb_lnc = model[0](train_features_lnc, train_adj_lnc)
                index = torch.Tensor([0]).long().cuda()
                emb_lnc = emb_lnc.index_select(dim=1, index=index).squeeze(dim=1)
                # emb_lnc = emb_lnc.cpu().detach().numpy()
                # temp_emb_list_lnc.append(emb_lnc)
                #
                # emb_lnc = np.concatenate(temp_emb_list_lnc, axis=0)
                emb_list_lnc.append(emb_lnc)

            for channel in range(len(node2neigh_list_dis)):
                temp_emb_list_dis = []
                train_features_dis = torch.cat([i[2][channel][2] for i in step_test_views_dis_sample], dim=0)
                train_adj_dis = torch.cat([i[2][channel][1] for i in step_test_views_dis_sample], dim=0)
                emb_dis, graph_emb_dis = model[1](train_features_dis, train_adj_dis)
                index = torch.Tensor([0]).long().cuda()
                emb_dis = emb_dis.index_select(dim=1, index=index).squeeze(dim=1)
                # emb_dis = emb_dis.cpu().detach().numpy()
                # temp_emb_list_dis.append(emb_dis)
                #
                # emb_dis = np.concatenate(temp_emb_list_dis, axis=0)
                emb_list_dis.append(emb_dis)
            emb_list_lnc = sum(emb_list_lnc)
            emb_list_dis = sum(emb_list_dis)
            # emb_list_lnc = torch.tensor(emb_list_lnc)
            # emb_list_dis = torch.tensor(emb_list_dis)
            emb_np = torch.cat((emb_list_lnc, emb_list_dis), dim=1)

            pre = model[2](emb_np)
            test_label1.extend(test_label)
            y = torch.LongTensor(np.array(test_label1).astype(int32))
            y = Variable(y).to(device)
            pre = torch.tensor([item.cpu().detach().numpy() for item in pre]).cuda()
            pre = pre.squeeze()
            pre = F.softmax(pre, dim=1)
            _, pred_y = pre.max(1)
            num_correct = (pred_y == y).sum().item()
            num_cor += num_correct
            o = np.vstack((o, pre.detach().cpu().numpy()))
        print('cor_num:{}'.format(num_cor))
        np.save("../data/save/test_out_1122night_left_%d.npy"%test, o)

        test_out_left = np.load("../data/save/test_out_1122night_left_%d.npy" % test)
        # test_out_right = np.load("data/test_out_last_right0418night_%d.npy" % test)
        R = np.zeros(shape=(lnc_dis.shape[0], lnc_dis.shape[1]))
        r = 0.2
        for i in range(Coordinate_Matrix_Test.shape[0]):
            # R[int(Coordinate_Matrix_Test[i][0])][int(Coordinate_Matrix_Test[i][1])] = r*test_out_right[i][1]+(1-r)*test_out_left[i][1]
            R[int(Coordinate_Matrix_Test[i][0])][int(Coordinate_Matrix_Test[i][1])] = test_out_left[i][1]
            # R[int(Coordinate_Matrix_Test[i][0])][int(Coordinate_Matrix_Test[i][1])] = test_out_right[i][1]
        B = lnc_dis / 1  # 相当于copy
        for i in range(Coordinate_Matrix_Train.shape[0]):
            B[int(Coordinate_Matrix_Train[i][0])][int(Coordinate_Matrix_Train[i][1])] = -1
            R[int(Coordinate_Matrix_Train[i][0])][int(Coordinate_Matrix_Train[i][1])] = -1
        np.save("./R_0418night_last%d.npy" % test, R)
        correct = 0
        for i in range(train_1.shape[0]):
            if R[int(train_1[i][0])][int(train_1[i][1])] > 0.5:
                correct = correct + 1
        print('测试集中正例预测对的个数：', correct)
        f = np.zeros(shape=(R.shape[0], 1))
        for i in range(R.shape[0]):
            f[i] = np.sum(R[i] >= 0)
        TPR, FPR, P = calculate_TPR_FPR(R, f, B)
        np.savetxt('../data/checkout/res/FPR_0418night%d.txt' % test, FPR)
        np.savetxt('../data/checkout/res/TPR_0418night%d.txt' % test, TPR)
        np.savetxt('../data/checkout/res/P_0418night%d.txt' % test, P)
        curve(FPR, TPR, P)
        TPR_ALL.append(TPR)
        FPR_ALL.append(FPR)
        P_ALL.append(P)
    A, B, C = fold_5(TPR_ALL, FPR_ALL, P_ALL)
    curve(B, A, C)


if __name__ == '__main__':
    main()
