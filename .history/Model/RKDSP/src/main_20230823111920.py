import sys
print(sys.getdefaultencoding())
from datetime import  datetime
import math
import argparse
import numpy as np
import networkx as nx
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.utils import shuffle
from utils import *
from model import *
torch.backends.cudnn.benchmark = True
from sklearn.metrics import roc_auc_score,average_precision_score
from functools import reduce
from tqdm import tqdm
def parse_args():
    parser = argparse.ArgumentParser(description='ckd')

    parser.add_argument('--output', type=str, required=False,help='emb file',default='emb.dat')
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

    if args.dataset=='ACM2':
        args.dim=100
        args.ltype = '0,1'
        args.lr = 0.00002
        args.topk = 15
        args.epochs = 60
        args.stop_cnt = 100
        args.global_weight = 0.1
        args.batch_size=4
    elif args.dataset=='ACM':
        args.dim = 100
        args.ltype = '0,2,4,6'
        args.lr = 0.0001
        args.topk = 20 #25
        args.epochs = 100
        args.stop_cnt = 100
        args.global_weight= 0.05
        args.batch_size = 6
        args.seed = 7
    elif args.dataset=='lnc':
        args.dim = 100
        args.ltype = '0,1,2'
        args.lr = 0.0001
        args.topk = 20 #25
        args.epochs = 100
        args.stop_cnt = 100
        args.global_weight = 0.05
        args.batch_size = 6
        args.seed = 7
    elif args.dataset=='DBLP2':
        args.dim=300
        args.ltype = '1,3,4,5'
        #args.lr = 0.00001
        args.lr = 0.00002
        args.topk = 35
        args.epochs = 60
        args.stop_cnt = 100
        args.global_weight = 0.12
        args.batch_size=2
        args.seed = 11
    elif args.dataset=='DBLP':
        args.ltype='0,1,2'
        args.lr=0.00005
        args.topk=30 #25
        args.epochs=200
        args.stop_cnt=100
        args.global_weight=0.1
    elif args.dataset=='Freebase':
        args.dim=200
        args.ltype='0,1,2,3,4'
        args.lr=0.00002
        args.topk=35 #25
        args.epochs=150
        args.stop_cnt=100
        args.global_weight=0.15
        args.batch_size=2
    elif args.dataset=='PubMed':
        args.dim=200
        args.ltype='1,2,4'
        args.lr=0.0002
        args.topk=25 
        args.epochs=60
        args.stop_cnt=100
        args.global_weight=0.1
        args.batch_size=2
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
            mask_idx = torch.Tensor([i for i in range(len(status_list)) if status_list[i][idy] == 0]).long().cuda()  # 找到六个节点中的在第一个元路径中没有邻居的节点索引
            neg_mask = np.ones(shape=(node_emb.shape[0], node_emb.shape[0]))  # 生成全是1的6, 6矩阵
            row, col = np.diag_indices_from(neg_mask)  # 获取主对角线元素的索引
            neg_mask[row, col] = 0
            neg_mask = torch.from_numpy(neg_mask).bool().cuda()
            neg_mask[mask_idx, ] = 0
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

def global_score(criterion, emb_list, graph_emb_list, neg_graph_emb_list,status_list):
    loss = None

    for idx in range(len(emb_list)):
        for idy in range(len(emb_list)):
            node_emb = emb_list[idx]
            global_emb = graph_emb_list[idy]
            neg_global_emb = neg_graph_emb_list[idy]
            mask = torch.Tensor([i[idx] for i in status_list]).bool().cuda()
            pos = torch.sum(node_emb * global_emb, dim=1).squeeze().masked_select(mask)
            neg = torch.sum(node_emb * neg_global_emb, dim=1).squeeze().masked_select(mask)
            if pos.shape[0]==0:
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

    print(f'dataset:{args.dataset},attributed:{args.attributed},ltypes:{args.ltype},topk:{args.topk},lr:{args.lr},batch-size:{args.batch_size},stop_cnt:{args.stop_cnt},epochs:{args.epochs}')
    print(f'global weight:{args.global_weight}')

    base_path = f'../data/{args.dataset}/'
    name2id, id2name, features, node2neigh_list, _ = load_data_drug(ltypes=[int(i) for i in args.ltype.strip().split(',')], base_path=base_path, use_features=True if args.attributed=='True' else False)
    # print(node2neigh_list[0][3024])
    # print(node2neigh_list[1][3024])
    # print(node2neigh_list[2][3024])
    # print(node2neigh_list[3][3024])

    print(f'load data finish:{datetime.now()}')
    print('graph num:', len(node2neigh_list))
    print('node num:', len(name2id))

    target_nodes = np.array(list(id2name.keys()))
    if args.attributed != "True":
        features = np.random.randn(len(target_nodes), args.size).astype(np.float32)
    embeddings = torch.from_numpy(features).float().to(args.device)
    shuffle_embeddings = torch.from_numpy(shuffle(features)).to(args.device)

    dim = embeddings.shape[-1]

    adjs, sim_matrix_list = PPR_drug(node2neigh_list)  # Am Sm
 
    print('load adj finish', datetime.now())
    total_train_views = get_topk_neigh_multi_drug(target_nodes, node2neigh_list, args.topk, adjs, sim_matrix_list)
    print(f'sample finish:{datetime.now()}')
    for node, status, view in total_train_views:
        for channel_data in view:
            channel_data[0] = torch.from_numpy(channel_data[0]).to(args.device).type(torch.long)  # topk_result
            channel_data[1] = torch.from_numpy(channel_data[1]).to(args.device).type(torch.float32)  # adj_result
            data = embeddings[channel_data[0]]
            channel_data.append(data.reshape(1, data.shape[0], data.shape[1]))
            shuffle_data = shuffle_embeddings[channel_data[0]]

            channel_data.append(shuffle_data.reshape(1, shuffle_data.shape[0], shuffle_data.shape[1]))

    sample_train_views=[i for i in total_train_views if sum(i[1])>=1]
    print(f'context subgraph num:{len(sample_train_views)}')

    print(f'sample finish:{datetime.now()}')
    out_dim = args.dim
    model = CKD_drug(dim, out_dim, layers=args.layers)
    model = model.cuda()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    criterion = torch.nn.BCEWithLogitsLoss()

    stop_cnt = args.stop_cnt
    min_loss = 100000
    for epoch in range(args.epochs):
        if stop_cnt <= 0:

            break

        print(f'run epoch{epoch}')
        losses = []
        local_losses = []
        global_losses = []
        train_views = shuffle(sample_train_views)
        steps = (len(train_views) // args.batch_size) + (0 if len(train_views) % args.batch_size == 0 else 1)

        # get global emb
        global_graph_emb_list = []
        neg_global_graph_emb_list=[]

        for channel in range(len(node2neigh_list)):  # 0, 1, 2, 3
            train_features = torch.cat([i[2][channel][2] for i in total_train_views], dim=0)
            neg_features = torch.cat([i[2][channel][3] for i in total_train_views], dim=0)
            train_adj = torch.cat([i[2][channel][1] for i in total_train_views], dim=0)
            emb, graph_emb = model(train_features, train_adj)  # input: 3025, 21, 100    3025, 21, 21   output: 3025, 21, 100 hi   3025, 100 li
            neg_emb, neg_graph_emb = model(neg_features, train_adj)
            index = torch.Tensor([0]).long().cuda()
            emb = emb.index_select(dim=1, index=index).squeeze()  # 3025 100 hi
            global_emb = torch.mean(emb, dim=0).detach()  # 100 pm

            neg_emb = neg_emb.index_select(dim=1, index=index).squeeze()
            global_neg_emb = torch.mean(neg_emb, dim=0).detach()  # 100 pm

            global_graph_emb_list.append(global_emb)
            neg_global_graph_emb_list.append(global_neg_emb)

        for step in tqdm(range(steps)):
            start = step * args.batch_size
            end = min((step + 1) * args.batch_size, len(train_views))
            if end-start <= 1:
                continue
            step_train_views = train_views[start:end]

            emb_list = []
            graph_emb_list = []
            for channel in range(len(node2neigh_list)):  # 0, 1, 2, 3
                train_features=torch.cat([i[2][channel][2] for i in step_train_views], dim=0)  # 6, 21, 100
                train_adj = torch.cat([i[2][channel][1] for i in step_train_views], dim=0)  # 6, 21, 21
                emb, graph_emb = model(train_features, train_adj)
                emb_list.append(emb)
                graph_emb_list.append(graph_emb)

            local_loss = score(criterion, emb_list, graph_emb_list, [i[1] for i in step_train_views])
            global_loss = global_score(criterion, emb_list, global_graph_emb_list, neg_global_graph_emb_list, [i[1] for i in step_train_views])
            loss = local_loss+global_loss*args.global_weight
            losses.append(loss.item())
            local_losses.append(local_loss.item())
            global_losses.append(global_loss.item())

            loss.backward()

            optimizer.step()
            optimizer.zero_grad()

        epoch_loss = np.mean(losses)
        print(f'epoch:{epoch},loss:{np.mean(losses)},{np.mean(local_losses)},{np.mean(global_losses)}')
        print(f'min_loss:{min_loss},epoch_loss:{epoch_loss}',epoch_loss<min_loss)

        if args.attributed == "True":
            emb_list = []
            eval_size = args.batch_size
            eval_steps = (len(total_train_views) // args.batch_size) + (0 if len(total_train_views) % args.batch_size == 0 else 1)
            for channel in range(len(node2neigh_list)):
                temp_emb_list = []
                for eval_step in range(eval_steps):
                    start = eval_step * eval_size
                    end = min((eval_step + 1) * eval_size, len(total_train_views))
                    step_eval_views = total_train_views[start:end]
                    train_features = torch.cat([i[2][channel][2] for i in step_eval_views], dim=0)
                    train_adj = torch.cat([i[2][channel][1] for i in step_eval_views], dim=0)
                    emb, graph_emb = model(train_features, train_adj)
                    index = torch.Tensor([0]).long().cuda()
                    emb = emb.index_select(dim=1, index=index).squeeze(dim=1)
                    emb = emb.cpu().detach().numpy()
                    temp_emb_list.append(emb)
                emb=np.concatenate(temp_emb_list, axis=0)
                emb_list.append(emb)
        else:
            emb_list = []
            for channel in range(len(node2neigh_list)):
                train_features = torch.cat([i[2][channel][2] for i in total_train_views], dim=0)
                train_adj = torch.cat([i[2][channel][1] for i in total_train_views], dim=0)
                emb, graph_emb = model(train_features, train_adj)
                index = torch.Tensor([0]).long().cuda()
                emb = emb.index_select(dim=1, index=index).squeeze()
                emb_list.append(emb)
        # epoch_auc=load_link_test_data(args.dataset,emb_list,name2id,)
        if epoch_loss < min_loss:
            min_loss = epoch_loss
            stop_cnt = args.stop_cnt
            output(args, emb_list[:1], id2name, need_handle=True)
            print(
                f'--------------------------------------------------------------------------------------------------------')
            """
            print(
                f'--------------------------------------------------------------------------------------------------------')
            print(
                f'-------------------------------------------save auc{epoch_auc}-------------------------------------------')
            print(
                f'--------------------------------------------------------------------------------------------------------')
            """


if __name__ == '__main__':
    main()
