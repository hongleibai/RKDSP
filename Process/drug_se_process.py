import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from numpy import *
from torch import optim
from sklearn.metrics import auc
import torch.utils.data as Data
from torch.autograd import Variable
from numpy import ndarray, eye, matmul, vstack, hstack, array, newaxis, zeros, genfromtxt, savetxt, exp
import random
import time
import datetime
import os
from collections import defaultdict
import networkx as nx
from pylab import show
# import org.apache.commons.lang.StringUtils;

node_file, link_file, label_file = 'node.dat', 'link.dat', 'label.dat'
info_file, meta_file = 'info.dat', 'meta.dat'

def cosine(arr):
    dot=arr@arr.T
    modulus=np.expand_dims(np.sqrt(np.diagonal(dot)),0)
    return (dot/(modulus.T @ modulus))

def ReadTxt():
    drug_drugc = np.loadtxt('Model/CKD/data/drug/Similarity_Matrix_Drugs.txt')
    drug_se = np.loadtxt('Model/CKD/data/drug/mat_drug_se.txt')
    se_se_sim = cosine(drug_se.T)
    se_se_sim[np.isnan(se_se_sim)] = 0
    se_se = np.array(se_se_sim)
    np.savetxt("data/se_se_sim.txt", se_se_sim)
    drug_drugd = np.loadtxt('Model/CKD/data/drug/drug_drug_sim_dis.txt')

    return drug_drugc, drug_se, se_se, drug_drugd

def ReadTxt2():
    new_node_ll_file = open(f'Model/CKD/data/lnc/newnode_ll.dat', 'w+')
    new_node_ld_file = open(f'Model/CKD/data/lnc/newnode_ld.dat', 'w+')
    new_node_lm_file = open(f'Model/CKD/data/lnc/newnode_lm.dat', 'w+')
    new_node_dl_file = open(f'Model/CKD/data/lnc/newnode_dl.dat', 'w+')
    new_node_dd_file = open(f'Model/CKD/data/lnc/newnode_dd.dat', 'w+')
    new_node_dm_file = open(f'Model/CKD/data/lnc/newnode_dm.dat', 'w+')

    dis_sim = np.loadtxt('Data/lnc/dis_sim.txt')
    with open('Data/lnc/dis_sim.txt', 'r') as f:
        lines = f.readlines()
        print('node_num')
        print(len(lines))

        print('feat dim')
        tp = lines[0]
        tp = tp.split()
        # tp = tp[3].split(' ')
        print(len(tp))
        for i in range(len(lines)):
            line = lines[i]
            line = line.split(' ')
            line = ','.join(line)
            new_node_dd_file.write(f'{line}\n')
    lnc_sim = np.loadtxt('Data/lnc/lnc_sim.txt')
    with open('Data/lnc/lnc_sim.txt', 'r') as f:
        lines = f.readlines()
        print('node_num')
        print(len(lines))

        print('feat dim')
        tp = lines[0]
        tp = tp.split()
        # tp = tp[3].split(' ')
        print(len(tp))
        for i in range(len(lines)):
            line = lines[i]
            line = line.split(' ')
            line = ','.join(line)
            new_node_ll_file.write(f'{line}\n')
    lnc_dis = np.loadtxt('Data/lnc/lnc_dis.txt')
    np.savetxt('Data/lnc/dis_lnc.txt', lnc_dis.T)
    with open('Data/lnc/dis_lnc.txt', 'r') as f:
        lines = f.readlines()
        print('node_num')
        print(len(lines))

        print('feat dim')
        tp = lines[0]
        tp = tp.split()
        # tp = tp[3].split(' ')
        print(len(tp))
        for i in range(len(lines)):
            line = lines[i]
            line = line.split(' ')
            line = ','.join(line)
            new_node_dl_file.write(f'{line}\n')
    with open('Data/lnc/lnc_dis.txt', 'r') as f:
        lines = f.readlines()
        print('node_num')
        print(len(lines))

        print('feat dim')
        tp = lines[0]
        tp = tp.split()
        # tp = tp[3].split(' ')
        print(len(tp))
        for i in range(len(lines)):
            line = lines[i]
            line = line.split(' ')
            line = ','.join(line)
            new_node_ld_file.write(f'{line}\n')
    mi_dis = np.loadtxt('Data/lnc/mi_dis.txt')
    np.savetxt('Data/lnc/dis_mi.txt', mi_dis.T)
    with open('Data/lnc/dis_mi.txt', 'r') as f:
        lines = f.readlines()
        print('node_num')
        print(len(lines))

        print('feat dim')
        tp = lines[0]
        tp = tp.split()
        # tp = tp[3].split(' ')
        print(len(tp))
        for i in range(len(lines)):
            line = lines[i]
            line = line.split(' ')
            line = ','.join(line)
            new_node_dm_file.write(f'{line}\n')
    with open('Data/lnc/lnc_mi.txt', 'r') as f:
        lines = f.readlines()
        print('node_num')
        print(len(lines))

        print('feat dim')
        tp = lines[0]
        tp = tp.split()
        # tp = tp[3].split(' ')
        print(len(tp))
        for i in range(len(lines)):
            line = lines[i]
            line = line.split(' ')
            line = ','.join(line)
            new_node_lm_file.write(f'{line}\n')
    # drug_dis_mat_noname = np.loadtxt('data/drug_dis_mat_noname.txt')
    new_node_ll_file.close()
    new_node_ld_file.close()
    new_node_lm_file.close()
    new_node_dl_file.close()
    new_node_dd_file.close()
    new_node_dm_file.close()

def calculate_sim(Interaction, oringe_sim):
    target_sim = np.zeros(shape=(Interaction.shape[0], Interaction.shape[0]), dtype=float)
    for i in range(target_sim.shape[0]):
        for j in range(target_sim.shape[1]):
            if i == j:
                target_sim[i][j] = 1
            else:
                l1_num = np.sum(Interaction[i] == 1.0)
                l2_num = np.sum(Interaction[j] == 1.0)
                if l1_num == 0 or l2_num == 0:
                    target_sim[i][j] = 0
                else:
                    l1_index = np.where(Interaction[i] == 1.0)
                    l2_index = np.where(Interaction[j] == 1.0)
                    sim_sum = 0.0
                    for l in range(len(l1_index[0])):
                        sim_sum = sim_sum + np.max(oringe_sim[l1_index[0][l]][l2_index[0]])
                    for l in range(len(l2_index[0])):
                        sim_sum = sim_sum + np.max(oringe_sim[l2_index[0][l]][l1_index[0]])
                    target_sim[i][j] = sim_sum/(l1_num + l2_num)
    return target_sim

if __name__=='__main__':
    drug_drugc, drug_se, se_se, drug_drugd = ReadTxt()
    # ReadTxt2()
    # dis_sim = torch.from_numpy(dis_sim)
    # lnc_sim = torch.from_numpy(lnc_sim)
    # lnc_dis = torch.from_numpy(lnc_dis)
    # mi_dis = torch.from_numpy(mi_dis)
    # lnc_mi = torch.from_numpy(lnc_mi)

    model_data_folder = f'Model/CKD/data/drug'
    ori_data_folder = f'Data/drug'

    old_node_file = open(f'{model_data_folder}/oldnode.dat', 'w')
    new_node_file = open(f'{model_data_folder}/newnode.dat', 'w')

    # original_node_file = open(f'Model/CKD/data/ACM/link/{node_file}', 'r')
    # all_feature_file = open(f'{model_data_folder}/node.dat', 'r')
    # for line in original_node_file:
    #     line = line[:-1].split('\t')
    #     # nine = nine[:-1].split('\t')
    #     b = line[2]
    #     old_node_file.write(f'{b}\n')
    #     # new_node_file.write(f'{nine[2]}\n')
    #     # new_node_file.write(f'{line[2]}\n')
    # for line in all_feature_file:
    #     line = line[:-1].split('\t')
    #     a = line[2]
    #     new_node_file.write(f'{a}\n')


    # mi_sim = calculate_sim(mi_dis, dis_sim)
    # mi_sim = torch.from_numpy(mi_sim)
    # dis_sim = torch.from_numpy(dis_sim)
    # mi_dis = torch.from_numpy(mi_dis)
    #

    row_1 = np.concatenate((drug_drugc, drug_se), axis=1)
    row_2 = np.concatenate((drug_se.T, se_se), axis=1)
    # row_3 = np.concatenate((lnc_mi.T, mi_dis, mi_sim), axis=1)
    # row_1 = row_1.tolist()
    # row_2 = row_2.tolist()
    # row_3 = row_3.tolist()
    roww_d = list()
    roww_s = list()
    # roww_3 = list()

    row_d = list()
    row_s = list()
    # row_m = list()

    for i in range(708):
        a = ' '.join(map(str, row_1[i].ravel().tolist()))
        # a = ' '.join(map(str, row_1[i]))
        # a = float(a)
        roww_d.append(a)
    for i in range(4192):
        a = ' '.join(map(str, row_2[i].ravel().tolist()))
        a = ' '.join(map(str, row_2[i]))
        roww_s.append(a)
    # for i in range(495):
    #     a = ' '.join(map(str, row_3[i].ravel().tolist()))
    #     roww_3.append(a)

    lnc_node_file = open(f'{model_data_folder}/ll_nodefeature.dat', 'w+')
    with open(f'{model_data_folder}/drug_drug_node.txt', 'w+') as f:
        for i in range(len(roww_d)):
            a = roww_d[i]
            a = a.split(' ')
            a = ','.join(a)
            row_d.append(a)
            # lnc_node_file.write(f'{a}\n')
        for i in range(len(roww_s)):
            a = roww_s[i]
            a = a.split(' ')
            a = ','.join(a)
            row_s.append(a)

    # with open(f'{model_data_folder}/last_ll_node.dat', 'r') as ll_file:
    #
    #     lines = f.readlines()
    #     for i in range(len(lines)):
    #         line = lines[i]
    #         line = line.split(' ')
    #         line = ','.join(line)
    #         ll_file.write(f'{line}\n')
    #
    # new_node_ll_file = open(f'Model/CKD/data/lnc/newnode_ll.dat', 'r')
    # ll_line = new_node_ll_file.readlines()
    # new_node_ld_file = open(f'Model/CKD/data/lnc/newnode_ld.dat', 'r')
    # new_node_lm_file = open(f'Model/CKD/data/lnc/newnode_lm.dat', 'r')
    # new_node_dl_file = open(f'Model/CKD/data/lnc/newnode_dl.dat', 'r')
    # new_node_dd_file = open(f'Model/CKD/data/lnc/newnode_dd.dat', 'r')
    # new_node_dm_file = open(f'Model/CKD/data/lnc/newnode_dm.dat', 'r')

    with open(f'{model_data_folder}/node.dat', 'w+') as all_feature_file:

        for line in range(708):
            # line = line[:-1].split('\t')
            all_feature_file.write(f'{line}\t{0}\t{row_d[line]}\n')
        for line in range(4192):
            # line = line[:-1].split('\t')
            all_feature_file.write(f'{line+708}\t{1}\t{row_s[line]}\n')
        # for line in range(495):
        #     # line = line[:-1].split('\t')
        #     all_feature_file.write(f'{line+645}\t{2}\t{row_m[line]}\n')
    node2id = {}
    target_node = 1
    useful_types = []
    #
    with open(f'{model_data_folder}/node.dat', 'r') as all_feature_file:
        for line in all_feature_file:
            line = line[:-1].split('\t')
            if int(line[1]) == target_node:
                node2id[int(line[0])] = len(node2id)+708

    savedd = []
    for i in range(708):
        temp_savedd = []
        for dd in range(708):
            if drug_drugc[i][dd] != 0:
                temp_savedd.extend([dd])
        savedd.append(temp_savedd)
    print(np.array(savedd).shape)
    gnn_dd_input = []
    for i in range(708):
        savedd_score = savedd
        dd_len = len(savedd[i])
        if dd_len <= 20:
            dd_output = savedd[i]
            # ll_output = [j for j in ll_output if j != i]
            dd_output = np.array(dd_output)
            gnn_dd_input.append(dd_output)
            # print(gnn_input)
        else:
            dd_zero = np.zeros((dd_len, 2))
            for k in range(dd_len):
                dd_zero[k][1] = savedd[i][k]
                dd_zero[k][0] = drug_drugc[i][savedd[i][k]]
            dd_sort = dd_zero[np.lexsort(-dd_zero[:, ::-1].T)]
            dd_output = dd_sort[:, 1]
            # ll_output = [m for m in ll_output if m != i]
            dd_output = np.array(dd_output)
            gnn_dd_input.append(dd_output)
            # print(gnn_input)
        print('目前是第', i, '个drug', 'dd')
    for x in range(708):
        gnn_dd_input[x] = gnn_dd_input[x].astype(np.int)

    savess = []
    for i in range(4192):
        temp_savess = []
        for ss in range(4192):
            if se_se[i][ss] != 0:
                temp_savess.extend([ss])
        savess.append(temp_savess)
    print(np.array(savess).shape)
    gnn_ss_input = []
    for i in range(4192):
        savess_score = savess
        ss_len = len(savess[i])
        if ss_len <= 20:
            ss_output = savess[i]
            # dd_output = [j for j in dd_output if j != i]
            ss_output = np.array(ss_output)
            gnn_ss_input.append(ss_output)
            # print(gnn_input)
        else:
            ss_zero = np.zeros((ss_len, 2))
            for k in range(ss_len):
                ss_zero[k][1] = savess[i][k]
                ss_zero[k][0] = se_se[i][savess[i][k]]
            ss_sort = ss_zero[np.lexsort(-ss_zero[:, ::-1].T)]
            ss_output = ss_sort[:, 1]
            # dd_output = [m for m in dd_output if m != i]
            ss_output = np.array(ss_output)
            gnn_ss_input.append(ss_output)
            # print(gnn_input)
        print('目前是第', i, '个se', 'ss')
    # np.save('data/dosome/dd.npy', gnn_input)
    for x in range(4192):
        gnn_ss_input[x] = gnn_ss_input[x].astype(np.int)

    ltypes = []

    with open(f'{ori_data_folder}/{info_file}', 'r') as original_info_file:
        lstart = False
        for line in original_info_file:
            if line.startswith('LINK'):
                lstart = True
                continue
            if lstart and line[0] == '\n': break
            if lstart:
                line = line[:-1].split('\t')
                ltype, snode, enode, _ = list(filter(lambda x: len(x) != 0, line))
                ltypes.append((snode, enode, ltype))

    with open(f'{model_data_folder}/link.dat', 'w+') as link_file:
        for i in range(708):
            for j in range(len(gnn_dd_input[i])):
                link_file.write(f'{i}\t{gnn_dd_input[i][j]}\t{0}\n')
        for i in range(708):
            for j in range(4192):
                if drug_se[i][j] != 0:
                    link_file.write(f'{i}\t{j+708}\t{1}\n')
        for i in range(4192):
            for j in range(708):
                if drug_se.T[i][j] != 0:
                    link_file.write(f'{i+708}\t{j}\t{2}\n')
        for i in range(4192):
            for j in range(len(gnn_ss_input[i])):
                if se_se[i][j] != 0:
                    link_file.write(f'{i+708}\t{j+708}\t{3}\n')
        # for i in range(405):
        #     for j in range(len(gnn_dd_input[i])):
        #         if dis_sim[i][j] != 0:
        #             link_file.write(f'{i+240}\t{gnn_dd_input[i][j]+240}\t{4}\n')
        # for i in range(mi_dis.T.shape[0]):
        #     for j in range(mi_dis.T.shape[1]):
        #         if mi_dis.T[i][j] != 0:
        #             link_file.write(f'{i+240}\t{j+645}\t{5}\n')

    type_corners = {int(ltype[2]): defaultdict(set) for ltype in ltypes}
    with open(f'{model_data_folder}/link.dat', 'r') as link_file:
        for line in link_file:
            left, right, ltype = line[:-1].split('\t')
            start, end, ltype = int(left), int(right), int(ltype)
            if start in node2id:
                type_corners[ltype][end].add(node2id[start])
            if end in node2id:
                type_corners[ltype][start].add(node2id[end])

    for ltype in ltypes:
        if int(ltype[0]) == target_node:
            useful_types.append(int(ltype[2]))
    for ltype in useful_types:
        corners = type_corners[ltype]
        two_hops = defaultdict(set)
        graph = nx.Graph(node_type=int)
        for _, neighbors in corners.items():
            for snode in neighbors:
                for enode in neighbors:
                    if snode != enode:
                        graph.add_edge(snode, enode)
        for node in node2id.values():
            if node not in graph:
                graph.add_edge(node, node)
        print(f'write graph {ltype},node:{len(graph.nodes)},edge:{len(graph.edges)}')
        nx.write_edgelist(graph, path=f"{model_data_folder}/sub_graph_{ltype}.edgelist",delimiter='\t', data=False)
        nx.draw_networkx(graph)
        show()
        a = 1

    for ltype in ltypes:
        snode,enode,l_type=[int(i) for i in ltype]
        if snode==target_node and enode==target_node:
            graph=nx.Graph(node_type=int)
            corners = type_corners[l_type]
            for origin_node, neighbors in corners.items():
                new_node_id=node2id[origin_node]
                for nei in neighbors:
                    graph.add_edge(new_node_id, nei)
            for node in node2id.values():
                if node not in graph:
                    graph.add_edge(node, node)
            print(f'write graph origin,node:{len(graph.nodes)},edge:{len(graph.edges)}')
            nx.write_edgelist(graph, path=f"{model_data_folder}/sub_graph_origin.edgelist", delimiter='\t', data=False)

    with open(f"{model_data_folder}/node2id_se.txt", 'w+') as f:
        for node, id in node2id.items():
            f.write('\t'.join([str(node), str(id)])+'\n')







