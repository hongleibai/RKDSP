import numpy as np
from collections import defaultdict
import networkx as nx

data_folder, model_folder = '../Data', '../Model'
node_file, link_file, label_file = 'node.dat', 'link.dat', 'label.dat'
info_file, meta_file = 'info.dat', 'meta.dat'

ori_data_folder = f'{data_folder}/ACM'  # Data/ACM
model_data_folder = f'{model_folder}/CKD/data/ACM/link'  # Model/CKD/data/ACM/
dataset = 'ACM'
version = 'link'

def add_node(neigh_map, start_node, end_node, type_map, all_node_types):
    if start_node not in neigh_map:
        neigh_map[start_node] = {}
        for node_type in all_node_types:
            neigh_map[start_node][node_type] = set()
    if end_node not in neigh_map:
        neigh_map[end_node] = {}
        for node_type in all_node_types:
            neigh_map[end_node][node_type] = set()
    neigh_map[start_node][type_map[end_node]].add(end_node)
    neigh_map[end_node][type_map[start_node]].add(start_node)


ori_data_folder = f'{data_folder}/{dataset}'  # Data/ACM
model_data_folder = f'{model_folder}/CKD/data/{dataset}/{version}'  # Model/CKD/data/ACM/

node_type_map = {}  # node->node_type
node_neigh_type_map = {}  # node->node_type->neigh_node
node_types = set()  # 节点类型的集合
target_node_set = set()  # 目标结点的集合
node2id = {}  # 目标节点转成新的idx
useful_types = []

print(f'CKD: writing {dataset}\'s config file!')
target_node, target_edge, ltypes = 0, 0, []
with open(f'{ori_data_folder}/{info_file}', 'r') as original_info_file:
    for line in original_info_file:
        if line.startswith('Targeting: Link Type'): target_edge = int(line[:-2].split(',')[-1])
        if line.startswith('Targeting: Label Type'): target_node = int(line.split(' ')[-1])
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
config_file = open(f'{model_data_folder}/config.dat', 'w')
config_file.write(f'{target_node}\n')
config_file.write(f'{target_edge}\n')
config_file.write('{}\n'.format('\t'.join(list(map(lambda x: ','.join(x), ltypes)))))
config_file.close()

print('CKD Link: converting {}\'s node file for {} training!'.format(dataset,
                                                                     'attributed' ))
new_node_file = open(f'{model_data_folder}/{node_file}', 'w')
with open(f'{ori_data_folder}/{node_file}', 'r') as original_node_file:
    for line in original_node_file:
        line = line[:-1].split('\t')
        new_node_file.write(f'{line[0]}\t{line[2]}\n')

        node_type_map[int(line[0])] = int(line[2])  # node type map 0 是节点类型0
        node_types.add(int(line[2]))
        if int(line[2]) == target_node:
            node2id[int(line[0])] = len(node2id)
            target_node_set.add(int(line[0]))
new_node_file.close()

print(f'CKD Link: converting {dataset}\'s label file')
new_label_file = open(f'{model_data_folder}/{label_file}', 'w')
with open(f'{ori_data_folder}/{label_file}', 'r') as original_label_file:
    for line in original_label_file:
        line = line[:-1].split('\t')
        new_label_file.write(f'{line[0]}\t{line[3]}\n')
new_label_file.close()

type_corners = {int(ltype[2]): defaultdict(set) for ltype in ltypes}

print(f'CKD: converting {dataset}\'s link file!')
new_link_file = open(f'{model_data_folder}/{link_file}', 'w')
with open(f'{ori_data_folder}/{link_file}', 'r') as original_link_file:
    for line in original_link_file:
        left, right, ltype, weight = line[:-1].split('\t')
        new_link_file.write(f'{left}\t{right}\t{ltype}\n')
        # add_node(node_neigh_type_map,int(left),int(right),node_type_map,node_types)
        # origin_graph.add_edge(int(left), int(right), weight=int(weight), ltype=int(ltype),
        #                direction=1 if left <= right else -1)
        start, end, ltype = int(left), int(right), int(ltype)
        if start in node2id:
            type_corners[ltype][end].add(node2id[start])
        if end in node2id:
            type_corners[ltype][start].add(node2id[end])
new_link_file.close()
# get homogeneous graph
for ltype in ltypes:
    if int(ltype[0]) == target_node or int(ltype[1]) == target_node:
        useful_types.append(int(ltype[2]))

for ltype in useful_types:
    # if dataset=='DBLP2' and ltype==2:
    #     continue
    corners = type_corners[ltype]
    # 根据同一个start node,从而判断节点之间的二阶关系
    two_hops = defaultdict(set)
    graph = nx.Graph(node_type=int)
    for _, neighbors in corners.items():
        # print(f'ltype:{ltype},node_cnt:{len(neighbors)}')
        for snode in neighbors:
            for enode in neighbors:
                if snode != enode:
                    # two_hops[snode].add(enode)
                    graph.add_edge(snode, enode)
    # 如果缺少边,则添加自环
    for node in node2id.values():
        if node not in graph:
            graph.add_edge(node, node)
    print(f'write graph {ltype},node:{len(graph.nodes)},edge:{len(graph.edges)}')
    nx.write_edgelist(graph, path=f"{model_data_folder}/sub_graph_{ltype}.edgelist", delimiter='\t', data=False)
    nx.draw_networkx(graph)
# 原始图的一阶关系
for ltype in ltypes:
    snode, enode, l_type = [int(i) for i in ltype]
    if snode == target_node and enode == target_node and l_type == target_edge:
        graph = nx.Graph(node_type=int)
        corners = type_corners[l_type]
        for origin_node, neighbors in corners.items():
            new_node_id = node2id[origin_node]
            for nei in neighbors:
                graph.add_edge(new_node_id, nei)
        for node in node2id.values():
            if node not in graph:
                graph.add_edge(node, node)
        print(f'write graph origin,node:{len(graph.nodes)},edge:{len(graph.edges)}')
        nx.write_edgelist(graph, path=f"{model_data_folder}/sub_graph_origin.edgelist", delimiter='\t', data=False)

# add node to new_id map file
with open(f"{model_data_folder}/node2id.txt", 'w') as f:
    for node, id in node2id.items():
        f.write('\t'.join([str(node), str(id)]) + '\n')





