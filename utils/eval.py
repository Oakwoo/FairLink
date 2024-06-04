from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score

import algs.MaxFair as mf
import numpy as np
import networkx as nx

import matplotlib.pyplot as plt
import statistics as st

def get_accuracy_scores(grt, y_pred, median = 0.5): #  其实如果从上往下取，median没关系
    # check if -1 is used for  negatives label instead of 0.
    idx = grt[:,2]==-1
    grt[idx,2] = 0
    # create the acctual labels vector
    y_true = grt[:,2]
    # calcualte the accuracy
    roc_score = roc_auc_score(y_true, y_pred)
    ap_score = average_precision_score(y_true, y_pred)
    acc = accuracy_score(y_true, y_pred > median)
    return acc, roc_score,  ap_score

def ComputeDistributionBasedCommunities(nodes, communities,A, node_list):
    #flowing from group1 to group1
    g1g1=[]
    # flowing from group1 to group2
    g1g2 = []
    # flowing from group1 to group2
    g2g2 = []
    # only consider the main two big community
    for i in range(len(A)):
        u = node_list[i]
        for j in range(len(A)):
            if i != j:
                v = node_list[j]
                if u in communities[0] and v in communities[0]:
                    g1g1.append(A[i][j])
                elif u in communities[1] and v in communities[1]:
                    g2g2.append(A[i][j])
                elif (u in communities[0] and v in communities[1]) or (
                        u in communities[1] and v in communities[0]):
                    g1g2.append(A[i][j])
    return [g1g1, g1g2, g2g2]


def compute_unfairness(G, communities, k=3, p=0.5, output_detail=0):
    M = nx.adjacency_matrix(G).toarray()
    A = mf.AM_walk(M, k, p)
    nodes = G.nodes()
    node_list = list(nodes)
    D_fg_list = ComputeDistributionBasedCommunities(nodes, communities,A, node_list)
    g1g1_mean = np.mean(D_fg_list[0])
    g1g2_mean = np.mean(D_fg_list[1])
    g2g2_mean = np.mean(D_fg_list[2])
    if output_detail != 0:
        print(g1g1_mean, g1g2_mean, g2g2_mean)
    return np.max([abs(g1g1_mean - g1g2_mean), abs(g1g1_mean - g2g2_mean), abs(g1g2_mean - g2g2_mean)])/np.max([g1g1_mean, g1g2_mean, g2g2_mean])

def get_information_unfairness(G,y_pred,communities , grt, median =0.5, add_part=float("inf")):
    G_ground = G.copy()
    G_new = G.copy()
    for i in range(grt.shape[0]):
        if grt[i][2]==1:
            G_ground.add_edge(grt[i][0],grt[i][1])
    count = 0
    for i in range(grt.shape[0]):
        if y_pred[i] > median:
            G_new.add_edge(grt[i][0],grt[i][1])
            count+=1
        if count >= add_part:
            break
    print("New number of edge:", G_new.number_of_edges())
    print("Ground number of edge:", G_ground.number_of_edges())
    unfairness_new = compute_unfairness(G_new,communities,output_detail=1)
    unfairness_ground = compute_unfairness(G_ground,communities,output_detail=1)
    return unfairness_new, unfairness_ground

def compute_unfairness_multi_group(G, communities, k=3, p=0.5, output_detail=0):
    M = nx.adjacency_matrix(G).toarray()
    A = mf.AM_walk(M, k, p)
    nodes = G.nodes()
    node_list = list(nodes)
    # mention communities is not class_list, different
    attribute_map = nx.get_node_attributes(G, 'att')
    class_list = list(set(list(attribute_map.values())))

    D_fg_list = mf.ComputeDistribution_multi_group(nodes, class_list,A, node_list)
    mean_distribution_list = [st.mean(item) for item in D_fg_list if len(item)>0]

    if output_detail != 0:
        print(mean_distribution_list)
    distance_between_each_group = []
    for i in range(len(mean_distribution_list)):
        for j in range(i+1, len(mean_distribution_list)):
            distance_between_each_group.append(abs(mean_distribution_list[i]-mean_distribution_list[j]))
    return np.max(distance_between_each_group)/np.max(mean_distribution_list)