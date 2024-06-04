import numpy as np
import networkx as nx
import statistics as st
import collections

def ComputeDistribution(nodes, groups,A, node_list):
    #flowing from group1 to group1
    g1g1=[]
    # flowing from group1 to group2
    g1g2 = []
    # flowing from group1 to group2
    g2g2 = []
    #as G is symmetric we consider the elements above the diagonal
    for i in range(len(A)):
        u = nodes[node_list[i]]
        for j in range(len(A)):
            if i != j:
                v = nodes[node_list[j]]
                if u['att'] == groups[0] and v['att'] == groups[0]:
                    g1g1.append(A[i][j])
                elif u['att'] == groups[1] and v['att'] == groups[1]:
                    g2g2.append(A[i][j])
                elif (u['att'] == groups[1] and v['att'] == groups[0]) or (
                        u['att'] == groups[0] and v['att'] == groups[1]):
                    g1g2.append(A[i][j])

    return [g1g1, g1g2, g2g2]

# generalized funtion for more than 2 groups 5/24/2022 Weixiang
def ComputeDistribution_multi_group(nodes, groups,A, node_list):
    joint_class_list = compute_joint_class_list(groups)
    joint_class_index_map = dict(zip(joint_class_list, range(len(joint_class_list))))
    g_i_g_j = [[] for i in range(len(joint_class_list))]

    #as G is symmetric we consider the elements above the diagonal
    for i in range(len(A)):
        u = nodes[node_list[i]]
        for j in range(len(A)):
            if i != j:
                v = nodes[node_list[j]]
                if (u['att'], v['att']) in joint_class_index_map.keys():
                    index = joint_class_index_map[(u['att'], v['att'])]
                else: # (v['att'], u['att']) in joint_class_index_map.keys():
                    index = joint_class_index_map[(v['att'], u['att'])]
                g_i_g_j[index].append(A[i][j])

    return g_i_g_j

#based on walks
def AM_walk(M, k, p=-1):

    if p != -1:
        PM = p * np.array(M)
        A= PM.copy()
        current =  PM.copy()
        for i in range(k - 1):
            current = np.matmul(PM, current)
            A += current
    else:
        PM = np.array(M)
        A = (1/2)*PM.copy()
        current = PM.copy()
        for i in range(k - 1):
            current = np.matmul(PM, current)
            A += (1/(i+3))*np.array(current)
    return A

def compute_joint_class_list(class_list, is_directed=False):
    output=[]
    for i in range(len(class_list)):
        class_i = class_list[i]
        for j in range(i, len(class_list)):
            class_j = class_list[j]
            output.append((class_i,class_j))
    return output


def ABC(G, k, p):
    att = nx.get_node_attributes(G, 'att')
    counts = collections.Counter(att.values())
    classes = list(set(att.values()))
    # 注意：下一行注释了 by Weixiang on 4/15/2022
    # classes =['male', 'female']
    C = np.zeros((len(classes), len(G)))
    C0 = np.zeros((len(classes), len(G)))
    nodes = list(G.nodes())
    for i in range(len(nodes)):
        u = nodes[i]
        a = att[u]
        if a!='None':
            c1 = classes.index(a)
            C[c1][i] = 1
            C0[c1][i] = 1

    vectors = [C]
    for i in range(k):
        C1 = np.zeros((len(classes), len(G)))
        for j in range(len(classes)):
            k1 = -1
            for u in nodes:
                k1 += 1
                count = 0
                neighs = G[u]
                for v in neighs:
                    ind = nodes.index(v)
                    count += vectors[-1][j][ind]
                C1[j][k1] = count
        vectors.append(C1)
    #print(vectors)

    for i in range(len(vectors)):
        vectors[i] = (p ** (i)) * vectors[i]
        # vectors[i] = (p**(i))* vectors[i]

    output = np.zeros((len(classes), len(G)))
    for i in range(len(vectors)):
        output += vectors[i]

    for i in range(len(classes)):
        output[i] = output[i] / counts[classes[i]]

    output = output + C0

    return output

def compute_score_MaxFair(G,k,p, candidates):
    M = nx.adjacency_matrix(G).toarray()  # adjacency matrix
   # node_info = G.nodes()
    node_list = list(G.nodes())
    node_index_map = dict(zip(node_list, range(len(node_list))))
    # 注意：next 3 lines changed by Weixiang on 4/15/2022
    attribute_map = nx.get_node_attributes(G, 'att')
    class_list = list(set(list(attribute_map.values())))
    # class_list =['male', 'female']
    class_index_map = dict(zip(class_list ,range(len(class_list))))
    A = AM_walk(M, k, p) # acccessibility matrix
    #C = compute_class_membership(class_list, attribute_map) # class memberhsip
    joint_class_list = compute_joint_class_list(class_list)
    joint_class_index_map = dict(zip(joint_class_list, range(len(joint_class_list))))
    # step 1) Compute the attribute-based centrality vector vecf for each attribute group Cf .
    # attribute-based centrality vectors (each vector corresponds to each joint class)
    #vec_f_list = AC.ABA3(A, C)
    #print(vec_f_list)
    vec_f_list = ABC(G,k=k, p=p)


    # Step 2) Compute the joint attribute accessibility distributions Df g for all group pairs Cf and Cg .
   # D_fg_list = ComputeDistribution(A, node_list, attribute_map, joint_class_index_map)
    nodes = G.nodes()
    node_list = list(nodes)
    D_fg_list = ComputeDistribution_multi_group(nodes, class_list, A, node_list)

    # Step 3) Compute the mean of each Dfg distribution, and the mean all mean of the distribution means. Let sfg = all mean − mean(Dfg).
    mean_distribution_list = [st.mean(item) for item in D_fg_list if len(item)>0]
    #print(mean_distribution_list)
    all_mean = st.mean(mean_distribution_list)
    #print(all_mean)
    S_fg_list = [all_mean - item for item in mean_distribution_list]

    # Step 4) Iterate over all pairs of nodes (u,v) that are not
    #          already connected in Gj . Define score(u, v) = 􏰀f,g sfg ∗ (vecf (u) ∗ vecg(v) + vecg(u) ∗ vecf (v)).
    score_map={}
    for (u,v) in candidates:
        score = 0
        for (class_i, class_j) in joint_class_list:
            S_ij = S_fg_list[joint_class_index_map[(class_i, class_j)]]
            index_i = class_index_map[class_i]
            index_j = class_index_map[class_j]
            vec_class_i = vec_f_list[index_i]
            vec_class_j = vec_f_list[index_j]
            score += S_ij*(vec_class_i[node_index_map[u]]*vec_class_j[node_index_map[v]] + vec_class_i[node_index_map[v]]*vec_class_j[node_index_map[u]])
        score_map[(u,v)] = score
    return score_map
