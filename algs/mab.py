import networkx as nx
import random as rd
from sklearn.cluster import KMeans
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV
from sklearn.decomposition import PCA

import matplotlib.pyplot as plt
from utils.eval import compute_unfairness
import numpy as np

from tqdm import tqdm
import algs.MaxFair as mf


class MAB_LP():

    def __init__(self, G=None, communities=None, add_radio=0.1, batch_step=1, acc_bound = None, fairness_bound = None, gamma=1, slot_number = 20, e = 0.3, decay = 0.8):
        self.G = G
        self.G_ground = G
        self.communities = communities
        self.slots_benefit = [0 for i in range(slot_number)]
        self.max_customize_benefit = -100
        self.max_customize_benefit_index = 0
        self.add_radio = add_radio
        self.batch_step = batch_step
        # used to called gamma as paramater self.parameter = parameter
        self.gamma = gamma
        self.slot_number = slot_number
        self.slots = [[] for i in range(slot_number)]
        self.e = e
        self.decay = decay
        self.slot_number = slot_number
        self.MAB_precision = []
        self.MAB_modularity_improve = []
        self.MAB_IU_improve = []
        self.acc_bound = acc_bound
        self.fairness_bound = fairness_bound
        self.clear_pre_history()



    def fit(self, x_train, y_train):
        # prepare original network
        for i in range(len(self.communities)):
            for node in self.communities[i]:
                self.G.nodes[node]['att'] = i
                self.G_ground.nodes[node]['att'] = i

        self.modular_ground = nx.community.modularity(self.G, self.communities)
        self.unfairness_ground = compute_unfairness(self.G,self.communities,output_detail=0)

        ttt = 0
        for edge in x_train:
            if edge[2] == 1:
                ttt+=1
                self.G.add_edge(edge[0], edge[1])
                self.G_ground.add_edge(edge[0], edge[1])

    def predict(self, test_data):
        self.test_data = test_data
        self.generate_slot_k_mean_inter_intra(self.slot_number)
        self.pull(int(len([i for i in self.test_data if i[2]==1]) * self.add_radio), self.batch_step)
        return 0

    def score(self, X, y, sample_weight=None):
        self.test_data = X
        self.generate_slot_k_mean_inter_intra(self.slot_number)
        self.pull(int(len([i for i in self.test_data if i[2]==1]) * self.add_radio), self.batch_step)
        # score
        unfairness_new = self.get_new_information_unfairness()
        unfairness = np.round((self.unfairness_ground- unfairness_new)/self.unfairness_ground, 4)
        acc = self.eval_accuracy()
        print("acc", acc)
        if self.acc_bound != None:
            if acc < self.acc_bound:
                print('accuracy is lower than tolerance')
                print('Penalty: score set to', float('-1'))
                return float('-1')
            else:
                print('score', unfairness)
                return unfairness
        print('score', unfairness)
        return unfairness



    # 注意这里的参数，要把全部的都取下来，因为后面是要根据真的return去set新的model
    # 不是传入的参数他是不会变的，不会被上一轮影响，应该是重新建了一个model，然后用
    # set设置了需要传入的参数
    def get_params(self, deep=True):
        """Get parameters for this estimator.

        Parameters
        ----------
        deep : boolean, optional
            If True, will return the parameters for this estimator and
            contained subobjects that are estimators.

        Returns
        -------
        params : mapping of string to any
            Parameter names mapped to their values.
        """
        out = dict()
        for key in ['gamma','slot_number','e','decay','G','communities', 'add_radio',
         'batch_step','acc_bound', 'fairness_bound']:#hyper-parameter list
            value = getattr(self, key, None)
            if deep and hasattr(value, 'get_params'):
                deep_items = value.get_params().items()
                out.update((key + '__' + k, val) for k, val in deep_items)
            out[key] = value
        return out

    def set_params(self, **params):
        """Set the parameters of this estimator.

        The method works on simple estimators as well as on nested objects
        (such as pipelines). The latter have parameters of the form
        ``<component>__<parameter>`` so that it's possible to update each
        component of a nested object.

        Returns
        -------
        self
        """
        if not params:
            # Simple optimization to gain speed (inspect is slow)
            return self
        valid_params = self.get_params(deep=True)


        for key, value in params.items():
            if key not in valid_params:
                raise ValueError('Invalid parameter %s for estimator %s. '
                                 'Check the list of available parameters '
                                 'with `estimator.get_params().keys()`.' %
                                 (key, self))
            setattr(self, key, value)
            valid_params[key] = value

        return self
        
    def generate_slot_k_mean_inter_intra(self, slot_number):
        # use k mean way to seperate the slot
        # add resource allocation feature
        preds = nx.resource_allocation_index(self.G, [(edge[0], edge[1]) for edge in self.test_data])
        # add attribute related feature -- the diversity of the neighbors
        maxfair_feature = mf.compute_score_MaxFair(self.G, 3, 0.5, [(edge[0], edge[1]) for edge in self.test_data])
        inter_intra_feature = []
        for edge in self.test_data:
            if self.G.nodes[edge[0]]['att'] == self.G.nodes[edge[1]]['att']:
                inter_intra_feature.append(0)
            else:
                inter_intra_feature.append(1)
        diversity_feature = []
        for edge in self.test_data:
            common_neighbors = nx.common_neighbors(self.G, edge[0], edge[1])
            number_male = 0
            number_female = 0
            for node in common_neighbors:
                if self.G.nodes[node]['att'] == 0:
                    number_male+=1
                elif self.G.nodes[node]['att'] == 1:
                    number_female +=1
            if max(number_male, number_female) == 0:
                diversity_feature.append(0)
            else:
                diversity_feature.append(min(number_male, number_female)/max(number_male, number_female))
        y = []
        k_mean_feature = []
        for i, (u, v, p) in enumerate(preds):
            y.append((u,v,p,inter_intra_feature[i],self.test_data[i][2]))
            k_mean_feature.append([p, inter_intra_feature[i]])
        clf = KMeans(n_clusters=slot_number)
        clf.fit(k_mean_feature)
        labels = clf.labels_
        for k, v in enumerate(labels):
            self.slots[v].append(y[k])
        # remove less step size use for batch processing
        size_of_step = self.batch_step
        while(1):
            iter = 0
            while(iter<len(self.slots)):
                if len(self.slots[iter]) < size_of_step:
                    self.clear_empty_slot(iter)
                    break
                else:
                    iter+=1
            if iter == len(self.slots):
                break
        # print("Generate slot machines succeed!\n")
        return 0

    def cal_customize_benefit(self, fairness_rate, accuracy_rate):
        self.max_customize_benefit = -100
        self.max_customize_benefit_index = 0
        for i in range(self.slot_number):
            self.update_max_customize_benefit(i, fairness_rate, accuracy_rate)

    def update_max_customize_benefit(self, selected_slot, fairness_rate, accuracy_rate):
        accur_benefit = 0
        if self.number_pred[selected_slot] != 0:
            accur_benefit = self.number_correct_pred[selected_slot] / self.number_pred[selected_slot]
        customize_benefit = fairness_rate * self.slots_benefit[selected_slot] + accuracy_rate * accur_benefit
        if customize_benefit > self.max_customize_benefit:
            self.max_customize_benefit = customize_benefit
            self.max_customize_benefit_index = selected_slot

    def pull(self, number_of_edges, size_of_step=10):
        for i in tqdm(range(int(number_of_edges/size_of_step)),position=0, leave=True):
            # e-greedy algorithm
            if (rd.random() < self.e): # explore
                selected_slot = rd.randint(0, self.slot_number-1)
                selected_edge_index = list(np.random.choice(len(self.slots[selected_slot]),size_of_step,replace=False))
                selected_edge = []
                for index in selected_edge_index:
                    selected_edge.append(self.slots[selected_slot][index])
            else: # exploit
                # instead of finding the max benefit, find the max customized benefit
                selected_slot = self.max_customize_benefit_index
                selected_edge_index = list(np.random.choice(len(self.slots[selected_slot]),size_of_step,replace=False))
                selected_edge = []
                for index in selected_edge_index:
                    selected_edge.append(self.slots[selected_slot][index])
            for edge in selected_edge:
                self.y_pred.append(edge)
            self.total_pred += size_of_step
            self.number_pred[selected_slot] += size_of_step
            # calculate the benefit
            correct_edges = []
            for edge in selected_edge:
                if edge[-1] == 1:
                    self.total_correct_pred += 1
                    self.number_correct_pred[selected_slot] += 1
                    correct_edges.append((edge[0],edge[1]))
            if len(correct_edges) != 0:
                benefit = self.cal_benefit(correct_edges)
                self.G.add_edges_from(correct_edges)
            else:
                benefit = 0
            # update the benefit
            # decay function
            self.slots_benefit[selected_slot] = self.slots_benefit[selected_slot] * self.decay + benefit * (1-self.decay)
            self.cal_customize_benefit(self.gamma, 1-self.gamma)
            # remove the selected_edge from slot
            for edge in selected_edge:
                self.slots[selected_slot].remove(edge)
            if len(self.slots[selected_slot]) < size_of_step:
                self.clear_empty_slot(selected_slot)
            unfairness_MAB = self.get_new_information_unfairness()
            IU_MAB_improve = np.round((self.unfairness_ground- unfairness_MAB)/self.unfairness_ground, 4)
            self.MAB_IU_improve.append(IU_MAB_improve)
            self.MAB_precision.append(self.eval_accuracy())
        return self.y_pred

    def get_MAB_IU_improve(self):
        return self.MAB_IU_improve

    def get_MAB_precision(self):
        return self.MAB_precision

    # clear empty slot, if index equals -1, check all the slots
    def clear_empty_slot(self, selected_slot):
        self.slots.pop(selected_slot)
        self.slots_benefit.pop(selected_slot)
        self.number_correct_pred.pop(selected_slot)
        self.number_pred.pop(selected_slot)
        self.slot_number -= 1
        self.cal_customize_benefit(self.gamma,1-self.gamma)

    def clear_pre_history(self):
        self.y_pred = []
        self.total_correct_pred = 0
        self.number_correct_pred = [0 for i in range(self.slot_number)]
        self.total_pred = 0
        self.number_pred = [0 for i in range(self.slot_number)]

    def eval_accuracy(self):
        return self.total_correct_pred/self.total_pred

    def inspect(self):
        for i in range(self.slot_number):
            print("slot", i, "benefit:", self.slots_benefit[i])
            print("slot", i, "accuracy:", self.number_correct_pred[i]/self.number_pred[i])
            print("slot", i, "has been selected", self.number_pred[i], "times")
            print("\n")

    def get_information_unfairness(self):
        unfairness_new = compute_unfairness(self.G, self.communities, output_detail=0)
        G_ground = self.G_ground.copy()
        for edge in self.test_data:
            if edge[2] == 1:
                G_ground.add_edge(edge[0], edge[1])

        unfairness_ground = compute_unfairness(G_ground, self.communities, output_detail=0)
        return unfairness_new, unfairness_ground

    def get_new_information_unfairness(self):
        unfairness_new = compute_unfairness(self.G, self.communities, output_detail=0)
        return unfairness_new

    def cal_benefit(self, edge_list):
        G_ground = self.G.copy()
        G_new = self.G.copy()
        G_new.add_edges_from(edge_list)
        unfairness_new = compute_unfairness(G_new,self.communities)
        unfairness_ground = compute_unfairness(G_ground,self.communities)
        return (unfairness_ground - unfairness_new)/unfairness_ground
