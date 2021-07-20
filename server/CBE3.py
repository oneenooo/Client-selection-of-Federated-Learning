import logging
import pickle
from copy import deepcopy

from server import volatile
import numpy as np
from threading import Thread
import random
import sys
import math
import utils.dists as dists  # pylint: disable=no-name-in-module
from sklearn.cluster import KMeans

class CBE3Server(volatile.Volatile):
    """Federated learning server that uses profiles to direct during selection."""
    # Run federated learning
    def run(self):
        self.success_temp=0
        self.selection_temp=0
        self.initialize_bandit()
        # print("clients{}".format(self.clients))
        # Continue federated learning
        super().run()

    def initialize_bandit(self):
        # set up bandit parameter
        self.K = self.config.clients.total
        self.k = self.config.clients.per_round
        self.T = self.config.fl.rounds
        # group fairness degree
        if self.config.bandit.sigma_ratio=="increment":
            self.sigma =0 * self.k / self.K
        else:
            self.sigma = self.config.bandit.sigma_ratio* self.k / self.K
        # learning rate
        # self.eta = math.sqrt(self.K * math.log(self.K / self.k) / (self.T * (self.sigma + self.k - self.N * self.sigma)))
        self.eta=0.6
        # print("eta{}".format(self.eta))
        self.bandit_w = [np.ones(self.K)]
        # self.window_size=400
        self.window_size = 100
        logging.info('bandit sigma ratio: {}'.format(self.config.bandit.sigma_ratio))

    # Federated learning phases
    def selection(self):
        prob_vec=self.probability_alloc_new()
        sample_clients=self.dep_round(prob_vec)
        return sample_clients

    def probability_alloc(self):
        w_array=self.bandit_w[-1]
        # print(w_array)
        p=np.zeros(self.K)
        if np.max(w_array)/np.sum(w_array) >1:
            sort_w=np.sort(w_array)
            for case,w in enumerate(sort_w[:-1],start=1):
                alpha= np.sum(sort_w[:case])/(self.k-self.K*self.sigma- (self.K-case)*(1-self.sigma))
                if sort_w[case-1]/(1-self.sigma) <= alpha< sort_w[case]/(1-self.sigma):
                    break
            self.truncate=[i for i, value in enumerate(w_array) if  value >alpha]
            print(alpha)
            revise_w_array=np.array([ alpha if w> alpha else w for w in w_array])
            # print(revise_w_array)
            p=revise_w_array/np.sum(revise_w_array)*self.k
        else:
            self.truncate =[]
            p =w_array / np.sum(w_array)*self.k
        # record the current probability allocation
        self.p=p
       # print(self.k)
        # print()
        print("p:{}".format(p))
        # print("sum_p:{}".format(np.sum(p)))
        # print("q:{}".format(q))
        return p

    def probability_alloc_new(self):
        w_array=self.bandit_w[-1]
        # print(w_array)
        p = np.zeros(self.K)
        self.truncate=[]

        for i in range(self.K):
            if w_array[i]*self.k/np.sum(w_array)>0.8:
               w_array[i]=0.8*np.sum(w_array)/self.k
               self.truncate.append(i)
        p =w_array*self.k / np.sum(w_array)
        # record the current probability allocation
        self.p=p
        #print(self.k)
        # print()
        print("p:{}".format(p))
        # print("sum_p:{}".format(np.sum(p)))
        # print("q:{}".format(q))
        return p

    def dep_round(self, prob_vec):
        temp=deepcopy(prob_vec)
        # print(temp)
        # record=[]
        # print(np.sum(prob_vec))
        assert np.sum(prob_vec)-1e-5< 20 or np.sum(prob_vec)+1e-5>20
        while len(list(filter( lambda x: 0 < x < 1, temp)))>0:

            non_round=list(filter(lambda x: 0 < x < 1, temp))
            if len(non_round)==1 :
                # print(temp)
                # print("precision problem: {}".format(non_round[0]))
                temp=[ round(value,0) for value in temp]
                continue
            # print(temp)
            position=[ i for i,p in enumerate(temp) if 0<p<1]
            # print(prob_vec)
            # print(position)
            i=position[0]
            j= position[1]
            # choose the first and second raw_prob
            a=min(1-temp[i],temp[j])
            b=min(temp[i],1-temp[j])
            if np.random.random() < a / (a + b):
                temp[i] = temp[i] - b
                temp[j] = temp[j] + b
            else:
                temp[i]= temp[i]+a
                temp[j]= temp[j]-a
            # print(temp)
            # record.append(temp)
        select_index={i for i, prob in enumerate(temp) if 1-1e-3<prob<1+1e-3}
        # print(temp)
        # print(np.sum(temp))
        # print(select_index)
        selection= [client  for client in self.get_all_clients()  if client.client_id in select_index ]
        # print(len(selection))
        impossible=[ i for i,p in  enumerate(prob_vec) if p< 1e-10 ]
        for index in select_index:
            # print(impossible)
            assert  index not in impossible
        assert len(selection)==10
        return selection

    def do_update_new(self, sample_clients, success_clients, jdelt_accuracy):

        # 更新方式标记
        print("jdelt_accuracy  coefficient is 150")

        # update bandit weight
        selection_flags = np.zeros(self.K)
        selection_flags[[client.client_id for client in sample_clients]] = 1
        success_flags = np.zeros(self.K)
        success_flags[[client.client_id for client in success_clients]] = 1
        # self.success_temp += success_flags[-1]
        # self.selection_temp += selection_flags[-1]
        # print("sample client:{}".format(sample_clients))
        # print("success client:{}".format(success_clients))
        # print("success number:{}".format(self.success_temp))
        # print("selection number:{}".format(self.selection_temp))
        # print("estimated rate:{}".format(self.success_temp/self.selection_temp))
        p = self.p
        unbiased_x = np.zeros(len(success_flags))
        # for i in range(len(success_flags)):
        for i in range(len(selection_flags)):
            if selection_flags[i] == 1:
                unbiased_x[i] = selection_flags[i] * jdelt_accuracy[i]/ p[i]
                # prevent accuaracy overflow unbiased_x should be limited to [-200,1]
                unbiased_x[i] = max(unbiased_x[i], -1)
                unbiased_x[i] = min(unbiased_x[i], 1)
            else:
                unbiased_x[i] = 0
        new_weights = deepcopy(self.bandit_w[-1])
        # print(self.truncate)
        # print("sample_clients:{}".format(selection_flags[-3]))
        # print("success_clients:{}".format(success_flags[-3]))
        # print(self.p)
        # print(unbiased_x)
        for i in range(self.K):
            if i not in self.truncate:
                new_weights[i] = new_weights[i] \
                                         * np.exp(self.eta * unbiased_x[i] )

        #     if i not in self.truncate:
        #         if success_flags[i] == 1:
        #             new_weights[i] = (1 + delt_accuracy * 10) * new_weights[i] \
        #                              * np.exp(self.eta * unbiased_x[i])
        #         else:
        #             new_weights[i] = new_weights[i] \
        #                              * np.exp(self.eta * unbiased_x[i])

        # new_weights[i]=max(new_weights[i],1e-20)
        # else:
        #     print("truncate{}".format(i))
        # print("unbiased_x:{}".format(unbiased_x)  )
        # print("weights:{}".format(new_weights))
        self.bandit_w.append(new_weights)