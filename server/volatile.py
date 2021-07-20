import numpy

import client
import load_data
import logging
import numpy as np
import pickle
import random
import sys
from threading import Thread
import torch
import utils.dists as dists  # pylint: disable=no-name-in-module
from server import Server
import matplotlib.pyplot as plt
import numpy


class Volatile(Server):
    """Basic federated learning server."""

    def boot(self):
        super().boot()
        self.total_success=0
        self.select_record=np.zeros(self.config.clients.total)

    def initialize_success_data(self):
        np.random.seed(3)
        clients= self.get_all_clients()
        rounds = self.config.fl.rounds
        self.success_data = np.zeros([len(clients), rounds])
        for round in range(rounds):
            # if round%500==0:
            #     parameters=np.random.random(len(clients))
            for client in clients:
                self.success_data[client.client_id][round]= np.random.binomial(1,client.success_rate )

    def run(self):
        accuracy_record=[]
        rounds = self.config.fl.rounds
        target_accuracy = self.config.fl.target_accuracy
        reports_path = self.config.paths.reports
        self.initialize_success_data()
        if target_accuracy:
            logging.info('Training: {} rounds or {}% accuracy\n'.format(
                rounds, 100 * target_accuracy))
        else:
            logging.info('Training: {} rounds\n'.format(rounds))

        print("this train the type of data: {}".format(self.config.data.IID))
        print("this train the model: {}".format(self.config.model))
        deltaccuracy=[]
        accuracy0 = []
        round0 = []
        accuracy = 0
        delt_accuracy=0
        # Perform rounds of federated learning
        for round in range(1, rounds + 1):
            logging.info('**** Round {}/{} ****'.format(round, rounds))

            deltaccuracy.append(delt_accuracy)
            accuracy,delt_accuracy= self.round(round,accuracy)

            print("the {} round increase {}% accuracy".format(round,100*delt_accuracy))
            print("the {} round achieve {}% accuracy".format(round, 100 * accuracy))
            print("the totall {} round choose time of each of arm".format(round))
            print(self.select_record)

            accuracy0.append(accuracy)
            round0.append(round)

            if target_accuracy and (accuracy >= target_accuracy):
                logging.info('Target accuracy reached.')
                break
            accuracy_record.append(accuracy)

            if round % 10 == 0:
                numpy.savetxt("", accuracy0)

        print(accuracy0)
        print(deltaccuracy)

        # write accuracy record
        with open("result/"+self.config.model+"/accuracy_record_"+self.config.algorithm+"_"+str(self.config.clients.per_round)+"_"+ str(self.config.data.IID)+".pkl", 'wb') as f:
            pickle.dump(accuracy_record,f)
        # for client in self.get_all_clients():
        #     print("{}:{}".format(client.client_id,client.success_rate))
        if reports_path:
            with open(reports_path, 'wb') as f:
                pickle.dump(self.saved_reports, f)
            logging.info('Saved reports: {}'.format(reports_path))
        print([(client.client_id,client.success_rate) for client  in self.get_all_clients()])
        print(self.select_record)
        print(np.sum(self.select_record))
        print("total success{}".format(self.total_success))


    def get_all_clients(self):
        return self.clients

    def do_update(self,select_client, success_client ):
        # not implemented
        return

    def make_clients(self, num_clients):
        IID = self.config.data.IID
        labels = self.loader.labels
        loader = self.config.loader
        loading = self.config.data.loading
        random.seed(1)
        if not IID:  # Create distribution for label preferences if non-IID
            dist = {
                "uniform": dists.uniform(num_clients, len(labels)),
                "normal": dists.normal(num_clients, len(labels))
            }[self.config.clients.label_distribution]
            random.shuffle(dist)  # Shuffle distribution

        # Make simulated clients
        clients = []
        for client_id in range(num_clients):

            # Create new client revised by hts
            success_parameters = [1, 1, 1 ]
            #success_parameters=[0.1,0.5,0.9]

            new_client = client.volatile_client(client_id,success_parameters[int(client_id%len(success_parameters))])

            if not IID:  # Configure clients for non-IID data
                if self.config.data.bias:
                    # Bias data partitions
                    bias = self.config.data.bias
                    # Choose weighted random preference
                    pref = random.choices(labels, dist)[0]

                    # Assign preference, bias config
                    new_client.set_bias(pref, bias)
                elif self.config.data.shard:
                    # Shard data partitions
                    shard = self.config.data.shard

                    # Assign shard config
                    new_client.set_shard(shard)

            clients.append(new_client)

        logging.info('Total clients: {}'.format(len(clients)))

        if loader == 'bias':
            logging.info('Label distribution: {}'.format(
                [[client.pref for client in clients].count(label) for label in labels]))

        if loading == 'static':
            if loader == 'shard':  # Create data shards
                self.loader.create_shards()

            # Send data partition to all clients
            [self.set_client_data(client) for client in clients]

        self.clients = clients

    def success_clients(self,sample_clients,round):
        success_clients=[]
        for client in sample_clients:
            if self.success_data[client.client_id][round-1]==1:
                success_clients.append(client)
        return success_clients


    def round(self,round_index,accuracy):
        import fl_model  # pylint: disable=import-error
        round_delt_accuracy=0
        jdelt_accuracy = np.zeros(self.K)

        print("\n\n")
        print("***Round {}/{}***".format(round_index,self.config.fl.rounds))

        if round_index==1:
            # Test global model accuracy
            if self.config.clients.do_test:  # Get average accuracy from client reports
                new_accuracy = self.accuracy_averaging(reports)
            else:  # Test updated model on server
                if not self.config.fl.skip:
                    testset = self.loader.get_testset()
                    batch_size = self.config.fl.batch_size
                    testloader = fl_model.get_testloader(testset, batch_size)
                    new_accuracy = fl_model.test(self.model.to(self.config.fl.device), testloader,
                                                 self.config.fl.device)
                else:
                    new_accuracy = 0
            logging.info('Average accuracy: {:.2f}%\n'.format(100 * new_accuracy))
            accuracy=new_accuracy

        # Select clients to participate in the round
        sample_clients = self.selection()
        a = [client.client_id for client in sample_clients]
        print("the {} round select the number of client {}".format(round_index,a))
        success_clients = self.success_clients(sample_clients, round_index)
        #for client in sample_clients:
        for single_sample_clients in sample_clients:
            single_sample_client=[single_sample_clients]
            self.select_record[single_sample_clients.client_id]+=1
            # Configure sample clients
            self.configuration(single_sample_client)
            # determine the success flag of training
            single_success_clients=self.success_clients(single_sample_client,round_index)
            self.total_success+=len(single_success_clients)
            # Run clients using multithreading for better parallelism
            if not self.config.fl.skip:
                if self.config.fl.device=="cpu":
                    threads = [Thread(target=client.run) for client in single_sample_client]
                    [t.start() for t in threads]
                    [t.join() for t in threads]
                else:
                    for client in single_sample_client:
                        client.train(self.config.fl.device)
                # Recieve client updates
                reports = self.reporting(single_success_clients)
            else:
                reports=[]

            # Perform weight aggregation
            logging.info('Aggregating updates')
            if reports :
                updated_weights = self.aggregation(reports)
                # Load updated weights
                fl_model.load_weights(self.model, updated_weights)
                # Extract flattened weights (if applicable)
                if self.config.paths.reports:
                    self.save_reports(round, reports)

            # Save updated global model
            self.save_model(self.model, self.config.paths.model)

            # for client in self.clients:
            #     self.save_model_client(self.model, self.config.paths.model, client)

            # Test global model accuracy
            if self.config.clients.do_test:  # Get average accuracy from client reports
                new_accuracy = self.accuracy_averaging(reports)
            else:  # Test updated model on server
                if not self.config.fl.skip:
                    testset = self.loader.get_testset()
                    batch_size = self.config.fl.batch_size
                    testloader = fl_model.get_testloader(testset, batch_size)
                    new_accuracy = fl_model.test(self.model.to(self.config.fl.device), testloader,self.config.fl.device)
                else:
                    new_accuracy =0
            logging.info('Average accuracy: {:.2f}%'.format(100 * new_accuracy))

            delt_accuracy=new_accuracy-accuracy
            if delt_accuracy==0:
                print("the {} round  the {} client accuracy is not changed".format(round_index,single_sample_clients.client_id))
            accuracy=new_accuracy
            print("each round single client {} make mode delt_accuracy {}".format(single_sample_clients.client_id,delt_accuracy),'\n')


            jdelt_accuracy[single_sample_clients.client_id]=delt_accuracy

            #self.do_update(sample_clients, success_clients)

            # print(delt_accuracy)
            round_delt_accuracy +=delt_accuracy

        print("each {} round  make mode delt_accuracy {}".format(round_index,jdelt_accuracy))
        self.do_update_new(sample_clients, success_clients, jdelt_accuracy)

        return accuracy,round_delt_accuracy