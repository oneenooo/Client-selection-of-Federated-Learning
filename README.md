About

  To avoid pressure on the bandwidth during each round of federated learning training, only a portion of the clients are usually selected for training, and the clients themselves have different value and different local data types, so the client selection strategy directly affects the effectiveness and convergence speed of the local model. In this paper, we study the client selection problem with differences in client value. Define the contribution of a single client as the influence on the global model accuracy, and propose a client selection algorithm (CBE3) based on the contribution. It updates the clients selection weights by estimating the expectation of the impact of different clients on the accuracy of the system global model. 

Dependencies
   
   Our algorithm CBE3 is based on the federal learning framework Flsim. FLSim uses Anaconda to manage Python and it's dependencies, listed in environment.yml. To install the fl-py37 Python environment, set up Anaconda (or Miniconda), then download the environment dependencies with:
    
 	           conda env create -f environment.yml
			   
Algorithm ideas

Here,define the improvement in the accuracy of the global model as the contribution of client in the current communication round. We design an offline selection model for the federated learning client selection system, which selects the set of clients with the largest amount of contribution to the global model.
Although the form of client selection may be simple, it cannot be solved by offline optimization methods because the amount of current clients' contribution to the global model cannot be obtained before the selection process. Therefore, we use an adaptive learning solution, the Multi-armed Bandit (MAB), to achieve an accurate estimation of the contribution amount.
The offline selection problem is transformed into an online learning scheduling problem based on the MAB. The client selection problem is also transformed into a probabilistic assignment problem for clients, and the optimization object is shifted from client selection to probabilistic assignment. Maximizing the reward expectation is achieved by optimizing the probability assignment and the random selection under this probability assignment.

The algorithm flow is as follows

Initialization:
initialize the parameters of the global model and initialize the selection weights of all clients to 1, i.e., ensure that all clients in the first round of training have the same selection probability.

Iteration process:

first calculate the selection probability of clients according to the selection weights; then randomly select clients according to the selection probability using the DepRound algorithm, and perform the local model training of clients and the aggregation of the global model, and record the contribution of clients; finally make unbiased estimation of the contribution of clients in the next round according to the contribution of clients in the current round and update the selection weights; keep iterating the above process until the end of training.

Results

Experiments are conducted based on the public dataset CIFAR-10 with multiple client selection algorithms for comparison. The experimental results show that the method in this paper can effectively improve the convergence speed of the global model while ensuring the effect of the global model. 

NOTE:

This work is developed by the Lab of Professor Weiwei Lin (linww@scut.edu.cn), School of Computer Science and Engineering, South China University of Technology.

Authors:

LinWeiwei email:147868463@qq.com；

XuYinhai email:xu135845935845@163.com;

HuangTiansheng email:873966702@qq.com;

