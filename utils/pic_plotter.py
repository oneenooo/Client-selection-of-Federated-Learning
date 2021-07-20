import pickle

import matplotlib.pyplot as plt
def plot_accuracy():
    dataset="CIFAR-10"
    algorithms=["bandit0_20_False","bandit0.5_20_False","bandit1_20_False","greedy_20_False","volatile_20_False"]
    # algorithms = ["volatile_5_True", "volatile_10_True", "volatile_20_True"]
    for algorithm in algorithms:
        with open("../result/"+dataset+"/accuracy_record_" + algorithm+ ".pkl",
                  'rb') as f:
            accuracy=pickle.load(f)
            plt.plot(range(len(accuracy)), accuracy, label="{}".format(algorithm))
    plt.legend()
    plt.show()
plot_accuracy()
