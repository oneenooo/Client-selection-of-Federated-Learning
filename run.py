import argparse
import client
import config
import logging
import os
import server
import numpy as np
from print_log import Logger
import sys
#from memory_profiler import profile

# Set up parser
parser = argparse.ArgumentParser()
parser.add_argument('-c', '--config', type=str, default='./configs/CIFAR-10/cifar-10.json',
                    help='Federated learning configuration file.')
parser.add_argument('-l', '--log', type=str, default='INFO',
                    help='Log messages level.')

args = parser.parse_args()

# Set logging
logging.basicConfig(
    format='[%(levelname)s][%(asctime)s]: %(message)s', level=getattr(logging, args.log.upper()), datefmt='%H:%M:%S')

#@profile
def main():
    """Run a federated learning simulation."""
    sys.stdout = Logger("")
    np.random.seed(2)

    # Read configuration file
    fl_config = config.Config(args.config)


    # Initialize server
    fl_server = {
        "basic": server.Server(fl_config),
        "kcenter": server.KCenterServer(fl_config),
        'greedy': server.GreedyServer(fl_config),
        "volatile": server.Volatile(fl_config),
        "CBE3": server.CBE3Server(fl_config)
    }[fl_config.server]
    fl_server.boot()

    # Run federated learning
    fl_server.run()

    # Delete global model
    os.remove(fl_config.paths.model + '/global_'+fl_config.algorithm+"_"+str(fl_config.clients.per_round)+"_"+ str(fl_config.data.IID))


if __name__ == "__main__":
    main()
