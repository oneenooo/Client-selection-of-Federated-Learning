import logging
from server import volatile
import numpy as np
from threading import Thread
import random

class GreedyServer(volatile.Volatile):
    """Federated learning server that uses profiles to direct during selection."""


    # Federated learning phases
    def selection(self):
        import fl_model  # pylint: disable=import-error
        clients=self.get_all_clients()
        sorted_clients=sorted(clients, key=lambda x: np.sum(self.success_data[x.client_id]), reverse=True)
        clients_per_round = self.config.clients.per_round
        return sorted_clients[:clients_per_round]
