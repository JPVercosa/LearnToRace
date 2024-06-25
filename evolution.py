from neural_network import NeuralNetwork, NeuralNetworkDDPG, CQ, Mu
from core import index_loop
from os import listdir, path
import json
from objects import Result, ResultDDPG, ResultTD3
import matplotlib.pyplot as plt


class Entity:
    """
    Neural network entity with name, settings, neural network, ...
    Savable and loadable.
    """

    def __init__(self):
        # parameters
        self.name = ""
        self.acceleration = 0
        self.max_speed = 0
        self.rotation_speed = 0
        self.friction = 0

        self.shape = []

        # info
        self.gen_count = 0
        self.max_score = 0

        # result with nn to save
        self.nn = None

    # Add 1 to gen count
    def increment_gen_count(self):
        self.gen_count += 1

    # Get nn from Result object
    def get_nn(self):
        return self.nn

    # Get nn with random weights with this shape
    def get_random_nn(self):
        nn = NeuralNetwork(self.shape)
        nn.set_random_weights()
        return nn

    # set new result and max score
    def set_nn_from_result(self, result: Result):
        self.nn = result.nn
        self.max_score = result.score

    def get_car_parameters(self):
        return {
            "acceleration": self.acceleration,
            "max_speed": self.max_speed,
            "rotation_speed": self.rotation_speed,
            "friction": self.friction
        }

    def get_save_parameters(self):
        return {
            "name" : self.name,
            "acceleration": self.acceleration,
            "max_speed": self.max_speed,
            "rotation_speed": self.rotation_speed,
            "shape": self.shape,
            "max_score": self.max_score,
            "gen_count": self.gen_count,
            "friction": self.friction
        }

    def set_parameters_from_dict(self, par: dict):
        # get from dict, if not in set default
        self.name = par.get("name", self.name)

        self.acceleration = par.get("acceleration", self.acceleration)
        self.max_speed = par.get("max_speed", self.max_speed)
        self.rotation_speed = par.get("rotation_speed", self.rotation_speed)
        self.shape = par.get("shape", self.shape)
        self.friction = par.get("friction", self.friction)

        self.gen_count = par.get("gen_count", self.gen_count)
        self.max_score = par.get("max_score", self.max_score)

    def save_file(self, save_name="", folder="saves"):
        # if dir already contains that name
        """
        files = listdir(folder)
        name_count = 0
        while save_name + ".json" in files:
            name_count += 1
            save_name = "%s(%s)" % (self.name, name_count)"""
        if not save_name.endswith(".json"):
            save_name += ".json"
        save_file = {
            "settings": self.get_save_parameters(),
            "weights": [np_arr.tolist() for np_arr in self.nn.weights]
        }
        with open(folder + "/" + save_name, "w") as json_file:
            json.dump(save_file, json_file)
        print("Saved ", save_name)

    def moving_average(self, data, window_size):
        moving_averages = []
        for i in range(len(data) - window_size + 1):
            this_window = data[i : i + window_size]
            window_average = sum(this_window) / window_size
            moving_averages.append(window_average)
        return moving_averages

    def save_history(self, save_name="", mode="", history=[], population=1, full_path="none"):

        save_name = save_name.replace(" ", "_").replace(".png", "")
        if not save_name.endswith(".png"):
            save_name += ".png"

        fig, (ax1, ax2) = plt.subplots(2, figsize=(8, 8))	

        if population > 1:
            pop_mean_history = [sum(history[i:i+population])/population for i in range(0, len(history), population)]
            ma_mean = self.moving_average(pop_mean_history, 10)
        pop_max_history = [max(history[i:i+population]) for i in range(0, len(history), population)]
        ma_max = self.moving_average(pop_max_history, 10)
        ax1_values = range(1, len(pop_max_history) + 1)
        ax2_values = range(1, len(ma_max) + 1)
        # create image graph from history
        if population > 1:
            ax1.plot(ax1_values, pop_mean_history, label="Population Mean")
            ax2.plot(ax2_values, ma_mean, label="Mean Moving Average (10)")
        ax1.plot(ax1_values, pop_max_history, label="Population Max")
        ax2.plot(ax2_values, ma_max, label="Max Moving Average (10)")
        ax1.set_title(f"{mode.capitalize()} History")
        ax2.set_title(f"{mode.capitalize()} Moving Average (10) History")
        if mode == "evolutionary": 
            ax1.set_xlabel("Generation") 
            ax2.set_xlabel("Generation")
        else: 
            ax1.set_xlabel("Episode")
            ax2.set_xlabel("Episode")
        ax1.set_ylabel("Reward")
        ax2.set_ylabel("Reward")
        ax1.legend(loc='upper left',
          fancybox=True)
        ax2.legend(loc='upper left',
          fancybox=True)
        fig.tight_layout()
        if path.exists(full_path):
            fig.savefig(full_path)
        else:
            fig.savefig(f"saves/images/{mode}/{save_name}")
        plt.close(fig)
        # save it
        print("Saved ", save_name)
    

    def load_file(self, path):
        #try:
        with open(path) as json_file:
            file_raw = json.load(json_file)

        file_parameters = file_raw["settings"]
        file_weights = file_raw["weights"]

        self.nn = NeuralNetwork(file_parameters["shape"])
        self.nn.set_weights(file_weights)
        self.set_parameters_from_dict(file_parameters)

        print(f"Loaded {path}")
        """except:
            print(f"Failed to load: {path}")
            return False"""

class EntityDDPG(Entity):
    def __init__(self):
        super().__init__()

        self.cq = None
        self.mu = None

    def get_nn(self):
        cq = self.cq
        mu = self.mu
        return cq, mu

    def get_random_nn(self):
        cq = CQ(6, 2, self.shape[1:-1])
        mu = Mu(6, 2, self.shape[1:-1])
        return cq, mu
    
        # set new result and max score
    def set_nn_from_result(self, result: ResultDDPG):
        self.cq = result.cq
        self.mu = result.mu
        self.max_score = result.reward

    def save_file(self, save_name="", folder="saves/reinforcement"):
        # if dir already contains that name
        """
        files = listdir(folder)
        name_count = 0
        while save_name + ".json" in files:
            name_count += 1
            save_name = "%s(%s)" % (self.name, name_count)"""
        save_name = save_name.replace(" ", "_").replace(".json", "")
        self.cq.save("saves/reinforcement/" + save_name + "_cq.pth")
        self.mu.save("saves/reinforcement/" + save_name + "_mu.pth")
        if not save_name.endswith(".json"):
            save_name += ".json"
        save_file = {
            "settings": self.get_save_parameters(),
        }
        with open(folder + "/" + save_name, "w") as json_file:
            json.dump(save_file, json_file)
        print("Saved ", save_name)
    
    def load_file(self, path):
        #try:
        # print(path)
        with open(path) as json_file:
            file_raw = json.load(json_file)

        file_parameters = file_raw["settings"]
        self.set_parameters_from_dict(file_parameters)

        file_name = path.split("/")[-1].split(".")[0]
        print(file_name)
        shape = file_parameters["shape"]
        loaded_cq = CQ(shape[0], shape[-1], shape[1:-1])
        loaded_mu = Mu(shape[0], shape[-1], shape[1:-1])
        
        self.cq = loaded_cq.load(f"saves/reinforcement/{file_name}_cq.pth")
        self.mu = loaded_mu.load(f"saves/reinforcement/{file_name}_mu.pth")
        

        print(f"Loaded {path}")

class EntityTD3(EntityDDPG):
    def __init__(self):
        super().__init__()

        self.cq = None
        self.cq2 = None
        self.mu = None

    def get_nn(self):
        cq = self.cq
        cq2 = self.cq2
        mu = self.mu
        return cq, cq2, mu

    def get_random_nn(self):
        cq = CQ(6, 2, self.shape[1:-1])
        cq2 = CQ(6, 2, self.shape[1:-1])
        mu = Mu(6, 2, self.shape[1:-1])
        return cq, cq2, mu
    
        # set new result and max score
    def set_nn_from_result(self, result: ResultTD3):
        self.cq = result.cq
        self.cq2 = result.cq2
        self.mu = result.mu
        self.max_score = result.score

    def save_file(self, save_name="", folder="saves/reinforcement"):
        # if dir already contains that name
        """
        files = listdir(folder)
        name_count = 0
        while save_name + ".json" in files:
            name_count += 1
            save_name = "%s(%s)" % (self.name, name_count)"""
        save_name = save_name.replace(" ", "_").replace(".json", "")
        self.cq.save("saves/reinforcement/" + save_name + "_cq.pth")
        self.cq2.save("saves/reinforcement/" + save_name + "_cq2.pth")
        self.mu.save("saves/reinforcement/" + save_name + "_mu.pth")
        if not save_name.endswith(".json"):
            save_name += ".json"
        save_file = {
            "settings": self.get_save_parameters(),
        }
        with open(folder + "/" + save_name, "w") as json_file:
            json.dump(save_file, json_file)
        print("Saved ", save_name)
    
    def load_file(self, path):
        #try:
        # print(path)
        with open(path) as json_file:
            file_raw = json.load(json_file)

        file_parameters = file_raw["settings"]
        self.set_parameters_from_dict(file_parameters)

        file_name = path.split("/")[-1].split(".")[0]
        print(file_name)
        shape = file_parameters["shape"]
        loaded_cq = CQ(shape[0], shape[-1], shape[1:-1])
        loaded_cq2 = CQ(shape[0], shape[-1], shape[1:-1])
        loaded_mu = Mu(shape[0], shape[-1], shape[1:-1])
        
        self.cq = loaded_cq.load(f"saves/reinforcement/{file_name}_cq.pth")
        self.cq2 = loaded_cq2.load(f"saves/reinforcement/{file_name}_cq2.pth")
        self.mu = loaded_mu.load(f"saves/reinforcement/{file_name}_mu.pth")
        

        print(f"Loaded {path}")

"""
Class containing info about NNs and its parameters & generates new generations.
"""
class Evolution:
    def __init__(self):
        self.best_result = Result(None,  -1, 0)
        self.mutation_rate = 0

    def load_generation(self, nn: NeuralNetwork, nn_stg: dict, population: int):
        return self.get_new_generation([nn], population)

    def get_new_generation(self, nns: [NeuralNetwork], population: int):
        return [nns[index_loop(i, len(nns))].reproduce(self.mutation_rate) for i in range(population)]

    def get_new_generation_from_results(self, results: [Result], population: int, to_add_count=3):
        best_nns = []
        # order by cp_score - if equal than dist_to_next_cp
        # sorted_results = sorted(results, key=lambda x: (x[1], -x[2]), reverse=True)
        sorted_results = sorted(results, reverse=True)

        # add best X
        to_add = to_add_count if len(sorted_results) >= to_add_count else len(sorted_results)
        for i in range(to_add):
            best_nns.append(sorted_results[i].nn)

        return self.get_new_generation(best_nns, population)

    def find_best_result(self, results: [Result]):
        # best cp_score - if equal than better dist_to_next_cp
        current_best_result = max(results)
        self.best_result = current_best_result if current_best_result > self.best_result else self.best_result
        return self.best_result

class EvolutionDDPG(Evolution):
    def __init__(self):
        self.best_result = ResultDDPG(None, None, -1, -1, 0)
        self.mutation_rate = 0

    def load_generation(self, cq: CQ, mu: Mu, nn_stg: dict, population: int):
        return self.get_new_generation([cq], [mu], population)

    def get_new_generation(self, cq: [CQ], mu: [Mu], population: int):
        shape = cq[0].shape
        base_CQ = CQ(6, 2, shape[1:-1])
        base_MU = Mu(6, 2, shape[1:-1])
        cq_list = []
        mu_list = []
        # nn_list = [nns[index_loop(i, len(nns))].reproduce(self.mutation_rate) for i in range(population)]
        # cq_list = [cq[index_loop(i, len(cq))].copyfrom(base_CQ) for i in range(population)]
        # mu_list = [mu[index_loop(i, len(mu))].copyfrom(base_MU) for i in range(population)]
        for i in range(population):
            base_CQ.copyfrom(cq[index_loop(i, len(cq))])
            cq_list.append(base_CQ)
            base_MU.copyfrom(mu[index_loop(i, len(mu))])
            mu_list.append(base_MU)
        return cq_list, mu_list

    def get_new_generation_from_results(self, results: [ResultDDPG], population: int, to_add_count=1):
        best_cqs = []
        best_mus = []
        
        # order by cp_score - if equal than dist_to_next_cp
        sorted_results = sorted(results, reverse=True)

        # add best X
        to_add = min(to_add_count, len(sorted_results))
        for i in range(to_add):
            best_cqs.append(sorted_results[i].cq)
            best_mus.append(sorted_results[i].mu)

        return self.get_new_generation(best_cqs, best_mus, population)

    def find_best_result(self, results: [ResultDDPG]):
        # best cp_score - if equal than better dist_to_next_cp
        current_best_result = max(results)
        self.best_result = current_best_result if current_best_result > self.best_result else self.best_result
        return self.best_result

        
class EvolutionTD3(EvolutionDDPG):
    def __init__(self):
        self.best_result = ResultTD3(None, None, None, -1, -1, 0)
        self.mutation_rate = 0
    
    def load_generation(self, cq: CQ, cq2: CQ, mu: Mu, nn_stg: dict, population: int):
        return self.get_new_generation([cq], [cq2], [mu], population)

    def get_new_generation(self, cq: [CQ], cq2: [CQ], mu: [Mu], population: int):
        shape = cq[0].shape
        base_CQ = CQ(6, 2, shape[1:-1])
        base_CQ2 = CQ(6, 2, shape[1:-1])
        base_MU = Mu(6, 2, shape[1:-1])
        cq_list = []
        cq2_list = []
        mu_list = []
        # nn_list = [nns[index_loop(i, len(nns))].reproduce(self.mutation_rate) for i in range(population)]
        # cq_list = [cq[index_loop(i, len(cq))].copyfrom(base_CQ) for i in range(population)]
        # mu_list = [mu[index_loop(i, len(mu))].copyfrom(base_MU) for i in range(population)]
        for i in range(population):
            base_CQ.copyfrom(cq[index_loop(i, len(cq))])
            cq_list.append(base_CQ)
            base_CQ2.copyfrom(cq2[index_loop(i, len(cq2))])
            cq2_list.append(base_CQ2)
            base_MU.copyfrom(mu[index_loop(i, len(mu))])
            mu_list.append(base_MU)
        return cq_list, cq2_list, mu_list

    def get_new_generation_from_results(self, results: [ResultTD3], population: int, to_add_count=1):
        best_cqs = []
        best_cqs2 = []
        best_mus = []
        
        # order by cp_score - if equal than dist_to_next_cp
        sorted_results = sorted(results, reverse=True)

        # add best X
        to_add = min(to_add_count, len(sorted_results))
        for i in range(to_add):
            best_cqs.append(sorted_results[i].cq)
            best_cqs2.append(sorted_results[i].cq2)
            best_mus.append(sorted_results[i].mu)

        return self.get_new_generation(best_cqs, best_cqs2, best_mus, population)

    def find_best_result(self, results: [ResultTD3]):
        # best cp_score - if equal than better dist_to_next_cp
        current_best_result = max(results)
        self.best_result = current_best_result if current_best_result > self.best_result else self.best_result
        return self.best_result

class CustomEvolution(Evolution):
    pass