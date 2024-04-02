import copy
from linecache import getline
import numpy as np
import random
import matplotlib.pyplot as plt


# --- Variables Indexs ---
# i = indice de la neurona de salida
# j = indice de la neurona de capa intermedia
# k = indice de entrada o de capa anterior
# m = indice de capa intermedia
# p = cantidad de datos

K = 30
K2 = 5

class MultiLayerPerceptron:
    def __init__(self, stimuli, expected_output, learning_rate, beta, structure):
        self.stimuli = stimuli
        self.out = expected_output
        self.n = learning_rate
        self.beta = beta
        self.layersCount = len(structure)
        self.neurons_in_layer = structure
        self.best_neurones = []
        self.best_exits = []
        self.neurones = []
        self.latent = []
        self.exits = []

        # Aclaracion: en layers tengo la cantidad total de layers inluyendo la latent
        # Initialize W's with random small numbers
        for i in range(0, self.layersCount):
            self.neurones.append([])
            for j in range(0, self.neurons_in_layer[i]):
                self.neurones[i].append({})
                if (i == 0):
                    self.neurones[i][j]["w"] = np.random.uniform(size=len(self.stimuli[0])+1, low=-1, high=1)
                else:
                    self.neurones[i][j]["w"] = np.random.uniform(size=self.neurons_in_layer[i - 1]+1, low=-1,
                                                                 high=1)  
                self.neurones[i][j]["m"] = i
                self.neurones[i][j]["delta"] = 0
                self.neurones[i][j]["old_dw"] = 0
                self.neurones[i][j]["h"] = 0

        for i in range(0, len(expected_output[0])):
            self.exits.append({})
            self.exits[i]["w"] = np.random.uniform(size=len(expected_output[0])+1, low=-1,
                                                   high=1)
            self.exits[i]["delta"] = 0
            self.exits[i]["error"] = 0
            self.exits[i]["h"] = 0
            self.exits[i]["old_dw"] = 0
            

    def g(self, x):
        return np.tanh(self.beta * x)

    def g_dx_dt(self, x):
        return self.beta * (1 - (self.g(x)) ** 2)

    def g_dx_dt2(self, g):
        return self.beta * (1 - g ** 2)

    def run(self):
        cota = 20000
        k = K #epocas para aumentar o reducir eta
        k2 = K2
        i = 0
        error_history = []
        it = []
        error = 1
        prev_error = error
        e_min = 1000000
        w_min = []
        while i < cota and error > 0.0001:
            error = 0
            for u in range(0, len(self.stimuli)):
                self.propagation(self.stimuli[u], 0)
                self.calculate_exit()
                self.backtracking(u)
                self.update_connections(self.layersCount, u)
                error += self.calculate_error(u)
            # if error < prev_error:
            #     k2 = K2
            #     k = k - 1
            #     if k == 0:
            #         print("Aumento lr: " + str(self.n))
            #         self.n = self.n*10
            #         k = K
            # else:
            #     if error > prev_error:
            #         k = K
            #         k2 = k2 - 1
            #         if k2 == 0:
            #             self.n = 0.1*self.n
            #             print("Disminuyo lr: " + str(self.n))
            #             k2 = K2
            # prev_error = error

            i += 1
            error_history.append(error/len(self.stimuli))
            it.append(i)
            if error < e_min:
                w_min = []
                e_min = error
                for m in range(0, self.layersCount):
                    for n in self.neurones[m]:
                        w_min.append(n["w"])
            if (i % 1000 == 0):
                print(f'It: {i} - Error: {error} - Lr: {self.n}')

        return w_min, it, error_history

    def propagation(self, stimuli, m):
        # neurones[i] = [{"w" : w, "h": h, "v" : v, "m": m}]    S -- L1 -- L2 -- ... -- O
        #if (m == self.layersCount): return
        for m in range(0, self.layersCount):
            for i in range(0, self.neurons_in_layer[m]):
                self.neurones[m][i]["h"] = self.neurones[m][i]["w"][-1]
                # print("m: " + str(m) + " --> i: " + str(i) + " --> j: " + str(j))
                if m != 0:
                    for j in range(0, self.neurons_in_layer[m-1]):
                        self.neurones[m][i]["h"] += self.neurones[m][i]["w"][j] * self.neurones[m-1][j]["v"]
                if m == 0:
                    for j in range(0, len(stimuli)):
                        self.neurones[m][i]["h"] += self.neurones[m][i]["w"][j] * stimuli[j]
                self.neurones[m][i]["v"] = self.g(self.neurones[m][i]["h"])

        #self.propagation(self.neurones[m], m + 1)
    
    def decode(self, x, y):
        latent_layer = 1
        self.neurones[latent_layer][0]["v"] = x
        self.neurones[latent_layer][1]["v"] = y
        for m in range(latent_layer + 1, self.layersCount):
            for i in range(0, self.neurons_in_layer[m]):
                self.neurones[m][i]["h"] = 0
                for j in range(0, self.neurons_in_layer[m-1]):
                        self.neurones[m][i]["h"] += self.neurones[m][i]["w"][j] * self.neurones[m-1][j]["v"]
                self.neurones[m][i]["v"] = self.g(self.neurones[m][i]["h"])
        self.calculate_exit()


    def calculate_exit(self):
        m = self.layersCount - 1
        for i in range(0, len(self.out[0])):
            self.exits[i]["h"] = self.exits[i]["w"][-1]
            for j in range(0, self.neurons_in_layer[m]):  # TODO: checkear este len(expected_output[0]) antes era self.neurons_per_layer
                self.exits[i]["h"] += self.exits[i]["w"][j] * self.neurones[m][j]["v"]
            self.exits[i]["v"] = self.g(self.exits[i]["h"])

    # def calculate_latent_values(self):
    #     latent_layer = 1
    #     for i in range(0,len(self.neurons_in_layer[latent_layer])):
    #         self.neurones[latent_layer][i]["h"] = 0
    #         for j in range(0, self.neurons_in_layer[latent_layer-1]):
    #             self.neurones[i]["h"] += self.exits[i]["w"][j] * self.neurones[m][j]["v"]
    #         self.exits[i]["v"] = self.g(self.exits[i]["h"])

    def backtracking(self, u):
        self.calculate_exit_delta(u)
        for m in reversed(range(1, self.layersCount + 1)):
            for i in range(0, self.neurons_in_layer[m - 1]):
                self.neurones[m - 1][i]["delta"] = 0
                if m == self.layersCount:
                    for j in range(0, len(self.out[u])):
                        self.neurones[m - 1][i]["delta"] += self.g_dx_dt2(self.neurones[m - 1][i]["v"]) * (
                                    self.exits[j]["w"][i] * self.exits[j]["delta"])
                else:
                    for j in range(0, self.neurons_in_layer[m]):
                        self.neurones[m - 1][i]["delta"] += self.g_dx_dt2(self.neurones[m - 1][i]["v"]) * (
                                    self.neurones[m][j]["w"][i] * self.neurones[m][j]["delta"])

    def calculate_exit_delta(self, u):
        for i in range(0, len(self.out[0])):
            self.exits[i]["delta"] = self.g_dx_dt2(self.exits[i]["v"]) * (self.out[u][i] - self.exits[i]["v"])

    def update_exit_connections(self):
        for i in range(0, len(self.out[0])):
            for j in range(0, self.neurons_in_layer[
                self.layersCount - 1]):  # TODO: checkear este len(expected_output[0]) antes era self.neurons_per_layer
                dw = self.n * self.exits[i]["delta"] * self.neurones[self.layersCount - 1][j]["v"] + 0.8 * self.exits[i]["old_dw"]
                self.exits[i]["w"][j] += dw
                self.exits[i]["w"][-1] += self.n * self.exits[i]["delta"]
                self.exits[i]["old_dw"] = dw
                # self.exits[i]["w"][j] = - self.n * self.g_dx_dt(self.exits[i]["w"][j]) + 0.8 * self.exits[i]["delta"]

    def update_first_layer(self, u):
        for i in range(0, self.neurons_in_layer[0]):
            for j in range(0, len(self.stimuli[0])):
                dw = self.n * self.neurones[0][i]["delta"] * self.stimuli[u][j] + 0.8 * self.neurones[0][i]["old_dw"]
                self.neurones[0][i]["w"][j] += dw
                self.neurones[0][i]["w"][-1] += self.n * self.neurones[0][i]["delta"]
                self.neurones[0][i]["old_dw"] = dw
                # self.neurones[0][i]["w"][j] = - self.n * self.g_dx_dt(self.neurones[0][i]["w"][j]) + 0.8 * self.neurones[0][i]["delta"]

    def update_connections(self, m, u):
        for m in reversed(range(0, self.layersCount+1)):
            if m == self.layersCount:
                self.update_exit_connections()
            elif m == 0:
                self.update_first_layer(u)
            else:
                for i in range(0, self.neurons_in_layer[m]):
                    for j in range(0, self.neurons_in_layer[m - 1]):
                        dw = self.n * self.neurones[m][i]["delta"] * self.neurones[m - 1][j]["v"] + 0.8 * self.neurones[m][i]["old_dw"]
                        self.neurones[m][i]["w"][j] += dw
                        self.neurones[m][i]["w"][-1] += self.n * self.neurones[m][i]["delta"]
                        self.neurones[m][i]["old_dw"] = dw
                        # self.neurones[m][i]["w"][j] = - self.n * self.g_dx_dt(self.neurones[m][i]["w"][j]) + 0.8 * self.neurones[m][i]["delta"]
        # if m == 0: return self.update_first_layer(u)
        # if m == self.layersCount:
        #     self.update_exit_connections()
        # else:
        #     for i in range(0, self.neurons_in_layer[m]):
        #         for j in range(0, self.neurons_in_layer[m - 1]):
        #             dw = self.n * self.neurones[m][i]["delta"] * self.neurones[m - 1][j]["v"] + 0.8 * self.neurones[m][i]["old_dw"]
        #             self.neurones[m][i]["w"][j] += dw
        #             self.neurones[m][i]["old_dw"] = dw
        #             # self.neurones[m][i]["w"][j] = - self.n * self.g_dx_dt(self.neurones[m][i]["w"][j]) + 0.8 * self.neurones[m][i]["delta"]
        # self.update_connections(m - 1, u)

    def calculate_error(self, u):
        errors = 0
        for i in range(0, len(self.out[u])):
            errors += (self.out[u][i] - self.exits[i]["v"]) ** 2
        return errors
        # ||x - x'||