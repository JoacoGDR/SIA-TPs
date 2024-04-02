import copy
from linecache import getline
import numpy as np
import random
import matplotlib.pyplot as plt
import imageio

# --- Variables Indexs ---
# i = indice de la neurona de salida
# j = indice de la neurona de capa intermedia
# k = indice de entrada o de capa anterior
# m = indice de capa intermedia
# p = cantidad de datos

class MultiLayerPerceptron:
    def __init__(self, stimuli, expected_output, learning_rate, beta,layers,neurons_per_layer):
        self.stimuli = stimuli
        self.out = expected_output
        self.n = learning_rate
        self.beta = beta
        self.layers = layers
        self.neurons_per_layer = neurons_per_layer
        self.best_neurones = []
        self.best_exits = []
        self.neurones = []
        self.exits = []
        #Initialize W's with random small numbers
        for i in range(0,self.layers):
            self.neurones.append([])
            for j in range(0,self.neurons_per_layer):
                self.neurones[i].append({})
                if(i == 0):
                    self.neurones[i][j]["w"] = np.random.uniform(size=len(self.stimuli[0]), low=-1, high=1)
                    #self.neurones[i][j]["w"] = np.zeros(len(self.stimuli[0])
                else:
                    self.neurones[i][j]["w"] = np.random.uniform(size=neurons_per_layer, low=-1,high=1)
                    #self.neurones[i][j]["w"] = np.zeros(self.neurons_per_layer)
                self.neurones[i][j]["m"] = i
                self.neurones[i][j]["delta"] = 0
                self.neurones[i][j]["h"] = 0
                
        
        for i in range(0,len(expected_output[0])):
            self.exits.append({})
            self.exits[i]["w"]= np.random.uniform(size=self.neurons_per_layer, low=-1, high=1)
            #self.exits[i]["w"]= np.zeros(self.neurons_per_layer)
            self.exits[i]["delta"] = 0
            self.exits[i]["error"] = 0
            self.exits[i]["h"] = 0
        
    def g(self,x):
        return np.tanh(self.beta * x)
    
    def g_dx_dt(self, x):
        return self.beta * (1 - ((self.g(x)))**2)
    
    def run(self):
        cota = 100000
        i = 0
        error_history = []
        it = []
        error = 1
        e_min = 1000000
        w_min = []
        while  i < cota and error > 0.0001:
            error = 0
            for u in range(0, len(self.stimuli)):
                self.propagation(self.stimuli[u],0)
                self.calculate_exit()
                self.backtracking(u)
                self.update_connections(self.layers,u)
                
                error += self.calculate_error(u)
            i += 1
            error_history.append(error)
            it.append(i)
            if error < e_min:
                w_min = []
                e_min = error
                for m in range(0, self.layers):
                    for n in self.neurones[m]:
                        w_min.append(n["w"])
        return w_min, it, error_history
    
    def propagation(self,prev_layer,m):
        #neurones[i] = [{"w" : w, "h": h, "v" : v, "m": m}]
        if(m == self.layers): return
        
        for i in range(0,self.neurons_per_layer):
            self.neurones[m][i]["h"] = 0
            for j in range(0,len(prev_layer)):
                if m != 0:
                    self.neurones[m][i]["h"] += self.neurones[m][i]["w"][j] * prev_layer[j]["v"]
                if m == 0:
                    self.neurones[m][i]["h"] += self.neurones[m][i]["w"][j] * prev_layer[j]
                    
            self.neurones[m][i]["v"] = self.g(self.neurones[m][i]["h"])
        
        self.propagation(self.neurones[m],m+1)
                
    def calculate_exit(self):
        m = self.layers - 1
        for i in range(0,len(self.out[0])):
            self.exits[i]["h"] = 0
            for j in range(0,self.neurons_per_layer):
                self.exits[i]["h"] += self.exits[i]["w"][j] * self.neurones[m][j]["v"]
            self.exits[i]["v"] = self.g(self.exits[i]["h"])
    
    def backtracking(self,u):
        self.calculate_exit_delta(u)           
        for m in reversed(range(1,self.layers+1)):
            for i in range(0,self.neurons_per_layer):
                self.neurones[m-1][i]["delta"] = 0
                if(m == self.layers):
                    for j in range(0,len(self.out[u])):
                        self.neurones[m-1][i]["delta"] += self.g_dx_dt(self.neurones[m-1][i]["h"])* (self.exits[j]["w"][i] * self.exits[j]["delta"])
                else:
                    for j in range(0,self.neurons_per_layer):
                        self.neurones[m-1][i]["delta"] += self.g_dx_dt(self.neurones[m-1][i]["h"])* (self.neurones[m][j]["w"][i] * self.neurones[m][j]["delta"])

    def calculate_exit_delta(self,u):
        for i in range(0,len(self.out[0])):
            self.exits[i]["delta"] = self.g_dx_dt(self.exits[i]["h"])*(self.out[u][i] - self.exits[i]["v"])
            
    def update_exit_connections(self):
        for i in range(0,len(self.out[0])):
            for j in range(0,self.neurons_per_layer):
                self.exits[i]["w"][j] += self.n * self.exits[i]["delta"] * self.neurones[self.layers-1][j]["v"]
    
    def update_first_layer(self,u):
        for i in range(0,self.neurons_per_layer):
            for j in range(0,len(self.stimuli[0])):
                self.neurones[0][i]["w"][j] += self.n * self.neurones[0][i]["delta"] * self.stimuli[u][j]
    
    def update_connections(self,m,u):                          
        if m == 0: return self.update_first_layer(u)
        if m == self.layers:
            self.update_exit_connections()
        else:
            for i in range(0,self.neurons_per_layer):
                for j in range(0,self.neurons_per_layer):
                    # print(self.n * self.neurones[m][i]["delta"] * self.neurones[m-1][j]["v"])
                    self.neurones[m][i]["w"][j] += self.n * self.neurones[m][i]["delta"] * self.neurones[m-1][j]["v"]
        
        self.update_connections(m-1,u)
        
    def calculate_error(self,u):
        errors  = 0
        for i in range(0,len(self.out[u])):
            errors += (self.out[u][i] - self.exits[i]["v"])**2
        return 0.5 * errors

    def propagation_test(self, st):
        return self.propagation_test_rec(st, 0)

    def propagation_test_rec(self,st,m):
        #neurones[i] = [{"w" : w, "h": h, "v" : v, "m": m}]
        if(m == self.layers):
            ret = np.zeros(len(self.out[0]))
            for j in range(0,len(ret)):
                for i in range(0,len(st)):
                    ret[j] +=  self.exits[j]["w"][i] * st[i]
                ret[j] = np.tanh(self.beta * ret[j])
            return ret
        
        aux = np.zeros(self.neurons_per_layer)
        for i in range(0,self.neurons_per_layer):
            for j in range(0,len(st)):
                aux[i] += self.neurones[m][i]["w"][j] * st[j]
        return self.propagation_test_rec(aux,m+1)
        
#----------------------------------
def graph(x, y):
    parameters = {'xtick.labelsize': 12,'ytick.labelsize': 12, 'axes.labelsize': 14}
    plt.rcParams.update(parameters)
    plt.figure(figsize=(7,5))

    plt.plot(x, y)
    plt.semilogy()
    plt.ylabel("Error")
    plt.xlabel("Iterations")
    plt.show()

def getLinePointsEj1(w):
    evaluate = lambda y, m, x, b: (m/y) * x + (b/y)
    x1 = 5
    x2 = -5
    y1 = evaluate(w[2],-w[1],x1,-w[0])
    y2 = evaluate(w[2],-w[1],x2,-w[0])
    return [x1, x2], [y1, y2]

def makeGIF(errors, it, beta):
    for i in range(0,len(errors)):
        parameters = {'xtick.labelsize': 12,'ytick.labelsize': 12, 'axes.labelsize': 14}
        plt.rcParams.update(parameters)
        plt.figure(figsize=(7,5))

        plt.ylabel("Error")
        plt.xlabel("Iterations")

        plt.title("Beta - " + str(beta[i]))
        # plt.ylim(0, 100000)
        plt.semilogy()
        plt.plot(it[i], errors[i])
        plt.savefig(f'Ejercicio3/PNGs/line-{i}.png')
        plt.close()
    with imageio.get_writer('Ejercicio3/PNGs/line.gif', mode='i') as writer:
        for i in range(0, len(errors)):
            image = imageio.imread(f'Ejercicio3/PNGs/line-{i}.png')
            # Para que el gif no vaya tan rapido cada imagen 
            # la agregamos 5 veces
            for i in range(0,5):
                writer.append_data(image)

def add_noise(data, qty):
    used_indexes = []
    l = 0
    data = copy.deepcopy(data)
    while l < qty: 
        i = random.randint(0,len(data))
        if i not in used_indexes:
            used_indexes.append(i)
            if data[i] == 0:
                data[i] = 1
            else:
                data[i] = 0
            l += 1
    return data
#----------------------------------
stimuli = [[1,1, -1],[1,-1, 1],[-1,1, 1],[-1,-1, -1]]
expected_output = [[1],[1],[-1],[-1]]

def runEj3_1(beta, learningRate):
    optimus = MultiLayerPerceptron(stimuli,expected_output,learningRate, beta ,1,3)
    w_min, it, err = optimus.run()

    # graph(it, err)
    
    optimus.propagation([1,-1, 1], 0)
    optimus.calculate_exit()
    print([1,-1, 1], optimus.exits[0]["v"])
    
    optimus.propagation([-1, 1, 1], 0)
    optimus.calculate_exit()
    print([-1, 1, 1], optimus.exits[0]["v"])
    
    optimus.propagation([-1, -1, -1], 0)
    optimus.calculate_exit()
    print([-1, -1, -1], optimus.exits[0]["v"])
    
    optimus.propagation([1, 1, -1], 0)
    optimus.calculate_exit()
    print([1, 1, -1], optimus.exits[0]["v"])

    # plt.scatter(1, 1, color = "red")
    # plt.scatter(1, -1, color = "blue")
    # plt.scatter(-1, 1, color = "blue")
    # plt.scatter(-1, -1, color="red")
    # a1, a2 = getLinePointsEj1(ret[0])
    # b1, b2 = getLinePointsEj1(ret[1])
    # c1, c2 = getLinePointsEj1(ret[2])
    # plt.plot(a1,a2)
    # plt.plot(b1,b2)
    # plt.plot(c1,c2)
    # plt.xlim([-4, 4])
    # plt.ylim([-4, 4])
    # plt.show()  

    return w_min, err, it

# ----------------------------------
train_set = [
    #0
    [0, 1, 1, 1, 0, 1, 0, 0, 0, 1, 1, 0, 0, 1, 1, 1, 0, 1, 0, 1, 1, 1, 0, 0, 1, 1, 0, 0, 0, 1, 0, 1, 1, 1, 0 ],
    #2
    [0, 1, 1, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 1, 1 ],
    #4
    [0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 1, 1, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0 ],
    #5
    [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 1, 1, 1, 0 ],
    #7
    [1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0 ],
    #8
    [0, 1, 1, 1, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 1, 1, 1, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 1, 1, 1, 0],
    #9
    [0, 1, 1, 1, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0 ],
]

train_set_output = [[-1], [-1], [-1], [1], [1], [-1], [1]]

test_set = [
    #1
    [0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 0],
    #3
    [0, 1, 1, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 1, 1, 1, 0],
    #6
    [0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 1, 1, 1, 0],
]

test_set_output = [[1], [1], [-1]]
# ----------------------------------
def runEj3_2(beta, learningRate):
    digits = MultiLayerPerceptron(train_set,train_set_output,learningRate, beta, 1, 5)
    ws, it, e = digits.run()

    for st in train_set:
        digits.propagation(st, 0)
        digits.calculate_exit()
        print(st, digits.exits[0]["v"])
    print("-----------------------------")
    for st in test_set:
        digits.propagation(st, 0)
        digits.calculate_exit()
        print(st, digits.exits[0]["v"])
    
    return it, e
#----------------------------------
data = [
    #0
    [0, 1, 1, 1, 0, 1, 0, 0, 0, 1, 1, 0, 0, 1, 1, 1, 0, 1, 0, 1, 1, 1, 0, 0, 1, 1, 0, 0, 0, 1, 0, 1, 1, 1, 0 ],
    #1
    [0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 0 ],
    #2
    [0, 1, 1, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 1, 1 ],
    #3
    [0, 1, 1, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 1, 1, 1, 0 ],
    #4
    [0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 1, 1, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0 ],
    #5
    [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 1, 1, 1, 0 ],
    #6
    [0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 1, 1, 1, 0 ],
    #7
    [1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0 ],
    #8
    [0, 1, 1, 1, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 1, 1, 1, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 1, 1, 1, 0],
    #9
    [0, 1, 1, 1, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0 ],
]

data_output = [
    [-1, 1, 1, 1, 1, 1, 1, 1, 1, 1], 
    [1, -1, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, 1, -1, 1, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, -1, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, -1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, -1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 1, -1, 1, 1, 1],
    [1, 1, 1, 1, 1, 1, 1, -1, 1, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, -1, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, -1],
]

data_test = [
    #0
    [0, 1, 1, 1, 0, 1, 0, 0, 0, 1, 1, 0, 0, 1, 1, 1, 0, 1, 0, 1, 1, 1, 0, 0, 1, 1, 0, 0, 0, 1, 0, 1, 1, 1, 0 ],
    #1
    [0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 0 ],
    #6
    [0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 1, 1, 1, 0 ],
]

data_expected_output = [
    [-1, 1, 1, 1, 1, 1, 1, 1, 1, 1], 
    [1, -1, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 1, -1, 1, 1, 1],
]

data_test_3_bit_changed = [
    #0
    add_noise(data_test[0], 3),
    #1
    add_noise(data_test[1], 3),
    #6
    add_noise(data_test[2], 3),
]

data_test_6_bit_changed = [
    #0
    add_noise(data_test[0], 6),
    #1
    add_noise(data_test[1], 6),
    #6
    add_noise(data_test[2], 6),
]

#----------------------------------
def runEj3_3(beta, learningRate):
    digits_v2 = MultiLayerPerceptron(data, data_output,learningRate, beta, 1, 10)
    ws, it, e = digits_v2.run()

    aux = []
    for st in data_test:
        digits_v2.propagation(st, 0)
        digits_v2.calculate_exit()
        ret = []
        for e in digits_v2.exits:
            ret.append(e["v"])
        aux.append(ret)
    for i in aux:
        print(i)
    print("---------------")

    aux = []
    for st in data_test_3_bit_changed:
        digits_v2.propagation(st, 0)
        digits_v2.calculate_exit()
        ret = []
        for e in digits_v2.exits:
            ret.append(e["v"])
        aux.append(ret)
    for i in aux:
        print(i)
    print("---------------")

    aux = []
    for st in data_test_6_bit_changed:
        digits_v2.propagation(st, 0)
        digits_v2.calculate_exit()
        ret = []
        for e in digits_v2.exits:
            ret.append(e["v"])
        aux.append(ret)
    for i in aux:
        print(i)
    print("---------------")

    return it, e
#----------------------------------
beta = 1
learningRate = 0.001

runEj3_1(beta, learningRate)

runEj3_2(beta, learningRate)

runEj3_3(beta, learningRate)