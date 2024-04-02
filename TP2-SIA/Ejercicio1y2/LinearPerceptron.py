import numpy as np
import random

class LinearPerceptron:
    def __init__(self, stimulus, expected_output, learningRate):
        self.stimulus = stimulus
        self.expected_output = expected_output
        self.learningRate = learningRate

    def calculateActivation(self, w, u):
        aux = self.stimulus[u]
        h = 0
        for i in range(0, len(aux)):
            h += float(w[i]) * float(aux[i])
        return h
    
    def calculateError(self, w):
        error = 0
        for u in range(0, len(self.stimulus)):
            o = self.calculateActivation(w, u)
            error += (self.expected_output[u] - o)**2
        return error/2
    
    def calculateDeltaW(self,w,u):
        ret = np.array([0,0,0,0])
        for u in range(0,len(self.stimulus)):
            o = self.calculateActivation(w,u)
            aux = (self.expected_output[u] - o)*self.stimulus[u]
            ret = np.add(ret, aux)
        return self.learningRate * ret
    
    def run(self):
        i = 0
        w = np.zeros(len(self.stimulus[0]))
        error = 0
        error_min = 10000000000
        cota = 100000
        e = []
        it = []
        while error_min > 0 and i < cota:
            u = random.randint(0, len(self.stimulus) - 1)
            deltaW = self.calculateDeltaW(w,u)
            w = np.add(w , deltaW)
            error = self.calculateError(w)
            if i % 200 == 0:
                e.append(error)
                it.append(i)
            if error < error_min:
                error_min = error
                w_min = w
            i += 1
        if(i >= cota):
            print("Corté por cota")
        return w_min, e, it
