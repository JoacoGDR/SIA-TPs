import numpy as np
import random

class SimplePerceptron:
    def __init__(self, stimulus, expected_output, learningRate, threshold):
        self.stimulus = stimulus
        self.expected_output = expected_output
        self.learningRate = learningRate
        self.threshold = threshold

    def calculateActivation(self, w, u):
        aux = self.stimulus[u]
        h = 0
        for i in range(0, len(aux)):
            h += w[i] * aux[i]
        if (h - self.threshold) < 0:
            return -1
        else:
            return 1

    def test(self, w, aux):
        h = 0
        for i in range(0, len(aux)):
            h += w[i] * aux[i]
        if (h - self.threshold) < 0:
            return -1
        else:
            return 1
    
    def calculateError(self, w):
        error = 0
        for i in range(0, len(self.stimulus)):
            o = self.calculateActivation(w, i)
            error += np.abs(self.expected_output[i] - o)
        return error
    
    def calculateDeltaW(self,w,u):
        o = self.calculateActivation(w,u)
        return ((self.expected_output[u] - o) * self.learningRate) * self.stimulus[u]
    
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
            # if i % 10000 == 0:
            e.append(error)
            it.append(i)
            if error < error_min:
                error_min = error
                w_min = w
            i += 1
        if(i >= cota):
            print("Corté por cota")
        return w_min, e, it