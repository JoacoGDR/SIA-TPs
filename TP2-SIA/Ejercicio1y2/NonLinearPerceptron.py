import random
import numpy as np
import matplotlib.pyplot as plt


class NonLinearPerceptron:

    def __init__(self, stimulus, expected_output, learningRate,beta):
        self.stimulus = stimulus
        self.expected_output = expected_output
        self.learningRate = learningRate
        self.beta = beta
        self.w = np.zeros(len(self.stimulus[0]))
        self.min = min(expected_output)
        self.max = max(expected_output)
        self.escalated_expected_output = self.escalate(expected_output)
    
    def escalate(self, outputs):
        range = self.max - self.min
        ret = []
        for out in outputs:
            ret.append(2*((out-self.min)/range)-1)
        return np.array(ret)

    def escalate_value(self, value):
        range = self.max - self.min
        return 2*((value-self.min)/range)-1

    def zScore(self, inputs):
        mean = np.mean(inputs)
        standard_deviation = np.std(inputs)
        z = []
        for i in inputs:
            z.append((i - mean)/standard_deviation)
        return np.array(z)

    def getH(self, w, u): #o
        aux = self.stimulus[u]
        h = 0
        for i in range(0, len(aux)):
            h += float(w[i]) * float(aux[i])
        return h
    
    def calculateActivation(self,w,u):
        h = self.getH(w,u)
        return np.tanh(self.beta*h)
    
    def calculateActivationDerivative(self,w,u):
        h = self.getH(w,u)
        return self.beta * (1 - ((np.tanh(self.beta * h)**2)))
    
    def calculateError(self, w):
        error = 0
        for u in range(0, len(self.stimulus)):
            o = self.calculateActivation(w, u)
            error += (self.escalated_expected_output[u] - o)**2  #los de la consulta tambien dividen lo de adentro por 2 y despues lo elevan al **2
        return error/2 #en la consulta unos dividian por la cantidad de ejemplos (28)

    def calculateDeltaW(self,w,u):
        ret = np.array([0,0,0,0])
        for u in range(0,len(self.stimulus)):
            aux = ((self.escalated_expected_output[u] - self.calculateActivation(w, u)) * self.calculateActivationDerivative(w, u)) * self.stimulus[u]
            ret = np.add(ret, aux)
        return self.learningRate * ret
    
    def run(self):
        i = 0
        error = 0
        error_min = 10000000000
        cota = 100000
        e = []
        it = []
        while error_min > 0.0001 and i < cota:
            e.append(error_min)
            it.append(i)
            u = random.randint(0, len(self.stimulus) - 1)
            deltaW = self.calculateDeltaW(self.w,u)
            self.w = np.add(self.w ,deltaW)
            error = self.calculateError(self.w)
            if i % 1000 == 0:
                e.append(error)
                it.append(i)
            if error < error_min:
                error_min = error
                w_min = self.w
            i += 1
        if(i >= cota):
            print("Corté por cota")
            self.w = w_min
        return w_min, e, it

    def test(self, stimulus):
        ret = 0
        for i in range(0,len(stimulus)):
            ret += stimulus[i] * self.w[i]
        return np.tanh(self.beta*ret)

