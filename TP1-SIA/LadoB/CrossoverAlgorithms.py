import random as rnd
from Color import Color
from Utils import Utils

class UniformCrossover:
    def __init__(self,p):
        self.p = p
        
    def cross_binary_array(array1,array2,p):
        new_array = []
        for i in range(0,len(array1)):
            if rnd.random() > p:
                new_array.append(array1[i])
            else:
                new_array.append(array2[i])
        return new_array

    def cross(self,color1,color2):
        # RED
        red = UniformCrossover.cross_binary_array(color1.red,color2.red,self.p)
        # GREEN
        green = UniformCrossover.cross_binary_array(color1.green,color2.green,self.p)
        # BLUE
        blue = UniformCrossover.cross_binary_array(color1.blue,color2.blue,self.p)
        return Color(Utils.binaryToDecimal(red),Utils.binaryToDecimal(green),Utils.binaryToDecimal(blue))
    
    def get_new_crossed_population(self,population):
        new_population = []
        for i in range(0,len(population)):
            for j in range(0,len(population[i:])):
                if i != j:
                    new_population.append(self.cross(population[i],population[j]))     
        return new_population