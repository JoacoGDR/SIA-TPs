from distutils.log import error
from Utils import Utils
from Color import Color


class MultigenMutation:
    
    #p generalmente va a ser 0.03
    def __init__(self,population,p):
        self.population = population
        self.p = p
    
    def set_population(self,population):
        self.population = population
        
    def mutate_binary(binary_array,p):
        for i in range(0,len(binary_array)):
            binary_array[i] = Utils.bitMutation(binary_array[i],p * (i+1))
    
    def mutate(color,p):
        MultigenMutation.mutate_binary(color.red,p)
        MultigenMutation.mutate_binary(color.green,p)
        MultigenMutation.mutate_binary(color.blue,p)
        
    def mutate_population(self):
        for color in self.population:
            MultigenMutation.mutate(color,self.p)
        
