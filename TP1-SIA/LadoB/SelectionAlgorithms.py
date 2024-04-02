import random as rnd
from Color import Color
import math as math

class ProbabilisticTourney:
    def __init__(self, threshold, selection_size,objective_color):
        self.threshold = threshold
        self.selection_size = selection_size
        self.objective_color = objective_color

    def select(self, population):
        selected_population = []
        for i in range(0,self.selection_size):
            #Agarro 2 individuos random
            idx1 = rnd.randint(0,len(population)-1)
            idx2 = idx1
            while(idx1 == idx2):
                idx2 = rnd.randint(0,len(population)-1)
                
            #Dado una probabilidad, me quedo con el mejor o el peor. 

            if(rnd.random() > self.threshold):
                selected_individual, idx = Color.get_worse_individual(idx1,idx2,population,self.objective_color)
            else:
                selected_individual, idx = Color.get_better_individual(idx1,idx2,population,self.objective_color)
                
            population.pop(idx)
            selected_population.append(selected_individual)
        return selected_population
    
    
class Elite:
    def __init__(self,selection_size,objective_color):
        self.selection_size = selection_size
        self.objective_color = objective_color
    
    def select(self,population):
        population = Color.sort_colors(population,self.objective_color)
        selected_population = []  
        aux = 0      
        for i in range(0,self.selection_size):
            n = math.ceil((self.selection_size-i)/len(population))
            for _ in range(0,n):
                if aux < self.selection_size:
                    selected_population.append(population[i])
                    aux += 1
                else:
                    return selected_population
        return selected_population


class RouletteSelection:
    def __init__(self, selection_size,objective_color):
        self.selection_size = selection_size
        self.objective_color = objective_color
        
    def select(self,population):
        selected_population = []
        relative_aptitudes = []
        accumulated_aptitudes = []
        total_aptitude = 0

        #Calculo la aptitud total
        for i in range(0, len(population)):
            total_aptitude += self.objective_color.compare_colors(population[i])
            
        #Populo el array de aptitudes relativas
        for i in range(0, len(population)):
            relative_aptitudes.append(self.objective_color.compare_colors(population[i])/total_aptitude)
        
        accumulated_aptitudes.append(relative_aptitudes[0])
        for i in range(1,len(population)):
            accumulated_aptitudes.append(relative_aptitudes[i] + accumulated_aptitudes[i-1])
        
        for i in range(0,self.selection_size):
            rj = 1
            while rj == 1:
                rj = rnd.random()
                
            for j in range (0,len(population)):
                if accumulated_aptitudes[j] > rj:
                    selected_population.append(population[j])
                    break
        return selected_population

# To sort the list in place...
# ut.sort(key=lambda x: x.count, reverse=True)
