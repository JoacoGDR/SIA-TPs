from Utils import Utils
import random

class Color:
    def __init__(self, red, green, blue):
        self.red = Utils.decimalToBinary(red)
        self.green = Utils.decimalToBinary(green)
        self.blue = Utils.decimalToBinary(blue)

    def __str__(self):
        return "(" + str(Utils.binaryToDecimal((self.red))) + ", " + str(Utils.binaryToDecimal((self.green))) + ", " + str(Utils.binaryToDecimal((self.blue))) + ")"

    def __repr__(self):
        return "(" + str(Utils.binaryToDecimal((self.red))) + ", " + str(Utils.binaryToDecimal((self.green))) + ", " + str(Utils.binaryToDecimal((self.blue))) + ")"

    #función de aptitud
    def compare_colors(self, other):
        ret = abs(Utils.binaryToDecimal(self.red) -  Utils.binaryToDecimal(other.red))
        ret += abs(Utils.binaryToDecimal(self.green) -  Utils.binaryToDecimal(other.green))
        ret += abs(Utils.binaryToDecimal(self.blue) -  Utils.binaryToDecimal(other.blue))
        if ret == 0:
            return 2
        return 1/ret

    def generate_random_color():
        return Color(random.randint(0,255), random.randint(0,255), random.randint(0,255))
    
    def get_worse_individual(idx1,idx2,population,objective):
        if population[idx1].compare_colors(objective) < population[idx2].compare_colors(objective):
            return population[idx1], idx1
        return population[idx2], idx2 
    
    def get_better_individual(idx1,idx2,population,objective):
        if population[idx1].compare_colors(objective) > population[idx2].compare_colors(objective):
            return population[idx1], idx1
        return population[idx2], idx2 
    
    def get_best_individual(population,objective):
        best = population[0]
        ret_idx = 0
        for idx in range(1,len(population)):
            if objective.compare_colors(best) < objective.compare_colors(population[idx]):
                best = population[idx]
                ret_idx = idx
        return best, ret_idx
        
    def sort_colors(population, objective):
        sorted_list = []
        population_size = len(population)
        for i in range(0, population_size):
            best, idx = Color.get_best_individual(population,objective)
            sorted_list.append(best)
            population.pop(idx)
            
        return sorted_list





