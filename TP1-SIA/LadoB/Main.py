###
##  While(!Condición de corte):
#       poblacion = seleccionar(poblacion)
#       crossover(poblacion)
##      mutacion(poblacion)


###
from Utils import Utils
import time
import json
from random import random as rnd
from Color import Color
from SelectionAlgorithms import RouletteSelection, ProbabilisticTourney, Elite
from MutationAlgorithms import MultigenMutation
from CrossoverAlgorithms import UniformCrossover
#from Color import Colors

def end_condition(selected_population,objective):
    best_color, idx = Color.get_best_individual(selected_population,objective)
    return objective.compare_colors(best_color) > 0.2

def run():
    population = []
    crossover = UniformCrossover(0.5)
    K = 0 # numero de selección
    objective = -1 #Color objetivo


    with open("../LadoB\\params.json") as file:
        data = json.load(file)
        K = data['K']
        objective = Color(data['objective']['red'], data['objective']['green'], data['objective']['blue'])
        for color in data['palette']:
            population.append(Color(color['red'], color['green'], color['blue']))
        selection_method = data['selection']['name']
        if selection_method == "Probabilistic Tourney":
            threshold = data['selection']['threshold']



    if selection_method == "Probabilistic Tourney":
        selection = ProbabilisticTourney(threshold,K,objective)
    elif selection_method == "Elite":
        selection = Elite(K,objective)
    elif selection_method == "Roulette":
        selection = RouletteSelection(K,objective)

    # #genero mi población inicial
    # for i in range(0,N):
    #     population.append(Color.generate_random_color())
        
    mutation = MultigenMutation(population,0.01)


    n_generations = 1
    ##Empieza loop hasta condicion de corte
    print(objective)

    start_time = time.time()

    selected_population = selection.select(population)
    while(not end_condition(selected_population,objective)):
        n_generations += 1
        population = crossover.get_new_crossed_population(selected_population)
        mutation.set_population(population)
        mutation.mutate_population()
        selected_population = selection.select(population)
        
    t = (time.time() - start_time)
    rta, idx = Color.get_best_individual(selected_population,objective)

    print("Output = " + str(rta))
    print("Time = " + str(t) + " seconds")
    print("Generaciones " + str(n_generations))
    return rta,t,n_generations


with open("Files/Tourney_0.5.csv", "w") as file:
    file.write("red,green,blue,time,generations\n")
    for i in range(0,50):
        rta,t,n = run()
        file.write( str(Utils.binaryToDecimal(rta.red)) + "," 
                   + str(Utils.binaryToDecimal(rta.green)) + "," 
                   + str(Utils.binaryToDecimal(rta.blue)) +"," 
                   + str(t) + "," 
                   + str(n) + "\n")
    file.close()
    
