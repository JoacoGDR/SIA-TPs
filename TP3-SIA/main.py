from MultiLayerPerceptron import MultiLayerPerceptron
import numpy as np
import random
import matplotlib.pyplot as plt
from font import Font3, Font3_dict, chrs

# AUX FUNCTIONS
# --------------------------------------------------------
def graph_error_by_epoch(x, y):
    parameters = {'xtick.labelsize': 12, 'ytick.labelsize': 12, 'axes.labelsize': 14}
    plt.rcParams.update(parameters)
    plt.figure(figsize=(7, 5))

    plt.plot(x, y)
    plt.ylabel("Error")
    plt.xlabel("Iterations")
    plt.show()

def printResult(array):
    ret = []
    j = 0
    for i in array:
        if i == 1 or i == "1":
            ret.append("*")
        else:
            ret.append(" ")
        j = j + 1
        if j == 5:
            ret.append("\n")
            j=0
    return ret

def getPoint(p1, p2):
    m = (p2[1] - p1[1])/(p2[0] - p1[0])
    # y = mx + b ==> b = y -mx
    b = p1[1] - m * p1[0]
    evaluate = lambda x : m * x + b
    newX = (p1[0] + p2[0])/2
    return newX, evaluate(newX)

def graph_latent_space(ret):
    parameters = {'xtick.labelsize': 12, 'ytick.labelsize': 12, 'axes.labelsize': 14}
    plt.rcParams.update(parameters)
    plt.figure(figsize=(7, 5))
    fig, ax = plt.subplots()
    for n in ret:
        ax.scatter(n[0], n[1])
    plt.xlabel('x')
    plt.ylabel('y')
    ann = []
    for key, value in Font3_dict.items():
        ann.append(key)
    for i, txt in enumerate(ann):
        ax.annotate(ann[i], (ret[i][0], ret[i][1]))
    plt.show()

def propagate(optimus, font3):
    optimus.propagation(font3, 0)
    optimus.calculate_exit()
    exit = []
    for i in optimus.exits:
        a = i["v"]
        if a >= 0.5:
            a = 1
        else:
            a = 0
        exit.append(a)
    return exit

def get_bits_diff(font3, output):
    ret = 0
    for i in range(0, len(font3)):
        if font3[i] != output[i]:
            ret += 1
    return ret

def graph_error_by_bit(error):
    parameters = {'xtick.labelsize': 12,'ytick.labelsize': 12, 'axes.labelsize': 14}
    plt.rcParams.update(parameters)
    plt.figure(figsize=(7,5))
    plt.bar(chrs, error)
    plt.ylabel('Error by bit')
    plt.show()

def get_decode_exit(x, y):
    optimus.decode(x, y)
    decode_exit = []
    for i in optimus.exits:
        a = i["v"]
        if a >= 0.5:
            a = 1
        else:
            a = 0
        decode_exit.append(a)
    return decode_exit

def apply_noise(stimuli, probability):
    noises = (np.vectorize(
                lambda s: 1 - s if np.random.choice(a=[False, True], p=[1 - probability, probability]) else s)(stimuli))
    return noises

# --------------------------------------------------------


# EJS FUNCTIONS
# --------------------------------------------------------
def run_Ej1a2(optimus):
    bit_errors = []
    for h in range(0, len(Font3)):
        exit = propagate(optimus, Font3[h])
        bit_errors.append(get_bits_diff(Font3[h], exit))
    graph_error_by_bit(bit_errors)

def run_Ej1a3(optimus):
    latent_points = []
    latent_layer = 1
    for h in range(0, len(Font3)):
        propagate(optimus, Font3[h])
        latent_points.append([optimus.neurones[latent_layer][0]["v"], optimus.neurones[latent_layer][1]["v"]])
    graph_latent_space(latent_points)
    return

def run_Ej1a4(optimus, chr1_index, chr2_index):
    latent_points = []
    latent_layer = 1
    for h in range(0, len(Font3)):
        propagate(optimus, Font3[h])
        latent_points.append([optimus.neurones[latent_layer][0]["v"], optimus.neurones[latent_layer][1]["v"]])
    
    p1 = [latent_points[chr1_index][0], latent_points[chr1_index][1]]
    p2 = [latent_points[chr2_index][0], latent_points[chr2_index][1]]
    x, y = getPoint(p1, p2)

    print(*printResult(get_decode_exit(p1[0], p1[1])))
    print(*printResult(get_decode_exit(p2[0], p2[1])))
    print(*printResult(get_decode_exit(x, y)))

def run_Ej1_b(beta, learningRate, structure):
    probability = [0, 0.1, 0.3]
    for prob in probability:
        Font3_noise = []
        for f in Font3:
            Font3_noise.append(apply_noise(f, prob))
        optimus = MultiLayerPerceptron(Font3_noise, Font3, learningRate, beta, structure)
        optimus.run()
        errors = []
        Font3_noise = []
        for f in Font3:
            Font3_noise.append(apply_noise(f, prob))
        for h in range(0, len(Font3_noise)):
            exit = propagate(optimus, Font3_noise[h])
            errors.append(np.sum(abs(np.array(exit) - np.array(Font3[h])), axis=0) / 35)
            plt.plot(chrs, errors, label=prob)
    plt.legend()
    plt.show()

def print_all_output(optimus):
    for h in range(0, len(Font3)):
        exit = propagate(optimus, Font3[h])
        print("\n-------------------------------------------------------------------------------------------\n")
        print("Character ->  \"", list(Font3_dict.keys())[list(Font3_dict.values()).index(Font3[h])], " \"\n")
        print("Input --> |\n", *printResult(Font3[h]), "|\n")
        print("Output -> |\n", *printResult(exit), "|\n")


def runEj3_2(beta, learningRate, noise):
    optimus = MultiLayerPerceptron(noise, Font3, learningRate, beta, structure)
    w_min, it, err = optimus.run()
    ret = []
    latent_layer = 1
    for h in range(0, len(noise)):
        optimus.propagation(noise[h], 0)
        optimus.calculate_exit()
        exit = []
        for i in optimus.exits:
            a = i["v"]
            if a >= 0.5:
                a = 1
            else:
                a = 0
            exit.append(a)
        ret.append([optimus.neurones[latent_layer][0]["v"], optimus.neurones[latent_layer][1]["v"]])
        print("\n-------------------------------------------------------------------------------------------\n")
        print("Character ->  \"", list(Font3_dict.keys())[list(Font3_dict.values()).index(Font3[h])], " \"\n")
        print("Input --> |\n", *printResult(noise[h]), "|\n")
        print("Output -> |\n", *printResult(exit), "|\n")
        print("Correct -> |\n", *printResult(Font3[h]), "|\n")
        # print(Font3[h],"\n\n", exit)
    graph_error_by_epoch(it, err)
    graph_latent_space(ret)
    decodeExits = []
    for i in optimus.exits:
            a = i["v"]
            if a >= 0.5:
                a = 1
            else:
                a = 0
            decodeExits.append(a)
    print("Output -> |", *printResult(decodeExits), "|\n")
    return w_min, err, it

def printDiff(noise, pos):
    for i in range(0, len(Font3)):
        print(Font3[i])
        print(noise[i])
        if i in pos:
            print("ALTERADO")
        print("-----------------")
# --------------------------------------------------------

beta = 0.5
learningRate = 0.01
structure = [25,15, 2, 15,25] # cantidad de neuronas por layer

optimus = MultiLayerPerceptron(Font3, Font3, learningRate, beta, structure)
w_min, it, err = optimus.run()
graph_error_by_epoch(it, err)
run_Ej1a2(optimus)
run_Ej1a3(optimus)
run_Ej1a4(optimus, 13, 14)
print_all_output(optimus)

# run_Ej1b(beta, learningRate, structure) # -> este grafica el error al aplicar disitntos niveles de ruido