import csv
import matplotlib.pyplot as plt

files = ["Elite.csv","Roulette.csv"
         ,"Tourney_0.1.csv"
         ,"Tourney_0.2.csv"
         ,"Tourney_0.3.csv"
         ,"Tourney_0.4.csv"
         ,"Tourney_0.5.csv"
         ,"Tourney_0.6.csv"
         ,"Tourney_0.7.csv"
         ,"Tourney_0.8.csv"
         ,"Tourney_0.9.csv"]

algorithms = [
        "Elite"
        ,"Roulette"
        ,"Tourney 0.1"
        ,"Tourney 0.2"
        ,"Tourney 0.3"
        ,"Tourney 0.4"
        ,"Tourney 0.5"
        ,"Tourney 0.6"
        ,"Tourney 0.7"
        ,"Tourney 0.8"
        ,"Tourney 0.9"
]

def get_avg(x):
    ret = 0
    for xi in x:
        ret += xi/len(x)
    return ret

def get_std(x):
    avg = get_avg(x)
    ret = 0
    for xi in x:
        ret += (xi - avg)**2
    return (ret/len(x))**(0.5) 
    
def get_info_from_csv(f):
    t = []
    rgb = []
    n = []
    with open("Files/" + str(f),'r') as csvf:
        plots = csv.reader(csvf, delimiter = ',')
        next(plots)
        for row in plots:
            rgb.append([int(row[0]),int(row[1]),int(row[2])])
            t.append(float(row[3]))
            n.append(int(row[4]))
    return t, rgb, n

t = {}
rgb = {}
n = {}
avg_t = {}
avg_n = {}
std_t = {}
std_n = {}

for a in algorithms:
    t[a] = []
    rgb[a] = []
    n[a] = []
    avg_t[a] = 0
    avg_n[a] = 0
    std_t[a] = 0
    std_n[a] = 0


for idx, f in enumerate(files):
    t[algorithms[idx]], rgb[algorithms[idx]], n[algorithms[idx]] = get_info_from_csv(f)
    avg_t[algorithms[idx]] = get_avg(t[algorithms[idx]])
    avg_n[algorithms[idx]] = get_avg(n[algorithms[idx]])
    std_t[algorithms[idx]] = get_std(t[algorithms[idx]])
    std_n[algorithms[idx]] = get_std(n[algorithms[idx]])

#Tourneys 
thresholds = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]   
avg_gens = []
std_gens = []
for threshold in thresholds:
    avg_gens.append( avg_n["Tourney " +str(threshold)])
    std_gens.append(std_n["Tourney " +str(threshold)])


plt.errorbar(thresholds,avg_gens,std_gens)
plt.show()