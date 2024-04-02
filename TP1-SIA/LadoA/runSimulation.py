from os import lseek
from FillZone import FillZone
from SearchMethods import DFS,BFS,A,Greedy

heuristic = 2
Algorithms = [1,2,3,4]
f1 = FillZone(5, 4, heuristic)
f2 = FillZone(5, 5, heuristic)
f3 = FillZone(7, 4, heuristic)
fillzones = [f1,f2,f3]

for idx,f in enumerate(fillzones):
    with open("Files/Output" + "_" + str(idx) + "_" + str(heuristic) + ".csv", "w") as file:
        file.write("time,v_nodes,f_nodes,path_len\n")
        for alg in Algorithms:
            if alg == 1:
                sm = BFS(f)
            elif alg == 2:
                sm = DFS(f)
            elif alg == 3:
                sm = Greedy(f)
            elif alg == 4:
                sm = A(f)
            t, v, fr, pl = sm.run()
            file.write(str(t) + "," + str(v) + "," + str(fr) + "," + str(pl) + "\n") 
    file.close()