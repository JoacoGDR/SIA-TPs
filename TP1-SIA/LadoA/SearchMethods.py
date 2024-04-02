import copy
from telnetlib import NOP
import time
import random
from collections import deque
from FillZone import FillZone, Pair, Triple


class BFS:
    def __init__(self, fillzone):
        # Nodos visitados
        self.v = []
        # Nodos frontera
        self.f = deque()
        self.f.append(fillzone.get_state())
        # Array con la cantidad de colores disponible
        self.available_colours = list(range(0, fillzone.state.colours))

    def run(self):
        print("Running BFS...")
        start_time = time.time()
        while len(self.f) > 0:
            state = self.f.popleft()
            if not state.is_finished():
                # Expando
                for i in self.available_colours:
                    if i != state.current_colour:
                        newState = copy.deepcopy(state)
                        newState.current_colour = i
                        newState.paint(i)
                        newState.moves_made += 1
                        newState.path.append(state.current_colour)
                        if newState not in self.v:
                            self.f.append(newState)
                            self.v.append(newState)
            else:
                state.path.append(state.current_colour)
                print("--- GANASTE ---")
                t = (time.time() - start_time)
                print("Tiempo: %s seconds" % t)
                print(f"Nodos visitados: {len(self.v)}")
                print(f"Nodos frontera: {len(self.f)}")
                print(f"Camino Solucion: {state.path}")
                print(f"Costo de la solucion: {len(state.path)} Movimientos")
                print("---------------")
                return t, len(self.v),len(self.f), len(state.path)

class DFS:
    def __init__(self, fillzone):
        # Nodos visitados
        self.v = []
        # Nodos frontera
        self.f = deque()
        self.f.append(fillzone.get_state())
        # Array con la cantidad de colores disponible
        self.available_colours = list(range(0, fillzone.state.colours))

    def run(self):
        print("Running DFS...")
        start_time = time.time()
        while len(self.f) > 0:
            state = self.f.pop()
            if not state.is_finished():
                # Expando
                for i in self.available_colours:
                    if i != state.current_colour:
                        newState = copy.deepcopy(state)
                        newState.current_colour = i
                        newState.paint(i)
                        newState.moves_made += 1
                        newState.path.append(state.current_colour)
                        if newState not in self.v:
                            self.f.append(newState)
                            self.v.append(newState)
            else:
                state.path.append(state.current_colour)
                print("--- GANASTE ---")
                t = (time.time() - start_time)
                print("Tiempo: %s seconds" % t)
                print(f"Nodos visitados: {len(self.v)}")
                print(f"Nodos frontera: {len(self.f)}")
                print(f"Camino Solucion: {state.path}")
                print(f"Costo de la solucion: {len(state.path)} Movimientos")
                print("---------------")
                return t, len(self.v),len(self.f), len(state.path)


class Greedy:
    def __init__(self, fillzone):
        self.fillzone = fillzone
        # Nodos visitados
        self.v = []
        # Nodos frontera
        self.f = [fillzone.get_state()]
        # Array con la cantidad de colores disponible
        self.available_colours = list(range(0, fillzone.state.colours))

    def run(self):
        print("Running Greedy...")
        start_time = time.time()
        while len(self.f) > 0:
            state = self.f.pop(0)
            if not state.is_finished():
                for i in self.available_colours:
                    if i != state.current_colour:
                        newState = copy.deepcopy(state)
                        newState.current_colour = i
                        newState.paint(i)
                        newState.moves_made += 1
                        newState.path.append(state.current_colour)
                        newState.heuristic = self.fillzone.run_heuristic(newState)
                        if newState not in self.v:
                            self.f.append(newState)
                            self.v.append(state)
                # self.f.insert(idx+1,next_state)
            else:
                state.path.append(state.current_colour)
                print("--- GANASTE ---")
                t = (time.time() - start_time)
                print("Tiempo: %s seconds" % t)
                print(f"Nodos visitados: {len(self.v)}")
                print(f"Nodos frontera: {len(self.f)}")
                print(f"Camino Solucion: {state.path}")
                print(f"Costo de la solucion: {len(state.path)} Movimientos")
                print("---------------")
                return t, len(self.v),len(self.f), len(state.path)
            self.f.sort(key=lambda state: state.heuristic)
                

class A:
    def __init__(self, fillzone):
        self.fillzone = fillzone
        # Nodos visitados
        self.v = []
        # Nodos frontera
        self.f = [fillzone.get_state()]
        # Array con la cantidad de colores disponible
        self.available_colours = list(range(0, fillzone.state.colours))

    def run(self):
        print("Running A*...")
        start_time = time.time()
        while len(self.f) > 0:
            state = self.f.pop(0)
            if not state.is_finished():
                for i in self.available_colours:
                    if i != state.current_colour:
                        newState = copy.deepcopy(state)
                        newState.current_colour = i
                        newState.paint(i)
                        newState.moves_made += 1
                        newState.path.append(state.current_colour)
                        newState.heuristic = self.fillzone.run_heuristic(newState)
                        if newState not in self.v:
                            self.f.append(newState)
                            self.v.append(state)
            else:
                state.path.append(state.current_colour)
                print("--- GANASTE ---")
                t = (time.time() - start_time)
                print("Tiempo: %s seconds" % t)
                print(f"Nodos visitados: {len(self.v)}")
                print(f"Nodos frontera: {len(self.f)}")
                print(f"Camino Solucion: {state.path}")
                print(f"Costo de la solucion: {len(state.path)} Movimientos")
                print("---------------")
                return t, len(self.v),len(self.f), len(state.path)
            self.f.sort(key=lambda state: (state.moves_made + state.heuristic, state.heuristic))
