# TP1-SIA

### Alumnos:
* German Ariel Martinez - 58574
* Joaquin Garcia Del Rio - 59051
* Santiago Preusche - 59233
* Franco Navarro - 59055

### Requerimientos:
* Python3

# Lado A - Métodos de Búsqueda en el juego Fillzone

## Configuración de los parámentros
```
1. Colors -> [4,8] Un numero entre 4 y 8 que representa la cantidad de colores distintos presentes en el juego.
2. Size -> [3,10] El tamaño que va a tener el tablero, siendo de NxN.
3. Heuristic -> [1,2,3] Determina que heurística se utilizara, donde:
    1. (No accesible) Cantidad de cuadrados que faltan pintar.
    2. (Accesible) Cantidad de colores distintos que hay en el área no pintada del tablero.
    3. (Accesible) Cantidad de colores distintos que son vecinos al área pintada del tablero.
4. SearchMethod -> [1,2,3,4] Determina que método de búsqueda se utilizara, donde:
    1. BFS
    2. DFS
    3. Greedy
    4. A*
```
## Instrucciones para ejecutar el programa
Posicionarse en la carpeta LadoA y ejecutar el siguiente comando:
```
python3 Main.py -c [colors] -s [N] -he [heuristic] -sm [search metod]
```
### Ejemplo:
Con el siguiente comando se crea un tablero aleatorio de 5x5 con 5 colores distintos. Utilizara la heuristica 2 y el algoritmo A* para encontrar la solucion.
```
python3 Main.py -c 5 -s 5 -he 2 -sm 4
```
# Lado B - Algoritmos Genéticos para buscar

## Configuración de los parámentros
Para configurar los parametros que usara el programa, se usa un archivo JSON ubicado en la carpeta "LadoB" llamado "params.json". El cual tiene la siguiente estructura:
```
{
    "K": 4,
    "objective": {
        "red": "(Integer) [0, 255] Valor RGB de rojo",
        "green": "(Integer) [0, 255] Valor RGB de verde",
        "blue": "(Integer) [0, 255] Valor RGB de azul"
    },
    "palette": [{
        "red": "(Integer) [0, 255] Valor RGB de rojo",
        "green": "(Integer) [0, 255] Valor RGB de verde",
        "blue": "(Integer) [0, 255] Valor RGB de azul"
    }],
    "selection": 
        {
            "name": "Probabilistic Tourney",
            "threshold": "(Float) (0,1) Threshold de selección"
        } | {
            "name": "Roulette | Elite"
        }   
}
```
## Instrucciones para ejecutar el programa
Posicionarse en la carpeta LadoA y ejecutar el siguiente comando:
```
python3 Main.py
```