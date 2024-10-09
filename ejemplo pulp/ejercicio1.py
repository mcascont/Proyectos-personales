#   PuLP Tutorial
#agregar la funcionalidad de LO a mi programa
from pulp import *

#crear un objeto problema, esta instancia se llamara Practica 1
lp = LpProblem("Practica_1", LpMaximize)

#Variables de decision
#parametro cat: Integer,Binary or Continuous
#Continuous es la categoria por defecto
x1 = LpVariable(name="x1", lowBound=0, cat="Integer")
x2 = LpVariable(name="x2", lowBound=0, cat="Integer")


#Funcion Objetivo
lp += 10 * x1 + 5 * x2
print(lp.objective)

# 3 restricciones
lp += (5 * x1 + x2 <= 90, "restriccion_1")
lp += (x1 + 10 * x2 <= 300, "restriccion_2")
lp += (4 * x1 + 6 * x2 <= 125, "restriccion_3")
print(lp.constraints)

# Llamada al solver 
status = lp.solve(PULP_CBC_CMD(msg=0))

#POO lp es un objeto, tiene atributos y metodos
#solve es un metodo de este objeto
#una vez invocado se actualizan las colecciones o listas de variables


# Tipo  de salida del solver
print("Status:", status) #1 optima, 2:not solved, 3:infactible, 4:no acotado, 5:no definido

#valor de variables
for var in lp.variables():
    print(var, "=", value(var))
print("OPT =", value(lp.objective))
