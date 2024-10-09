#   PuLP Tutorial
#agregar la funcionalidad de LO a mi programa
from pulp import *

solver_list = listSolvers()
print(solver_list)

#******************************************************************************************************************************
#******************************************************************************************************************************
#Practica 2
#En caso de tener un mayor número de variables

lp = LpProblem("Practica_2", LpMaximize)

#ahora definiremos las variables utilizando índices

#var_keys contiene los índices de dos variables
#llenar esta lista dinamicamente posiblemente desde un archivo
var_keys = [1,2] 

#esta instrucción nos permite definir un diccionario de variables de decisión
#en base a var_keys  LpVariable.dicts retorna un diccionario
x = LpVariable.dicts("x_i", var_keys, lowBound=0, cat="Integer")
print(x)




#función objetivo
lp += 10* x[1] + 5 * x[2]

# añadimos las restricciones utilizando el diccionario x
lp += (x[1] + 10 * x[2] <= 300, "restriccion_2")
lp += (4 * x[1] + 6 * x[2] <= 125, "restriccion_3")


#Es posible que los coeficientes de las
#restricciones estén almacenados en un archivo externo: Base de datos
#SQL server, Access, MySql
coeff = [5, 1] # 5 , 1 puede ser una fila en una tabla de una base de datos


#zip intuitivamente arma parejas con los elementos en las listas
# var_keys y coeff
coeff_dict = dict( zip(var_keys,coeff) )
print(coeff_dict)


#restriccion_1
lp += ( lpSum(coeff_dict[i] * x[i] for i in var_keys) <= 90)



#solver
status = lp.solve(PULP_CBC_CMD(msg=0))

print("Status:", status)

#solucion
for var in lp.variables():
    print(var, "=", value(var))
print("OPT =", value(lp.objective))

print("CPLEX")

solver = CPLEX_CMD()

result = lp.solve(solver)

#solucion
for var in lp.variables():
    print(var, "=", value(var))
print("OPT =", value(lp.objective))


 