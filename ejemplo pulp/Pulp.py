# Importamos la librería PuLP que nos permitirá trabajar con programación lineal
import pulp

# Crear el problema de minimización
# Aquí definimos un problema de optimización llamado "Minimizar_costos", cuyo objetivo es minimizar (pulp.LpMinimize)
mi_problema = pulp.LpProblem("Minimizar_costos", pulp.LpMinimize)

# Crear las variables de decisión
# Definimos dos variables continuas, Producto_A y Producto_B, que representan las cantidades a producir de cada producto.
# lowBound=0 significa que no pueden tomar valores negativos.
A = pulp.LpVariable('Producto_A', lowBound=0, cat='Continuous')
B = pulp.LpVariable('Producto_B', lowBound=0, cat='Continuous')

# Definir la función objetivo: Minimizar el costo total
# La función objetivo es 40 * A + 30 * B, que representa los costos asociados con la producción.
# El objetivo es minimizar este valor. Aquí, "Costo_total" es solo una etiqueta descriptiva.
mi_problema += 40 * A + 30 * B, "Costo_total"

# Definir las restricciones
# Restricción 1: 2A + B <= 80 (limitación de tiempo de producción disponible).
# Esto significa que la combinación de Producto A y Producto B no puede exceder 80 unidades de tiempo de producción.
mi_problema += 2 * A + B <= 80, "Restriccion_de_tiempo"

# Restricción 2: A >= 10 (se deben producir al menos 10 unidades del Producto A).
mi_problema += A >= 10, "Restriccion_min_A"

# Restricción 3: B >= 5 (se deben producir al menos 5 unidades del Producto B).
mi_problema += B >= 5, "Restriccion_min_B"

# Resolver el problema
# Este comando le dice a PuLP que resuelva el problema de optimización utilizando el solver predeterminado.
mi_problema.solve()
# dentro del solve() podermos escribir el solver que queremos usar, si esta vacio se usa CBC por defecto

# Mostrar el estado del problema
# Verificamos el estado del problema para saber si se ha encontrado una solución óptima, no factible, o si no se ha podido resolver.
print("Estado:", pulp.LpStatus[mi_problema.status])

# Mostrar los valores óptimos de las variables
# Una vez resuelto el problema, imprimimos la cantidad óptima de Producto A y Producto B a producir para minimizar los costos.
print("Cantidad óptima de Producto A:", A.varValue)
print("Cantidad óptima de Producto B:", B.varValue)

# Mostrar el valor óptimo de la función objetivo (costo mínimo)
# Imprimimos el valor óptimo de la función objetivo, que es el costo mínimo calculado por el solver.
print("Costo total mínimo:", pulp.value(mi_problema.objective))

print(pulp.listSolvers(onlyAvailable=True)) # para ver que solvers puedes usar a aparte de CBC

