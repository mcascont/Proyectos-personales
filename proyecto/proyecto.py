import verosimilitud as vero
import Pattern_search as ps
import newton as nt
import autograd.numpy as np
import pandas as pd
from autograd import *
"""
    este es mi proyecto principal 
"""
# Cargar los datos desde el archivo CSV
file_path = r'C:\Users\marur\Desktop\Proyecto numerica\proyecto\heart_failure_clinical_records_dataset.csv'
data = pd.read_csv(file_path)

# Seleccionar las características y la etiqueta
caracteristicas_x = ['age', 'ejection_fraction', 'serum_creatinine', 'serum_sodium']
valor_y = 'DEATH_EVENT'

X = data[caracteristicas_x].values
y = data[valor_y].values

# Inicialización de los parámetros y las direcciones
w0 = np.zeros(X.shape[1]) # array de longuitd n = len(X) donde todos los elementos son 0
directions = [np.eye(X.shape[1])[i] for i in range(X.shape[1])] 
#genera vectores linealmente independientes "vectores identidad" nx1 donde n es el numero de columnas de la matriz X

# Función objetivo
def fo(w):
    return -(vero.log_likelihood(w, X, y))

# Obtener valores iniciales w usando Hooke y Jeeves
initial_w = ps.hooke_jeeves(fo, w0, directions, alpha=0.5, epsilon=1e-6)
print("Parámetros iniciales encontrados por Hooke y Jeeves:\n", initial_w)
print("\n")   
# obtener los valores finales w usando el metodo de newton 
valores = []
alpha_opt = 0
final_w, alpha_opt,valores= nt.newtonop(fo,initial_w,tol=1e-6, max_iter=100)
print("Parámetros finales encontrados por newton:\n", final_w ,"\n")
print("valores log-like:", valores)
print("stepsize:",alpha_opt)

#recorrer la matriz y usar los coeficientes de regresion para calcular y_estimado
y_estimado = np.array([])  
result_y = np.array([])
#recorro la matriz X para multiplicarlo con el vector W 
for fila in X:
    y_probabilidad = vero.hypothesis(final_w, fila) #obtengo la probabilidad de y_estimado 
    y_estimado = np.append(y_estimado, y_probabilidad)  # Agregar la probabilidad al array y_estimado
    # Convertir y_estimado a valores binarios (0 o 1) usando un umbral de 0.5
    y_estimado_bin = (y_probabilidad >= 0.5).astype(int) # Resulta en: array([True, False, True, False]) y luego los convierte en 0 y 1 
    result_y = np.append(result_y, y_estimado_bin).astype(int) # Agregar el resultado binario al array result_y
print("\n")
print("los valores de Y observado son:")
print(y)
print("\n")
print("los valores de Y estimado son:")   
print(result_y)
# Comparar result_y con y observado
print("\n")
accuracy = np.mean(result_y == y)
print("Exactitud del modelo:", round(accuracy*100,2),"%")


    
    
    

