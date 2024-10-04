import autograd.numpy as np

def hooke_jeeves(f, x0, directions, alpha=0.5, epsilon=1e-6):
    """
    Implementación del método de búsqueda de Hooke y Jeeves.
    Args:
        f (callable): La función objetivo a minimizar.
        x0 (np.array): El punto inicial.
        directions (list of np.array): Las direcciones de búsqueda linealmente independientes.
        alpha (float): El factor de amortiguamiento.
        epsilon (float): El parámetro de precisión.
    Returns:
        np.array: El punto óptimo encontrado.
    """
    def pattern_search(f, y, directions):
        """
        Subrutina de búsqueda por patrones.
        Explora los puntos en las direcciones dadas y encuentra la mejor dirección.
        """
        for j in range(len(directions)):
            # Explora en la dirección positiva
            if f(y + directions[j]) < f(y):
                y = y + directions[j]
            # Si no mejora, explora en la dirección negativa
            elif f(y - directions[j]) < f(y):
                y = y - directions[j]
        return y
    
    k = 0  # Inicializa el contador de iteraciones
    xk = x0  # Punto inicial
    max_iter = 50
    while k < max_iter:  # Bucle principal del algoritmo
        y = xk  # Establece y en el punto actual
        y = pattern_search(f, y, directions)  # Llama a la subrutina de búsqueda por patrones 
        if np.all(y == xk):  # Si no se encuentra mejora en la búsqueda por patrones , np.all es para comparar arreglos
            directions = [alpha * d for d in directions]  # Reduce el tamaño de las direcciones de búsqueda, d son los vectores que se encuentra en el conjunto directions
            if np.linalg.norm(directions) < epsilon:  # Comprueba el criterio de parada
                #np.linalg calcula la norma (longitud) de un vector o, en este caso, de una lista de vectores.
                k = max_iter  # Si las direcciones son lo suficientemente pequeñas, detiene el algoritmo
        else:
            z = y  # Guarda el nuevo punto encontrado, y es el valor encontrado por ps
            y = y + (y - xk)  # Realiza un movimiento exploratorio, compara la diferencia del valor encontrado por PS con el punto incial x0, se  se está proyectando un nuevo punto más allá de y en la dirección de mejora.
            y = pattern_search(f, y, directions)  # Llama nuevamente a la subrutina de búsqueda por patrones
            if f(y) < f(z):  # Si el nuevo punto es mejor, si el y actual es mejor que el y anterior "z"
                z = y  # Actualiza z con el valor del "y actual"
            if f(z) < f(xk):  # Si z es mejor que el punto actual xk=x0
                xk = z  # Actualiza el punto actual a z
            else:
                xk = y  # Si no, establece el punto actual como y
        k += 1  # Incrementa el contador de iteraciones
    return xk  # Devuelve el punto óptimo encontrado

