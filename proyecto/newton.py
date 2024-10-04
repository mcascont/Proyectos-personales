import autograd.numpy as np  # Importa numpy con soporte para autograd, que permite la diferenciación automática
from autograd import *  # Importa las funciones grad y hessian de autograd para calcular gradientes y matrices Hessianas
#metodo para 
def gs(a,b,f,gr,tol,maxit):
    d= (b-a)*gr
    x1 = b - d 
    x2 = a + d
    k = 1
    while(k <= maxit and abs(x1-x2)> tol):
        if(f(x1) < f(x2)):
            b = x2 
        else:
            a = x1
        if(abs(f(x1) - f(x2)) <= tol):
            k = maxit
        d= (b-a)*gr
        x1 = b - d 
        x2 = a + d
        k += 1     
    return (x1 + x2) / 2

# Método de Newton para minimización
def newtonop(f, x_init, tol, max_iter):
    # Lista para guardar los valores de log-verosimilitud en cada iteración
    log_likelihood_history = []
    k=0
    x = x_init  # Inicializa el punto de partida
    x_anterior = x # Guarda el punto anterior (inicialmente igual al punto de partida) xk-1
    while(k <= max_iter):  # Bucle de iteración hasta el máximo de iteraciones
        gradient = grad(f)(x)  # Calcula el gradiente de la función en el punto actual
        hessian_matrix = hessian(f)(x)  # Calcula la matriz Hessiana de la función en el punto actual
        if np.linalg.norm(gradient) < tol:  # Comprueba si la norma del gradiente es menor que la tolerancia
            k = max_iter  # Si es así, detiene la iteración
        try:
            hessian_inv = np.linalg.inv(hessian_matrix)  # Intenta invertir la matriz Hessiana
        except np.linalg.LinAlgError:
            print("Hessiano no invertible")  # Si la Hessiana no es invertible, imprime un mensaje y detiene la iteración
            k = max_iter
        # Dirección de descenso
        sk = -np.dot(hessian_inv, gradient)
        
        # Función auxiliar para la Búsqueda Dorada
        def phi(alpha):
            return f(x + alpha * sk)
        
        # Obtener el stepsize óptimo usando Golden Search
        alpha_opt = gs(0, 1, phi, gr=0.618, tol=1e-6, maxit=100)
        stepsizeop = alpha_opt 
        
        # Actualizar el punto con el stepsize óptimo
        x_new = x + alpha_opt * sk
        #criterio de parada
        if np.linalg.norm(x_new - x_anterior) < tol:
            k = max_iter
        log_likelihood_history.append(f(x_new))#monitorear el log likelihood
        x_anterior = x  # Actualiza x_anterior al valor actual, en la siguiente iteracion seria xk anterior
        k += 1  # Incrementa el contador de iteraciones
    return x, stepsizeop,log_likelihood_history # Devuelve el punto óptimo encontrado




