from PIL import Image #oara lee coordenadas rgb
import numpy as np # para operaciones matriciales
import matplotlib.pyplot as plt # para los plots


def convertir_a_rgb(ruta_imagen):
    imagen =  Image.open(ruta_imagen)
    pixeles = list(imagen.getdata())
    print(len(pixeles))
    r_c = [pixel[0] for pixel in pixeles]
    g_c = [pixel[1] for pixel in pixeles]
    b_c = [pixel[2] for pixel in pixeles]
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(r_c, g_c, b_c, c='green', marker='x', alpha=0.8, s=1)
    ax.set_xlabel('R')
    ax.set_ylabel('G')
    ax.set_zlabel('B')
    ax.axis('equal') 
    
    
    return np.array((r_c, g_c, b_c))

def promedio(array):
    promedio = sum(array) / len(array)
    return promedio

def cov(X,Y):
    mean_X = promedio(X)
    mean_Y = promedio(Y)
    n = len(X)
    covariance = sum((X[i] - mean_X) * (Y[i] - mean_Y) for i in range(n)) / (n - 1)
    return covariance

def matriz_covarianza(array):
    n = len(array)
    matriz_covarianza = [[0 for _ in range(n)] for _ in range(n)]
    for i in range(len(array)):
        for j in range(len(array)):
            matriz_covarianza[i][j]=cov(array[i],array[j])
    print("la matriz de covarianza es")
    for fila in matriz_covarianza:
        print(fila)
    return np.array(matriz_covarianza)

def matriz_correlacion(array):
    n = len(array)
    matriz_cov = matriz_covarianza(array)
    matriz_correlacion = np.ones((n, n))
    
    for i in range(n):
        for j in range(n):
            if i!=j:
                std_i = np.std(array[i])
                std_j = np.std(array[j])
                matriz_correlacion[i][j] = matriz_cov[i][j] / (std_i * std_j)
    
    print("La matriz de correlación es:")
    for fila in matriz_correlacion:
        print(fila)
    
    return matriz_correlacion

def eigen(matriz):    
    eigenvalores, eigenvectores = np.linalg.eig(matriz.T)
    print("Eigenvalores:")
    print(eigenvalores)
    print("--------------------------------------")
    print("\nEigenvectores:")
    print(eigenvectores)
    suma_total_eigenvalores = np.sum(eigenvalores)
    proporcion_varianza = eigenvalores / suma_total_eigenvalores
    eigenvectores = eigenvectores.T
    print("--------------------------------------")
    for eigenvalor, prop in zip(eigenvectores, proporcion_varianza):
        print(f"\n Eigenvector {eigenvalor}| proporcion {prop:.4f}")
    return [eigenvalores, eigenvectores]

def centroide_plot(p):
    vector_prom = [promedio(p[0]), promedio(p[1]), promedio(p[2])]
    r,g,b=vector_prom
    origen = np.zeros(3)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.quiver(*origen, *[r,g,b],color="green", linestyle='-', linewidth=1, label='Vector promedio')
    ax.scatter(r,g,b, s=5, c='red', label="Centroide")
    ax.set_xlabel('R')
    ax.set_ylabel('G')
    ax.set_zlabel('B')
    ax.axis('equal') 
    ax.legend()
    

def ejes_principales(eigenvectores):
    origen = np.zeros(3)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    for eigenvector, color in zip(eigenvectores, ['#581845', '#C70039', '#FF5733']):
        ax.quiver(*origen, *eigenvector, color=color)
    ax.set_xlabel('R')
    ax.set_ylabel('G')
    ax.set_zlabel('B')
    ax.axis('equal') 
    

def plot_completo(eigenvectores, eigenvalores, array):
    x=array[0]
    y=array[1]
    z=array[2]
    vector_prom = [promedio(x), promedio(y), promedio(z)]
    r,g,b=vector_prom
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    origen = np.zeros(3)
    for eigenvector, color, eigenvalue in zip(eigenvectores, ['#581845', '#C70039', '#FF5733'], eigenvalores):
        eigenvector_shifted = eigenvector * eigenvalue + [r, g, b]  # Sumar el vector promedio
        ax.quiver(r, g, b, *eigenvector_shifted, color=color, label=f'Eigenvector {eigenvector}')

    ax.scatter(r,g,b, s=50, c='red')
    ax.scatter(x, y, z, c='#9AD0F1', marker='x', alpha=0.5, s=1, label='Datos originales')

    ax.set_xlabel('R')
    ax.set_ylabel('G')
    ax.set_zlabel('B')
    ax.axis('equal') 
    

pixeles=convertir_a_rgb('images.jpeg')

matriz_cov =matriz_covarianza(pixeles)

matriz_cor = matriz_correlacion(pixeles)

eigen_result = eigen(matriz_cov)

centroide_plot(pixeles)

ejes_principales(eigen_result[1])

plot_completo(eigen_result[1],eigen_result[0],pixeles)
plt.show()