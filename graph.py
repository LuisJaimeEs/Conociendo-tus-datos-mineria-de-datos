import pandas as pd
import matplotlib.pyplot as plt

# Leer el archivo CSV
df = pd.read_csv('mushrooms.csv')

# Obtener los datos de una columna específica
columna = 'nombre_columna'
datos = df[columna]

# Graficar histograma
plt.hist(datos, bins=30, density=True, alpha=0.5, color='steelblue')
plt.xlabel('Valores')
plt.ylabel('Densidad')
plt.title('Distribución de datos')
plt.grid(True)

# Mostrar el gráfico
plt.show()