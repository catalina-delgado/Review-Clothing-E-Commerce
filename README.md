## Project Review from Clothing E-Commerce

### Content
## 1. Streamlit App: https://riview-clothing-e-commerce-agpddca2qrn8effpa9ptjk.streamlit.app/

Permite visualizer el análisis exploratorio de los datos mediante una configuración de página con slider. En este se encuentran tres opciones para checkear de acuerdo al tipo de análisis realizado a la base de datos.

1.	Valores atípicos: la aplicación inicializa con la visualización utilizando random forest de anomalías para la variable edad. De tal manera que se identifique los registros que están fuera del rango promedio de esta variable.
2.	Distribución de Frecuencias: permite visualizar la distribución de frecuencias para las variables “Class Name”, “Age”, “Review Text”
3.	Análisis de Sentimientos: permite visualizar la clasificación de los sentimientos que genera cada producto en los compradores de acuerdo a un análisis de puntuación de sentimientos, donde 1 es un sentimiento positivo, 0 un sentimiento neutro y -1 un sentimiento negativo.

## 2. MLflow Analisis
Registro de las métricas de entrenamiento de un modelo de regresión logística para la predicción de clases de ropa, con un 80% de los datos para el entrenamiento y un 20% de los datos para los datos de prueba. El modelo es entrenado con 1000 iteraciones y entrega un total de 4334 de clases predichas y un valor de precisión del 0.27.