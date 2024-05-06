# Mariline Catalina Delgado Martínez
# Analítica de Datos
# Mayo 2024
#%% Librerias
import mlflow
import pandas as pd

#%% Inicializar MLflow
mlflow.set_tracking_uri("content")
mlflow.set_experiment("Modelos")

#%% Cargar datos
db = pd.read_csv('content/validation/Womens Clothing E-Commerce Reviews.csv')

#%%
db=db[['Clothing ID', 'Age', 'Review Text', 'Class Name']]
db['Clothing ID']=db['Clothing ID'].astype(str)
p_nulos = (db.isnull().sum() / len(db)) * 100
unique = db.nunique()
types = db.dtypes
pd.DataFrame({'% valores nulos': p_nulos, 'valores unicos': unique, 'Tipo': types})


# %% Creación de datos en streamlit
import seaborn as sns
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest

modelo = IsolationForest(contamination=0.05) # Proporción de valores atípicos esperados en el conjunto de datos
modelo.fit(db[['Age']])
valores_atipicos = modelo.predict(db[['Age']]) # Identificar valores atípicos
db['anomalia'] = valores_atipicos # Añadir una columna 'anomalia' al DataFrame que indica si cada fila es una anomalía o no
outliers = db[db['anomalia'] == -1]
#db = db[db['anomalia'] !=-1]         

#%%
import nltk
from nltk.tokenize import word_tokenize

# Descargar recursos adicionales de NLTK (si es necesario)
nltk.download('punkt')
data = db.dropna(subset=['Review Text'])
# Calcular la longitud de cada revisión
data['Review Length'] = data['Review Text'].apply(lambda x: len(word_tokenize(x)))
#%% Realizar análisis de sentimientos
# (Se requiere un modelo de análisis de sentimientos previamente entrenado)

from nltk.corpus import stopwords
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# Descargar recursos adicionales de NLTK (si es necesario)
nltk.download('stopwords')
nltk.download('vader_lexicon')
# Preprocesamiento del texto
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    tokens = word_tokenize(text.lower())  # Tokenización y conversión a minúsculas
    tokens = [token for token in tokens if token.isalpha()]  # Eliminar signos de puntuación y números
    tokens = [token for token in tokens if token not in stop_words]  # Eliminar palabras vacías (stopwords)
    return tokens

# Análisis de frecuencia de palabras
def word_frequency(texts):
    all_words = [word for text in texts for word in text]
    freq_dist = nltk.FreqDist(all_words)
    return freq_dist

# Análisis de sentimientos
def sentiment_analysis(texts):
    sid = SentimentIntensityAnalyzer()
    sentiment_scores = [sid.polarity_scores(text)['compound'] for text in texts]
    return sentiment_scores

# Aplicar preprocesamiento y análisis de sentimientos a las revisiones
data['Processed Text'] = data['Review Text'].apply(preprocess_text)
data['Sentiment Score'] = sentiment_analysis(data['Review Text'])

#%% #Sentiment Intensity Analyzer

from nltk.sentiment.vader import SentimentIntensityAnalyzer

sid = SentimentIntensityAnalyzer() # Crear una instancia del analizador de sentimientos VADER

# Calcular el puntaje de sentimiento para cada revisión
data['Sentiment Score'] = data['Review Text'].apply(lambda x: sid.polarity_scores(x)['compound'])

# Clasificar las revisiones en positivas, negativas o neutrales
data['Sentiment'] = data['Sentiment Score'].apply(lambda x: 'Positiva' if x > 0 else ('Negativa' if x < 0 else 'Neutral'))

# Contar la cantidad de revisiones en cada categoría de sentimiento
sentiment_counts = data['Sentiment'].value_counts()

#%% Configuración de la página
# Establecer el tema con fondo blanco
st.set_page_config(layout="wide", 
                   page_title="Analisis de Datos", 
                   page_icon="📊", 
                   initial_sidebar_state="expanded")

st.header('Analisis de Sentimientos')

# Sidebar
st.sidebar.header('Analisis de Datos')
variable = st.sidebar.radio(
    'Selecciona una Opción',
    options=['Valores Atipicos',
            'Distribución de Frecuencias', 
            'Análisis de Sentimientos'])

st.sidebar.header('Detalles de la Base de Datos')
st.sidebar.write('Número de Registros:', db.shape[0])
st.sidebar.subheader('Columnas:')
st.sidebar.write(db.columns)

#%% Función para mostrar la opción seleccionada
def mostrar_opcion(opcion, db, data):
    #%%Valores atípicos detectados en edad
    if opcion == 'Valores Atipicos':
      
        st.write("Valores atípicos detectados en edad")
        # Visualización de los valores atípicos
        fig, ax = plt.subplots(figsize=(8, 6))
        scatter=ax.scatter(range(len(db)), db['Age'], c=valores_atipicos, cmap='viridis')
        ax.set_title('Detección de anomalías con Isolation Forest')
        ax.set_xlabel('Índice')
        ax.set_ylabel('Columna')
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Anomalía')
        st.pyplot(fig)
        
        st.write('Porcentaje de Valores atípicos detectados: ', (len(outliers) / len(db)) * 100)
    #%% Distribución de Frecuencias
    elif opcion == 'Distribución de Frecuencias':
        
        # Lista de opciones de pestaña
        tabs = ['Frecuencia de Clases', 
                'Frecuencia de Edad',
                'Frecuencia de Longitud de Revisiones']

        # Seleccionar la pestaña
        selected_tab = st.selectbox('Selecciona una pestaña:', tabs)

        #%%Frecuencia de Clases
        if selected_tab == 'Frecuencia de Clases':
            
            columna_1, columna_2 = st.columns([2,1])

            with columna_1:
                fig, ax = plt.subplots(figsize=(10, 8))
                db.groupby('Class Name').sum().sort_values('Age', ascending=True)['Age'].plot(kind='barh', color=sns.color_palette('Dark2', len(db['Class Name'].unique())), ax=ax)
                plt.gca().spines[['top', 'right']].set_visible(False)
                plt.title('Frecuencias por Clase')
                plt.xlabel('Frecuencia')
                plt.ylabel('Clase')
                st.pyplot(fig)

            with columna_2:
                # Mostrar estadísticas descriptivas de la columna 'Age'
                st.write('Estadísticas descriptivas de las Clases:')
                st.write(db['Class Name'].describe())

        #%%Frecuencia de edad
        elif selected_tab == 'Frecuencia de Edad':
            
            columna_1, columna_2 = st.columns([2,1])

            with columna_1:
                fig, ax = plt.subplots(figsize=(9, 8))
                sns.histplot(db['Age'], ax=ax)
                plt.title('Frecuencias de Edad')
                plt.xlabel('Edad')
                plt.ylabel('Frecuencia')
                st.pyplot(fig)
                
                fig, ax = plt.subplots(figsize=(10, 6))
                sns.boxplot(x='Class Name', y='Age', data=db, palette='Set2')
                plt.title('Distribución de Edad por Categoría de Ropa')
                plt.xlabel('Categoría de Ropa')
                plt.ylabel('Edad')
                plt.xticks(rotation=90, ha='right')
                plt.tight_layout()
                st.pyplot(fig)

            with columna_2:
                # Mostrar estadísticas descriptivas de la columna 'Age'
                st.write('Estadísticas descriptivas de la edad:')
                st.write(db['Age'].describe())
        #%%Frecuencia de Longitud de Revisiones       
        elif selected_tab == 'Frecuencia de Longitud de Revisiones':
            
            columna_1, columna_2 = st.columns([2,1])

            with columna_1:
                fig, ax = plt.subplots(figsize=(10, 8))
                data['Review Length'].plot(kind='hist', bins=20, color='skyblue')
                plt.gca().spines[['top', 'right',]].set_visible(False)
                plt.title('Distribución de longitud de revisiones')
                plt.xlabel('Longitud de la revisión')
                plt.ylabel('Frecuencia')
                st.pyplot(fig)

            with columna_2:
                # Mostrar estadísticas descriptivas de la columna 'Age'
                st.write('Estadísticas descriptivas de las Revisiones:')
                st.write(data['Review Length'].describe())

    #%% Análisis de Sentimientos
    elif opcion == 'Análisis de Sentimientos':
        
        st.write('Distribución de los sentimientos por categoría de ropa')
                
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.boxplot(x='Class Name', y='Sentiment Score', data=data, palette='Set2')
        plt.title('Distribución de Sentimientos por Categoría de Ropa')
        plt.xlabel('Categoría de Ropa')
        plt.ylabel('Puntuación de Sentimiento')
        plt.xticks(rotation=90, ha='right')
        plt.tight_layout()
        st.pyplot(fig)
        
        st.write('Clasificación de Sentimientos')
        
        columna_1, columna_2 = st.columns([2,1])

        with columna_1:
            
            fig, ax = plt.subplots(figsize=(10, 8))
            plt.pie(sentiment_counts, labels=sentiment_counts.index, autopct='%1.1f%%', colors=['lightgreen', 'lightcoral', 'lightskyblue'])
            plt.title('Clasificación por Grupos de Sentimiento')
            plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
            plt.show()
            st.pyplot(fig)

        with columna_2:
         
            st.write("Clasificación por grupos:")
            st.write(sentiment_counts)
        
        
#%% Mostrar la opción seleccionada
mostrar_opcion(variable, db, data)