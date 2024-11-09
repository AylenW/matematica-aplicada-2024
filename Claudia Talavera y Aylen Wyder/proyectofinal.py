import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.sentiment import SentimentIntensityAnalyzer
import numpy as np
import skfuzzy as fuzz
import time

# Descargar los recursos de NLTK
nltk.download('stopwords')
nltk.download('vader_lexicon')

# Carga del dataset
def leer_dataset():
    dataset = 'test_data.csv'
    df = pd.read_csv(dataset, encoding='latin-1')
    df.columns = ['tweet', 'label']
    df['label'] = df['label'].apply(lambda x: 'Positivo' if x == 1 else 'Negativo')
    return df[['tweet', 'label']]

# Limpieza del tweet
def pre_procesar(datos):
    contracciones = {
        "can't": "cannot", "won't": "will not", "n't": " not", "'s": " is",
        "'re": " are", "'ll": " will", "'ve": " have", "'d": " would", "'m": " am"
    }
    for contra, reemplazo in contracciones.items():
        datos = datos.replace(contra, reemplazo)
    datos = re.sub(r'http\S+|www\S+|https\S+', '', datos, flags=re.MULTILINE)
    datos = re.sub(r'@\w+', '', datos)
    datos = re.sub(r'[^a-zA-Z\s#]', '', datos).lower()
    datos = re.sub(r'#', '', datos)
    palabras = datos.split()
    palabras = [p for p in palabras if p not in stopwords.words('english')]
    return ' '.join(palabras)

def Calcular_puntajes_VADER(datos):
    puntos = SentimentIntensityAnalyzer()
    puntajes = puntos.polarity_scores(datos)
    return pd.Series([puntajes["pos"], puntajes['neg'], puntajes['neu']])  # Incluimos el puntaje neutral

def funcion_membresia_triang(d, e, f, x):
    if x <= d:
        return 0.5
    elif d < x <= e:
        return (x - d) / (e - d)
    elif e < x <= f:
        return (f - x) / (f - e)
    else:
        return 0.5

# Fuzzificación de los puntajes
def fuzzificar_puntajes(puntaje_pos, puntaje_neg, puntaje_neu, min_pos, max_pos, min_neg, max_neg, min_neu, max_neu):
    mid_pos = (min_pos + max_pos) / 2
    mid_neg = (min_neg + max_neg) / 2
    mid_neu = (min_neu + max_neu) / 2

    pos_bajo = funcion_membresia_triang(min_pos, min_pos, mid_pos, puntaje_pos)
    pos_medio = funcion_membresia_triang(min_pos, mid_pos, max_pos, puntaje_pos)
    pos_alto = funcion_membresia_triang(mid_pos, max_pos, max_pos, puntaje_pos)

    neg_bajo = funcion_membresia_triang(min_neg, min_neg, mid_neg, puntaje_neg)
    neg_medio = funcion_membresia_triang(min_neg, mid_neg, max_neg, puntaje_neg)
    neg_alto = funcion_membresia_triang(mid_neg, max_neg, max_neg, puntaje_neg)

    neu_bajo = funcion_membresia_triang(min_neu, min_neu, mid_neu, puntaje_neu)
    neu_medio = funcion_membresia_triang(min_neu, mid_neu, max_neu, puntaje_neu)
    neu_alto = funcion_membresia_triang(mid_neu, max_neu, max_neu, puntaje_neu)

    return {
        'pos_fuzzy': {
            'bajo': pos_bajo,
            'medio': pos_medio,
            'alto': pos_alto
        },
        'neg_fuzzy': {
            'bajo': neg_bajo,
            'medio': neg_medio,
            'alto': neg_alto
        },
        'neu_fuzzy': {
            'bajo': neu_bajo,
            'medio': neu_medio,
            'alto': neu_alto
        }
    }

# Evaluación de reglas y agregación
def evaluar_reglas_y_agregar(fuzzy_scores):
    # Definir las activaciones de las reglas basadas en la combinación lógica (AND)
    w_r1 = min(fuzzy_scores['pos_fuzzy']['bajo'], fuzzy_scores['neg_fuzzy']['bajo'])
    w_r2 = min(fuzzy_scores['pos_fuzzy']['medio'], fuzzy_scores['neg_fuzzy']['bajo'])
    w_r3 = min(fuzzy_scores['pos_fuzzy']['alto'], fuzzy_scores['neg_fuzzy']['bajo'])
    w_r4 = min(fuzzy_scores['pos_fuzzy']['bajo'], fuzzy_scores['neg_fuzzy']['medio'])
    w_r5 = min(fuzzy_scores['pos_fuzzy']['medio'], fuzzy_scores['neg_fuzzy']['medio'])
    w_r6 = min(fuzzy_scores['pos_fuzzy']['alto'], fuzzy_scores['neg_fuzzy']['medio'])
    w_r7 = min(fuzzy_scores['pos_fuzzy']['bajo'], fuzzy_scores['neg_fuzzy']['alto'])
    w_r8 = min(fuzzy_scores['pos_fuzzy']['medio'], fuzzy_scores['neg_fuzzy']['alto'])
    w_r9 = min(fuzzy_scores['pos_fuzzy']['alto'], fuzzy_scores['neg_fuzzy']['alto'])

    # Activación de sentimientos: Negativo, Neutro, Positivo
    w_negativo = min(w_r4, w_r7, w_r8)
    w_neutro = min(w_r1, w_r5, w_r9)
    w_positivo = min(w_r2, w_r3, w_r6)

    # Funciones de membresía de salida para cada tipo de sentimiento
    op_neg = fuzz.trimf(np.arange(0, 10, 0.1), [0, 0, 5])
    op_neu = fuzz.trimf(np.arange(0, 10, 0.1), [0, 5, 10])
    op_pos = fuzz.trimf(np.arange(0, 10, 0.1), [5, 10, 10])

    # Agregar las activaciones
    op_activation_low = np.minimum(w_negativo, op_neg)
    op_activation_med = np.minimum(w_neutro, op_neu)
    op_activation_positivo = np.minimum(w_positivo, op_pos)

    # Unión (max) para la salida agregada
    op_aggregated = np.maximum(np.maximum(op_activation_low, op_activation_med), op_activation_positivo)

    return op_aggregated

# Defuzzificación usando el método del centroide
def defuzzificar(op_aggregated):
    z_range = np.arange(0, 10, 0.1)
    if np.sum(op_aggregated) != 0:
        centroid = np.sum(z_range * op_aggregated) / np.sum(op_aggregated)
    else:
        centroid = np.mean(z_range)  # Promedio si la agregación es cero
    return centroid

# Clasificación del sentimiento basado en el valor de centroide
def clasificar_sentimiento(centroid_value):
    if centroid_value < 3.33:
        return 'Negativo'
    elif centroid_value < 6.66:
        return 'Neutro'
    else:
        return 'Positivo'

def calcular_senti(fila):
    min_pos = 0
    max_pos = 1
    min_neg = 0
    max_neg = 1
    min_neu = 0
    max_neu = 1

    # Tiempo de fuzzificación
    inicio_fuzz = time.time()

    # Fuzzificar puntajes
    fuzzy_scores = fuzzificar_puntajes(fila['puntaje positivo'], fila['puntaje negativo'], fila['puntaje neutral'], min_pos, max_pos, min_neg, max_neg, min_neu, max_neu)

    # Tiempo de fuzzificación
    tiempo_fuzz = time.time() - inicio_fuzz

    # Tiempo de defuzzificación
    inicio_defuzz = time.time()

    # Defuzzificación para obtener valor crisp
    sentimiento_crisp = defuzzificar(evaluar_reglas_y_agregar(fuzzy_scores))

    # Tiempo de defuzzificación
    tiempo_defuzz = time.time() - inicio_defuzz

    # Clasificación del sentimiento
    sentimiento_clasificado = clasificar_sentimiento(sentimiento_crisp)

    return sentimiento_clasificado, sentimiento_crisp, tiempo_fuzz, tiempo_defuzz

def Aplic_Modulos():
    df = leer_dataset()
    df['tweet'] = df['tweet'].apply(pre_procesar)
    df[['puntaje positivo', 'puntaje negativo', 'puntaje neutral']] = df['tweet'].apply(Calcular_puntajes_VADER)

    # Crear columnas para puntaje de sentimiento, clasificación y tiempos
    df['sentimiento clasificado'], df['valor sentimiento'], df['tiempo fuzzificación'], df['tiempo defuzzificación'] = zip(*df.apply(calcular_senti, axis=1))

    # Calcular tiempo total por tweet
    df['tiempo total'] = df['tiempo fuzzificación'] + df['tiempo defuzzificación']

    # Exportar el DataFrame con los resultados
    df.to_csv('resultados_sentimientos.csv', index=False)

    # Mostrar estadísticas
    print(f"Número de Tweets Positivos: {df['sentimiento clasificado'].value_counts().get('Positivo', 0)}")
    print(f"Número de Tweets Negativos: {df['sentimiento clasificado'].value_counts().get('Negativo', 0)}")
    print(f"Número de Tweets Neutros: {df['sentimiento clasificado'].value_counts().get('Neutro', 0)}")
    print(f"Tiempo total promedio de fuzzificación: {df['tiempo fuzzificación'].mean()} segundos")
    print(f"Tiempo total promedio de defuzzificación: {df['tiempo defuzzificación'].mean()} segundos")
    print(f"Tiempo total promedio por tweet: {df['tiempo total'].mean()} segundos")

Aplic_Modulos()