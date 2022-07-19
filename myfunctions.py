from sklearn.metrics import precision_recall_fscore_support
import pandas as pd
from sklearn.svm import SVC
from sklearn.metrics import roc_auc_score
import joblib
import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import RocCurveDisplay
from sklearn.datasets import load_wine
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn import svm
from sklearn.svm import SVC

from fim import *

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import random 
import graphviz
from graphviz import *
import time

#from pyspark.ml.fpm import FPGrowth

# ML Pkgs
# Load Models


def load_model(model_file):
    loaded_model = joblib.load(open(os.path.join(model_file), "rb"))
    return loaded_model


def get_key(val, my_dict):
    for key, value in my_dict.items():
        if val == value:
            return key

def test_model(model, df, nfols):  # Implementación de función para probar modelo con metricas
    X = df.iloc[:, :-1].values  # Caracteristicas
    Y = df.iloc[:, -1].values  # Columna objetivo
    skf = StratifiedKFold(n_splits=nfols)

    acuracy = 0
    precision = 0
    recall = 0
    f1 = 0

    # Cross-validation estratificado con k folds
    for train_index, test_index in skf.split(X, Y):
	    X_train, X_test = X[train_index], X[test_index]
	    Y_train, Y_test = Y[train_index], Y[test_index]

	    model.fit(X_train, Y_train)
	    predicciones = model.predict(X_test)

	    acuracy += accuracy_score(Y_test, predicciones)

	    precisionp, recallp, f1p, support = precision_recall_fscore_support(
	        Y_test, predicciones, pos_label=1, average='weighted', zero_division=0)

	    precision += precisionp
	    recall += recallp
	    f1 += f1p

    #aucm += roc_auc_score(Y_test, model.predict_proba(X_test, 1)[::,1], multi_class="ovr",)
    return acuracy/nfols, precision/nfols, recall/nfols, f1/nfols

def to_transactionnal(df,column_trans,column_items):
    transactions = []
    for v in df[column_trans].unique():
        transactions.append(list(df[df[column_trans] == v][column_items].values))

    return transactions

def supp(x, labels, trans):
    s = []
    for t in range(len(trans)):
        if set(x).issubset(set(trans[t])):
            s += [labels[t]]
    return s

def all_itemsets(trans_, supp_):
    #calcular los itemsets frecuentes con un soporte supp_ y reportando su 
    #frecuencia absoluta a y relativa S
    r = fpgrowth(trans_, target='a', supp=supp_, report='aS') 
    df_items = pd.DataFrame(r)#transformar en dataframe
    df_items.columns = ['Itemset', 'Freq', 'Freq(%)']#cambiar encabezados de las columnas
    df_items.sort_values(by='Freq',ascending=False,inplace=True)#ordenar los itemset por freq
    #ordenar los items de cada itemset para evitar confusion
    df_items['Size'] = [len(x) for x in df_items['Itemset'].tolist()]
    df_items['Itemset'] = [str(sorted(x)) for x in df_items['Itemset'].tolist()]
    return df_items

# ****************************************************** MODELO PATTERN MINING *******************************************************
def cargar_dataset():
    # Cargar Dataset:
    df_siembra = pd.read_excel("https://www.datosabiertos.gob.pe/node/6920/download")
    return df_siembra

def modificar_dataset(df_siembra):
    # *** Preprocesamiento del DATASET ***
    # Renombrar Columnas:
    df_siembra.rename(columns={'PROVINICA':'PROVINCIA'}, inplace=True)
    # Modificar Dataset:
    df_siembra_freq = df_siembra.copy()
    df_siembra_freq['DISTRITO'] = df_siembra_freq['DEPARTAMENTO']+'-'+df_siembra_freq['PROVINCIA']+'-'+df_siembra_freq['DISTRITO']
    # Eliminar Columnas Inecesarias:
    df_siembra_freq = df_siembra_freq[['DISTRITO','CULTIVO']]
    df_siembra_freq = df_siembra_freq.drop_duplicates()
    return df_siembra_freq

def to_transactionnal(df, column_trans, column_items):
    transactions = []

    for v in df[column_trans].unique():
        transactions.append(list(df[df[column_trans] == v][column_items].values))
    return transactions


def dataset_transactionnal(trans_):
    r = fpgrowth(trans_, target='c', supp=10, zmin=2)
    df = pd.DataFrame(r)
    df.columns = ['Itemset','Freq']
    df.sort_values(by='Freq',ascending=False,inplace=True)
    return df


def supp_(x, labels, trans):
    s = []
    for t in range(len(trans)):
        if set(x).issubset(set(trans[t])):
            s += [labels[t]]
    return s


def deteccion_supp(df, df_trans, trans_):
    labels = df["DISTRITO"].unique()
    df_trans_ = df_trans.copy()
    df_trans_["Supp"] = [supp_(x,labels,trans_) for x in df_trans_["Itemset"].tolist()]
    return df_trans_


def reglas_asociacion(trans_):

    # inputs
    supp = 2 # minimum support of an assoc. rule (default: 10)
    conf = 50 # minimum confidence of an assoc. rule (default: 80%)
    report = 'asC'

    # make dict for nicer looking column names
    report_colnames = {
        'a': 'support_itemset_absolute',
        's': 'support_itemset_relative',
        'S': 'support_itemset_relative_pct',
        'b': 'support_bodyset_absolute',
        'x': 'support_bodyset_relative',
        'X': 'support_bodyset_relative_pct',
        'h': 'support_headitem_absolute',
        'y': 'support_headitem_relative',
        'Y': 'support_headitem_relative_pct',
        'c': 'confidence',
        'C': 'confidence_pct',
        'l': 'lift',
        'L': 'lift_pct',
        'e': 'evaluation',
        'E': 'evaluation_pct',
        'Q': 'xx',
        'S': 'support_emptyset',
    }

    result = arules(trans_, supp=supp, conf=conf, report=report)
    # make df of results
    colnames = ['consequent', 'antecedent'] + [report_colnames.get(k, k) for k in list(report)]
    df_rules = pd.DataFrame(result, columns=colnames)
    df_rules = df_rules.sort_values('support_itemset_absolute', ascending=False)
    return df_rules

def itemsets_fpgrowth(trans_, supp_):
  #calcular los itemsets frecuentes con un soporte supp_ y reportando su 
  #frecuencia absoluta a y relativa S
  r = fpgrowth(trans_, target='a', supp=supp_, report='aS') 
  df_items = pd.DataFrame(r)#transformar en dataframe
  df_items.columns = ['Itemset - Productos', 'Frecuencia', 'Frecuencia (%)']#cambiar encabezados de las columnas
  df_items.sort_values(by='Frecuencia',ascending=False,inplace=True)#ordenar los itemset por freq
  #ordenar los items de cada itemset para evitar confusion
  df_items['Nro. Productos'] = [len(x) for x in df_items['Itemset - Productos'].tolist()]
  df_items['Itemset - Productos'] = [str(sorted(x)) for x in df_items['Itemset - Productos'].tolist()]
  return df_items

def itemsets_apriori(trans_, supp_):
  #calcular los itemsets frecuentes con un soporte supp_ y reportando su 
  #frecuencia absoluta a y relativa S
  r = apriori(trans_, target='a', supp=supp_, report='aS') 
  df_items = pd.DataFrame(r)#transformar en dataframe
  df_items.columns = ['Itemset', 'Freq', 'Freq(%)']#cambiar encabezados de las columnas
  df_items.sort_values(by='Freq',ascending=False,inplace=True)#ordenar los itemset por freq
  #ordenar los items de cada itemset para evitar confusion
  df_items['Size'] = [len(x) for x in df_items['Itemset'].tolist()]
  df_items['Itemset'] = [str(sorted(x)) for x in df_items['Itemset'].tolist()]
  return df_items

# ****************************************************** Definir Variables Globales *******************************************************
dataset = cargar_dataset()
data = modificar_dataset(dataset)
transaccional = to_transactionnal(data, 'DISTRITO', 'CULTIVO')
data_transactionnal = dataset_transactionnal(transaccional)
data_supports = deteccion_supp(data, data_transactionnal, transaccional)
data_reglas = reglas_asociacion(transaccional)

# Metricas FP-Growth y Apriori
start_fpgrowth = time.time()
data_itemsets_fpgrowth = itemsets_fpgrowth(transaccional, -1)
end_fpgrowth = time.time()
tiempo_fpgrowth = end_fpgrowth - start_fpgrowth

start_apriori = time.time()
data_itemsets_apriori = itemsets_apriori(transaccional, -1)
end_apriori = time.time()
tiempo_apriori = end_apriori - start_apriori

print(data_reglas.shape)

# ****************************************************** MAPA *******************************************************
import geopandas as gpd

url_geojson_distrital = "https://raw.githubusercontent.com/juaneladio/peru-geojson/master/peru_distrital_simple.geojson"
region_geojson = gpd.read_file(url_geojson_distrital)
region_geojson["NOMBDIST"] = region_geojson["NOMBDEP"]+'-'+region_geojson["NOMBPROV"]+'-'+region_geojson["NOMBDIST"]

from matplotlib.colors import ListedColormap
import io
import base64

my_palette = plt.cm.get_cmap("Set1",len(data_supports.index))

def index_departamentos(df_,row):
  l = df_.loc[df_.index[row]].values[[-1]].tolist()[0]
  r = []
  for dist in l:
    v = list(region_geojson[region_geojson["NOMBDIST"] == dist]["OBJECTID"].values)
    if len(v) > 0:
      r.append(v[0])
  w = []
  for i in range(len(region_geojson.index)):
    if i in r:
      w += [my_palette(row)]
    else:
      w += ["white"]
  return w

def draw_map(df_,row):
    cmap = ListedColormap(index_departamentos(df_,row), name='allred')
    ax = region_geojson.plot(figsize=(14,14), edgecolor=u'gray', cmap=cmap)
    #ax.annotate("dsa",xy=(-80,0))
    plt.ylabel('Latitude')
    plt.xlabel('Longitude')
    title = '\n'.join(str(df_[df_.columns[0]][df_.index[row]]).split(",")) + " " + str(df_[df_.columns[1]][df_.index[row]])
    plt.title(title, size=12,color=my_palette(row),y=1.01)
    #plt.show()
    return plt

def fig_to_base64(fig):
    img = io.BytesIO()
    fig.savefig(img, format='png',
                bbox_inches='tight')
    img.seek(0)
    fig.savefig('static/images/mapa.png')

# ****************************************************** PREDICCIÓN *******************************************************
def Reglas_Tabla_Ordenada(df_to_sort):
  
  df_to_sort = df_to_sort.sort_values(by='confidence_pct', ascending=False)

  con = df_to_sort['consequent'].to_numpy()
  ant = df_to_sort['antecedent'].to_numpy()
  sup_abs = df_to_sort['support_itemset_absolute'].to_numpy()
  sup_rel = df_to_sort['support_itemset_relative'].to_numpy()
  confi = df_to_sort['confidence_pct'].to_numpy()

  new_df_sort = pd.DataFrame(con, columns = ['consequent'])

  new_df_sort['antecedent'] = ant.tolist()
  new_df_sort['support_itemset_absolute'] = sup_abs.tolist()
  new_df_sort['support_itemset_relative'] = sup_rel.tolist()
  new_df_sort['confidence_pct'] = confi.tolist()

  return new_df_sort

def Seleccionar_Mejores_Reglas(porcentage, data_sort):
  index = 0 

  for k in range(len(data_sort['confidence_pct'])):
    if data_sort['confidence_pct'][k] < porcentage:
      index = k
      break
  
  if index > 1:
    df_sort_mejores = data_sort.head(index)
  else:
    df_sort_mejores = data_sort

  return df_sort_mejores


def Buscar_Todo(nro_filas, tag):
  df_sort_mejores = Reglas_Tabla_Ordenada(data_reglas)

  consequent = []
  antecedent = []
  confidence = []

  for k in range(len(df_sort_mejores['consequent'])):

    if tag == df_sort_mejores['consequent'][k]:
      consequent.append(df_sort_mejores['consequent'][k])
      antecedent.append(df_sort_mejores['antecedent'][k])
      confidence.append(df_sort_mejores['confidence_pct'][k])

    if len(confidence) > nro_filas:
      ghgh = len(confidence)
      break

  return consequent, antecedent, confidence


def Buscar_Reglas_Asociacion(tag, nro_filas, df_sort_mejores):
  consequent = []
  antecedent = []
  confidence = []

  for k in range(len(df_sort_mejores['consequent'])):

    if tag == df_sort_mejores['consequent'][k]:
      consequent.append(df_sort_mejores['consequent'][k])
      antecedent.append(df_sort_mejores['antecedent'][k])
      confidence.append(df_sort_mejores['confidence_pct'][k])

    if len(confidence) > nro_filas:
      ghgh = len(confidence)
      break
  
  mensaje = 'Resultados de la busqueda para el porcentaje deseado.'
  if consequent == []:
    mensaje = 'No se encontraron datos dentro del procentaje de busqueda. Se buscaran resultados sin importar el porcentaje.'
    consequent, antecedent, confidence = Buscar_Todo(nro_filas, tag)
  
  return consequent, antecedent, confidence, mensaje


def crear_tabla_resultados(consequent, antecedent, confidence):
  np_consequent = np.array(consequent)
  np_antecedent = np.array(antecedent)
  np_confidence = np.array(confidence)

  tabla_resultados_reglas = pd.DataFrame(np_consequent, columns = ['Consecuente'])
  tabla_resultados_reglas['Antecedente'] = np_antecedent.tolist()
  tabla_resultados_reglas['Confianza'] = np_confidence.tolist()

  return tabla_resultados_reglas

# ****************************************************** SIMULACIÓN FINAL *******************************************************
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from numpy import *

from sklearn.svm import SVC

df_SPC_Original= pd.read_csv("data/SPC_Final_Nombre.csv")
df_SPC_FINAL= pd.read_csv("data/SPC_Final.csv")

def test_model_simulacion(model, df, nfols): 
  X = df.iloc[:,:-1].values #Caracteristicas
  Y = df.iloc[:,-1].values #Columna objetivo
  skf = StratifiedKFold(n_splits=nfols)
  acuracym = 0 #Exactitud
  precisionm = 0 #Precisión
  recallm = 0 #Recall
  f1m = 0 #F1
  for train_index, test_index in skf.split(X, Y): #Cross-validation estratificado con k folds
    X_train, X_test = X[train_index], X[test_index]
    Y_train, Y_test = Y[train_index], Y[test_index]
    model.fit(X_train, Y_train)
    predicciones = model.predict(X_test)
    acuracym += accuracy_score(Y_test, predicciones)
    precision, recall, f1, support = precision_recall_fscore_support(Y_test, predicciones, average='weighted', zero_division=0)
    precisionm += precision
    recallm += recall
    f1m += f1
  return acuracym/nfols, precisionm/nfols, recallm/nfols, f1m/nfols

# ****************************************************** REGIONES *******************************************************

def itemsets_fpgrowth_region(trans_, supp_):
  #calcular los itemsets frecuentes con un soporte supp_ y reportando su 
  #frecuencia absoluta a y relativa S
  r = fpgrowth(trans_, target='a', supp=supp_, report='aS') 
  df_items = pd.DataFrame(r)#transformar en dataframe
  df_items.columns = ['Itemset', 'Freq', 'Freq(%)']#cambiar encabezados de las columnas
  df_items.sort_values(by='Freq',ascending=False,inplace=True)#ordenar los itemset por freq
  #ordenar los items de cada itemset para evitar confusion
  df_items['Tamaño'] = [len(x) for x in df_items['Itemset'].tolist()]
  df_items['Itemset'] = [str(sorted(x)) for x in df_items['Itemset'].tolist()]
  return df_items

def items_por_region(df_siembra, region):

  df_siembra_region = df_siembra[df_siembra['DEPARTAMENTO'] == region]
  transaccional_region = to_transactionnal(df_siembra_region, 'DISTRITO', 'CULTIVO')
  df_itemsets_region = itemsets_fpgrowth_region(transaccional_region, 10)

  return df_itemsets_region

def emerging_comparacion_regiones(df_siembra_, region_1, region_2):

  df_region_1 = items_por_region(df_siembra_, region_1)
  df_region_2 = items_por_region(df_siembra_, region_2)
  emerging_by_region_1 = df_region_1.join(df_region_2.set_index("Itemset"), on="Itemset",lsuffix="_Region_1",rsuffix="_Region_2").fillna(0)
  emerging_by_region_1['GrowthRate'] = (emerging_by_region_1["Freq(%)_Region_1"]/emerging_by_region_1["Freq(%)_Region_2"])
  emerging_by_region_1.sort_values(by="GrowthRate",ascending=False,inplace=True)

  df_emerging_by_region_1 = emerging_by_region_1[emerging_by_region_1["GrowthRate"] < np.inf]

  return df_emerging_by_region_1
