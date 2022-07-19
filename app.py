from sklearn.metrics import precision_recall_fscore_support
import pandas as pd
from sklearn.svm import SVC
from sklearn.metrics import roc_auc_score
import joblib
import os
import numpy as np
from flask import Flask, render_template, url_for, request, jsonify
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
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.ensemble import RandomForestClassifier as RFC

from fim import *

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import random 
import graphviz
from graphviz import *
import time

#from pyspark.ml.fpm import FPGrowth

app = Flask(__name__)
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0

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


@app.route('/')
def index():
    return render_template("home.html")


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    # Receives the input query from form
    if request.method == 'POST':
        age = request.form['age']
        sex = request.form['sex']
        steroid = request.form['steroid']
        antivirals = request.form['antivirals']
        fatigue = request.form['fatigue']
        spiders = request.form['spiders']
        ascites = request.form['ascites']
        varices = request.form['varices']
        bilirubin = request.form['bilirubin']
        alk_phosphate = request.form['alk_phosphate']
        sgot = request.form['sgot']
        albumin = request.form['albumin']
        protime = request.form['protime']
        histology = request.form['histology']
        sample_result = {"age": age, "sex": sex, "steroid": steroid, "antivirals": antivirals, "fatigue": fatigue, "spiders": spiders, "ascites": ascites,
                         "varices": varices, "bilirubin": bilirubin, "alk_phosphate": alk_phosphate, "sgot": sgot, "albumin": albumin, "protime": protime, "histolog": histology}
        single_data = [age, sex, steroid, antivirals, fatigue, spiders, ascites,
                       varices, bilirubin, alk_phosphate, sgot, albumin, protime, histology]
        print(single_data)
        print(len(single_data))
        numerical_encoded_data = [float(int(x)) for x in single_data]
        model = load_model('models/logistic_regression_hepB_model.pkl')
        prediction = model.predict(
            np.array(numerical_encoded_data).reshape(1, -1))
        print(prediction)
        prediction_label = {"Die": 1, "Live": 2}
        final_result = get_key(prediction[0], prediction_label)
        pred_prob = model.predict_proba(
            np.array(numerical_encoded_data).reshape(1, -1))
        pred_probalility_score = {
            "Die": pred_prob[0][0]*100, "Live": pred_prob[0][1]*100}

    return render_template("index.html", sample_result=sample_result, prediction=final_result, pred_probalility_score=pred_probalility_score)


@app.route('/dataset')
def dataset():
    df = pd.read_csv("data/meteorologica_automatica.csv")
    return render_template("dataset.html", df_table=df)

@app.route('/home')
def home():
    return render_template("home.html")


@app.route('/patern', methods=['GET', 'POST'])
def pattern_ming():
	prediction = "('Arveja grano seco', 'Frijol grano seco', 'Papa color', 'Olluco', 'Trigo', 'Papa blanca', 'Cebada grano', 'Maiz amilaceo') 151"
	return render_template("pattern_mining.html", prediction=prediction)

@app.route('/fpgrowth', methods=['GET', 'POST'])
def fpgrowth_():
    prediction = "('Arveja grano seco', 'Frijol grano seco', 'Papa color', 'Olluco', 'Trigo', 'Papa blanca', 'Cebada grano', 'Maiz amilaceo') 151"
    return render_template("fp-growth.html", prediction=prediction)

@app.route('/simulacion', methods=['GET', 'POST'])
def simulacion():

    accuracy = 0
    precision = 0
    recall = 0
    f1 = 0

    if request.method == 'POST':
        c = request.form['c']
        #kernel = request.form['kernel']
        #sample_result = {"c": c, "kernel": kernel}
        sample_result = {"c": c}
        #single_data = [c, kernel]
        single_data = [c]

        numerical_encoded_data = [float(int(x)) for x in single_data]        

        c_ = numerical_encoded_data[0]
        #kernel_ = single_data[0]

        df = pd.read_csv("data/pruebaB.csv")
        df.drop(['eli'], axis=1)
        bestsvm = SVC(C=c_, kernel='rbf', probability=True)
        acc, pre, rec, f1_ = test_model(bestsvm, df, 10)
        accuracy = acc
        precision = pre
        recall = rec
        f1 = f1_
    return render_template("simulacion.html", accuracy=accuracy, precision=precision, recall=recall, f1=f1)

@app.route('/svm', methods=['GET', 'POST'])
def svm():

    accuracy = 0
    precision = 0
    recall = 0
    f1 = 0

    if request.method == 'POST':
        c = request.form['c']
        kernel_ = request.form['kernel']
        #sample_result = {"c": c, "kernel": kernel}
        sample_result = {"c": c}
        #single_data = [c, kernel]
        single_data = [c]

        numerical_encoded_data = [float(int(x)) for x in single_data]        

        c_ = numerical_encoded_data[0]
        #kernel_ = single_data[0]

        df = pd.read_csv("data/SPC_Final.csv")
        bestsvm = SVC(C=c_, kernel=kernel_, probability=True)
        acc, pre, rec, f1_ = test_model(bestsvm, df, 10)
        accuracy = acc
        precision = pre
        recall = rec
        f1 = f1_
    return render_template("svm.html", accuracy=accuracy, precision=precision, recall=recall, f1=f1)

@app.route('/knn', methods=['GET', 'POST'])
def knn():

    accuracy = 0
    precision = 0
    recall = 0
    f1 = 0

    if request.method == 'POST':
        neighbors = request.form['neighbors']
        weights_ = request.form['weights']
        algorithm_ = request.form['algorithm']

        sample_result = {"neighbors": neighbors}

        single_data = [neighbors]

        numerical_encoded_data = [int(int(x)) for x in single_data]        

        neighbors_ = numerical_encoded_data[0]

        df = pd.read_csv("data/SPC_Final.csv")
        bestsvm = KNN(n_neighbors=neighbors_, weights=weights_, algorithm=algorithm_)
        acc, pre, rec, f1_ = test_model(bestsvm, df, 10)
        accuracy = acc
        precision = pre
        recall = rec
        f1 = f1_
    return render_template("knn.html", accuracy=accuracy, precision=precision, recall=recall, f1=f1)

@app.route('/rf', methods=['GET', 'POST'])
def rf():

    accuracy = 0
    precision = 0
    recall = 0
    f1 = 0

    if request.method == 'POST':
        estimators = request.form['estimators']
        max_depth = request.form['max_depth']
        max_features_ = request.form['max_features']

        sample_result = {"estimators": estimators, "max_depth": max_depth}

        single_data = [estimators, max_depth]

        numerical_encoded_data = [int(int(x)) for x in single_data]        

        estimators_ = numerical_encoded_data[0]
        max_depth_ = numerical_encoded_data[1]

        df = pd.read_csv("data/SPC_Final.csv")
        bestsvm = RFC(n_estimators=estimators_, max_depth=max_depth_, max_features=max_features_)
        acc, pre, rec, f1_ = test_model(bestsvm, df, 10)
        accuracy = acc
        precision = pre
        recall = rec
        f1 = f1_
    return render_template("rf.html", accuracy=accuracy, precision=precision, recall=recall, f1=f1)

@app.route('/dashboard-1')
def dash01():
    return render_template("dash01.html")

@app.route('/prueba_dash')
def prueba_dash():
    return render_template("prueba_dash.html")

@app.route('/registrar')
def registrar():
    return render_template("register.html")

@app.route('/login')
def login():
    return render_template("login.html")

@app.route('/contraseña')
def contraseña():
    return render_template("password.html")

@app.route('/prueba_mining')
def prueba_mining():
    return render_template("prueba_mining.html")


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

@app.route('/mining-01')
def mining01():

    df_siembra_freq = pd.read_csv("data/pruebaPM.csv")
    df_siembra_freq.drop(['eli'], axis=1)

    trans = to_transactionnal(df_siembra_freq, 'DISTRITO', 'CULTIVO')

    r = fpgrowth(trans, target='c', supp=10, zmin=2)
    df = pd.DataFrame(r)
    df.columns = ['Itemset','Freq']
    df.sort_values(by='Freq',ascending=False,inplace=True)

    #df = all_itemsets(trans, -1)

    labels = df_siembra_freq["DISTRITO"].unique()
    df["Supp"] = [supp(x,labels,trans) for x in df["Itemset"].tolist()]

    #return render_template("mining-01.html", df_table=df)
    return render_template("mining-01.html", df_table=df)

@app.route('/min01')
def mini01():

    df_siembra_freq = pd.read_csv("data/pruebaPM.csv")
    df_siembra_freq.drop(['eli'], axis=1)

    trans = to_transactionnal(df_siembra_freq, 'DISTRITO', 'CULTIVO')

    r = fpgrowth(trans, target='c', supp=10, zmin=2)
    df = pd.DataFrame(r)
    df.columns = ['Itemset','Freq']
    df.sort_values(by='Freq',ascending=False,inplace=True)

    labels = df_siembra_freq["DISTRITO"].unique()
    df["Supp"] = [supp(x,labels,trans) for x in df["Itemset"].tolist()]

    #df = all_itemsets(trans, -1)

    #return render_template("mining-01.html", df_table=df)
    return render_template("prueba_mining.html", df_table=df)


@app.route('/min02')
def mini02():

    df_siembra_freq = pd.read_csv("data/pruebaPM.csv")
    df_siembra_freq.drop(['eli'], axis=1)

    trans = to_transactionnal(df_siembra_freq, 'DISTRITO', 'CULTIVO')

    r = fpgrowth(trans, target='c', supp=10, zmin=2)
    df = pd.DataFrame(r)
    df.columns = ['Itemset','Freq']
    df.sort_values(by='Freq',ascending=False,inplace=True)

    labels = df_siembra_freq["DISTRITO"].unique()
    df["Supp"] = [supp(x,labels,trans) for x in df["Itemset"].tolist()]

    df = all_itemsets(trans, -1)
    
    #return render_template("mining-01.html", df_table=df)

    return render_template("prueba_mining_.html", df_table=df)


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

df_precios = pd.read_csv("data/reporte_sisap.csv", sep=';')
df_clima = pd.read_csv("data/reporte_integrado.csv")

df_SPC_Original= pd.read_csv("data/SPC_Final_Nombre.csv")
df_SPC_FINAL= pd.read_csv("data/SPC_Final.csv")



print(data_reglas.shape)
# ****************************************************** Mostrar HTML PM *******************************************************
@app.route('/mining_01')
def pattern_mining_01():
    return render_template("pattern_mining_01.html", df_table=data_transactionnal)

@app.route('/mining_02')
def pattern_mining_02():
    return render_template("pattern_mining_02.html", df_table=data_supports)

@app.route('/mining_03')
def pattern_mining_03():
    return render_template("pattern_mining_03.html", df_table=data_reglas.head(1000))

# HTML --> Metricas FP-Grow y Apriori
@app.route('/itemsets_fpgrowth')
def itemsets_fpgrowth():
    return render_template("itemsets_fpgrowth.html", df_table=data_itemsets_fpgrowth.head(1000), tiempo=tiempo_fpgrowth)

@app.route('/itemsets_apriori')
def itemsets_apriori():
    return render_template("itemsets_apriori.html", df_table=data_itemsets_apriori.head(1000), tiempo=tiempo_apriori)


# ****************************************************** DATASETS *******************************************************
@app.route('/midagri')
def midagri():
    return render_template("midagri.html", df_table=dataset.head(1000))

@app.route('/precios')
def precios():
    #df = pd.read_csv("data/reporte_sisap.csv", sep=';')
    return render_template("precios.html", df_table=df_precios)

@app.route('/senamhi')
def senamhi():
    #df = pd.read_csv("data/reporte_integrado.csv")
    return render_template("senamhi.html", df_table=df_clima.head(1000))

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

@app.route('/mapa', methods=['GET', 'POST'])
def mapa():
    var = 0
    if request.method == 'POST':
        var = request.form['var']
        #kernel = request.form['kernel']
        #sample_result = {"c": c, "kernel": kernel}
        sample_result = {"var": var}
        #single_data = [c, kernel]
        single_data = [var]

        numerical_encoded_data = [float(int(x)) for x in single_data]        

        var_ = numerical_encoded_data[0]
        print(var_)
        mydrawnmap = draw_map(data_supports,var_)
        fig_to_base64(draw_map(data_supports,var_))
        #time.sleep(5)
        #return render_template("map_01.html", name = 'Mapa Perú', url ='/static/images/mapa.png')

    return render_template("map_01.html", name = 'Mapa Perú', url ='/static/images/mapa.png', numero_itemset = var)

    #else:
        #return render_template("map_01.html", name = 'Mapa Perú', url ='/static/images/mapa.png')


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

@app.route('/prediccion', methods=['GET', 'POST'])
def prediccion():

    consequent = []
    antecedent = []
    confidence = []
    mensaje = ''

    tabla_resultados = pd.DataFrame()

    if request.method == 'POST':
        producto = request.form['producto']
        porcentaje = request.form['porcentaje']
        filas = request.form['filas']

        sample_result = {"producto": producto, "porcentaje": porcentaje, "filas": filas}
        single_data = [producto, porcentaje, filas]
        single_data_number = [porcentaje, filas]

        numerical_encoded_data = [float(int(x)) for x in single_data_number]        

        producto_ = single_data[0]
        porcentaje_ = numerical_encoded_data[0]
        filas_ = numerical_encoded_data[1]

        consequent, antecedent, confidence, mensaje  = Buscar_Reglas_Asociacion(producto_, filas_, Seleccionar_Mejores_Reglas(porcentaje_, Reglas_Tabla_Ordenada(data_reglas)))

        tabla_resultados = crear_tabla_resultados(consequent, antecedent, confidence)
    
    return render_template("prediccion.html", df_resultados=tabla_resultados, mensaje=mensaje)


# ****************************************************** CREAR DATASET POR MES *******************************************************

# ***************************** JUNTAR MIDAGRI CON PRECIOS *****************************
def Precios_Tabla_Ordenada(df_precios):
  
  df_to_sort = df_precios.sort_values(by='nombre', ascending=False)

  nombre = df_to_sort['nombre'].to_numpy()
  anio = df_to_sort['año'].to_numpy()
  mes = df_to_sort['mes'].to_numpy()
  precio = df_to_sort['precio'].to_numpy()
  volumen = df_to_sort['volumen'].to_numpy()

  new_df_sort = pd.DataFrame(nombre, columns = ['nombre'])

  new_df_sort['año'] = anio.tolist()
  new_df_sort['mes'] = mes.tolist()
  new_df_sort['precio'] = precio.tolist()
  new_df_sort['volumen'] = volumen.tolist()

  return new_df_sort

def tabla_precios_mes(nro_mes, nro_anio, data_precios):
  data_mes = data_precios.query("mes == @nro_mes and año == @nro_anio")
  data_mes = Precios_Tabla_Ordenada(data_mes)

  return data_mes

def juntar_df_precios(df_siembra_, df_precios_mes_):
  arr_precios = []
  arr_volumenes = []

  for k in range(len(df_siembra_['CULTIVO'])):
    for i in range(len(df_precios_mes_['nombre'])):
      if df_siembra_['CULTIVO'][k] == df_precios_mes_['nombre'][i]:
        arr_precios.append(df_precios_mes_['precio'][i])
        arr_volumenes.append(df_precios_mes_['volumen'][i])

    if len(arr_precios) <= k:
      arr_precios.append(0)
      arr_volumenes.append(0)

  np_precios = np.array(arr_precios)
  np_volumenes = np.array(arr_volumenes)

  df_siembra_['Precio Promedio'] = np_precios.tolist()
  df_siembra_['Volumen Promedio'] = np_volumenes.tolist()
  
  return df_siembra_

# ***************************** JUNTAR SIEMCRA-PRECIOS CON CLIMA *****************************
def Clima_Tabla_Ordenada(df_clima):
  
  df_to_sort = df_clima.sort_values(by='depa', ascending=False)

  depa = df_to_sort['depa'].to_numpy()
  prov = df_to_sort['prov'].to_numpy()
  dist = df_to_sort['dist'].to_numpy()
  nombre = df_to_sort['nombre'].to_numpy()
  year = df_to_sort['year'].to_numpy()
  month = df_to_sort['month'].to_numpy()
  temp = df_to_sort['temp'].to_numpy()
  hum = df_to_sort['hum'].to_numpy()
  precip = df_to_sort['precip'].to_numpy()

  new_df_sort = pd.DataFrame(depa, columns = ['depa'])

  new_df_sort['prov'] = prov.tolist()
  new_df_sort['dist'] = dist.tolist()
  new_df_sort['nombre'] = nombre.tolist()
  new_df_sort['year'] = year.tolist()
  new_df_sort['month'] = month.tolist()
  new_df_sort['temp'] = temp.tolist()
  new_df_sort['hum'] = hum.tolist()
  new_df_sort['precip'] = precip.tolist()

  return new_df_sort

def tabla_clima_mes(nro_mes, nro_anio, data_clima):
  data_mes = data_clima.query("month == @nro_mes and year == @nro_anio")
  data_mes = Clima_Tabla_Ordenada(data_mes)

  return data_mes

def juntar_df_clima(df_siembra_precios_, df_clima_mes_):
  arr_temperatura = []
  arr_humedad = []
  arr_precipitacion = []

  for k in range(len(df_siembra_precios_['DISTRITO'])):
    for i in range(len(df_clima_mes_['dist'])):

      if df_siembra_precios_['DISTRITO'][k] == df_clima_mes_['dist'][i] or df_siembra_precios_['PROVINICA'][k] == df_clima_mes_['prov'][i] or df_siembra_precios_['DEPARTAMENTO'][k] == df_clima_mes_['depa'][i]:
        arr_temperatura.append(df_clima_mes_['temp'][i])
        arr_humedad.append(df_clima_mes_['hum'][i])
        arr_precipitacion.append(df_clima_mes_['precip'][i])
        break

    if len(arr_temperatura) <= k:
      arr_temperatura.append(0)
      arr_humedad.append(0)
      arr_precipitacion.append(0)    

  np_temperatura = np.array(arr_temperatura)
  np_humedad = np.array(arr_humedad)
  np_precipitacion = np.array(arr_precipitacion)

  df_siembra_precios_['Temperatura'] = np_temperatura.tolist()
  df_siembra_precios_['Humedad'] = np_humedad.tolist()
  df_siembra_precios_['Precipitación'] = np_precipitacion.tolist()
  
  return df_siembra_precios_

def reducir_tabla_spc_columnas(df_spc):
  df_spc = df_spc.drop('AGO', 1)
  df_spc = df_spc.drop('SEP', 1)
  df_spc = df_spc.drop('OCT', 1)
  df_spc = df_spc.drop('NOV', 1)
  df_spc = df_spc.drop('DIC', 1)
  df_spc = df_spc.drop('ENE', 1)
  df_spc = df_spc.drop('FEB', 1)
  df_spc = df_spc.drop('MAR', 1)
  df_spc = df_spc.drop('ABR', 1)
  df_spc = df_spc.drop('MAY', 1)
  df_spc = df_spc.drop('JUN', 1)
  df_spc = df_spc.drop('JUL', 1)
  df_spc = df_spc.drop('CAMPANA', 1)

  return df_spc

def elegir_mes_anio(mes, anio):

    df_precios_mes = tabla_precios_mes(mes, anio, df_precios)
    df_siembra_precios = juntar_df_precios(dataset, df_precios_mes)

    df_clima_mes = tabla_clima_mes(mes, anio, df_clima)
    df_siembra_precios_clima = juntar_df_clima(df_siembra_precios, df_clima_mes)

    # Reducir Columnas:
    df_spc_ = df_siembra_precios_clima
    df_spc_ = reducir_tabla_spc_columnas(df_spc_)

    # Reducir Filas:
    df_spc_aux = df_spc_

    for k in range(len(df_spc_aux['Precio Promedio'])):
      if df_spc_aux['Precio Promedio'][k] == 0.0:
        df_spc_aux.drop(k,axis=0, inplace=True)

    df_spc_final = df_spc_aux

    df_SPC_Original = df_spc_aux
    df_spc_aux.to_csv('data/SPC_Final_Nombre.csv', index=False)

@app.route('/cambiar_mes', methods=['GET', 'POST'])
def cambiar_mes():

    if request.method == 'POST':
        mes = request.form['mes']
        anio = request.form['anio']

        sample_result = {"mes": mes, "anio": anio}
        single_data_number = [mes, anio]
        #single_data_number = [porcentaje, filas]

        numerical_encoded_data = [int(int(x)) for x in single_data_number]        

        mes_ = numerical_encoded_data[0]
        anio_ = numerical_encoded_data[1]

        elegir_mes_anio(mes_, anio_)
    
    return render_template("cambiar_mes.html")

# ****************************************************** PREPROCESAMIENTO - PREDICCIÓN *******************************************************

def preprocesamiento_Modelo():

    df_spc_final = df_SPC_Original
    from sklearn.preprocessing import LabelEncoder
    #Codificando todas las variables categoricas, ya que los clasificadores solo entienden datos numericos
    categorical_feature_mask = df_spc_final.dtypes==object
    categorical_cols = df_spc_final.columns[categorical_feature_mask].tolist()
    le = LabelEncoder()
    df_spc_final[categorical_cols] = df_spc_final[categorical_cols].apply(lambda col: le.fit_transform(col))

    df_spc_final['Cultivo'] = df_spc_final['CULTIVO']

    df_spc_final = df_spc_final.drop('CULTIVO', axis=1)

    df_spc_final_ = df_spc_final

    df_spc_final_.dropna(subset = ["Temperatura"], axis = 0, inplace = True)
    df_spc_final_.dropna(subset = ["Humedad"], axis = 0, inplace = True)

    df_SPC_FINAL = df_spc_final_
    df_spc_final_.to_csv('data/SPC_Final.csv', index=False)


# ****************************************************** SIMULACIÓN FINAL *******************************************************
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from numpy import *

from sklearn.svm import SVC

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

@app.route('/simulacion_final', methods=['GET', 'POST'])
def simulacion_final():
    df_reglas = pd.DataFrame()

    svm = SVC(C=100, kernel='linear')

    acc, pre, rec, f1 = test_model_simulacion(svm, df_SPC_FINAL, 10)

    mensaje_ = " "

    nombre_prediccion = " "

    if request.method == 'POST':
        departamento = request.form['departamento']
        provincia = request.form['provincia']
        distrito = request.form['distrito']
        precio = request.form['precio']
        volumen = request.form['volumen']
        temperatura = request.form['temperatura']
        humedad = request.form['humedad']
        precipitacion = request.form['precipitacion']

        #sample_result = {"producto": producto, "porcentaje": porcentaje, "filas": filas}
        #single_data = [producto, porcentaje, filas]

        single_data_number = [departamento, provincia, distrito, precio, volumen, temperatura, humedad, precipitacion]

        numerical_encoded_data = [float(float(x)) for x in single_data_number]        

        departamento_ = numerical_encoded_data[0]
        provincia_ = numerical_encoded_data[1]
        distrito_ = numerical_encoded_data[2]
        precio_ = numerical_encoded_data[3]
        volumen_ = numerical_encoded_data[4]
        temperatura_ = numerical_encoded_data[5]
        humedad_ = numerical_encoded_data[6]
        precipitacion_ = numerical_encoded_data[7]

        #Separate Feature and Target Matrix
        x = df_SPC_FINAL.drop('Cultivo',axis = 1) 
        y = df_SPC_FINAL.Cultivo
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2,random_state=100)
        SupportVectorClassModel = SVC(kernel='linear')
        SupportVectorClassModel.fit(x_train,y_train)

        valor = SupportVectorClassModel.predict([[departamento_,    provincia_,  distrito_,   precio_,   volumen_, 
            temperatura_,   humedad_,  precipitacion_]])

        if valor == 0:
            nombre_prediccion = "Aji"
        elif valor == 1:
            nombre_prediccion = "Ajo"
        elif valor == 2:
            nombre_prediccion = "Arroz cascara"
        elif valor == 3:
            nombre_prediccion = "Arveja grano seco"
        elif valor == 4:
            nombre_prediccion = "Arveja grano verde"
        elif valor == 5:
            nombre_prediccion = "Camote"
        elif valor == 6:
            nombre_prediccion = "Cebolla cabeza roja"
        elif valor == 7:
            nombre_prediccion = "Frijol grano seco"
        elif valor == 8:
            nombre_prediccion = "Haba grano seco"
        elif valor == 9:
            nombre_prediccion = "Maiz amarillo duro"
        elif valor == 10:
            nombre_prediccion = "Maiz amilaceo"
        elif valor == 11:
            nombre_prediccion = "Olluco"
        elif valor == 12:
            nombre_prediccion = "Papa blanca"
        elif valor == 13:
            nombre_prediccion = "Papa color"
        elif valor == 14:
            nombre_prediccion = "Papa nativa"
        elif valor == 15:
            nombre_prediccion = "Paprika"
        elif valor == 16:
            nombre_prediccion = "Tomate"
        elif valor == 17:
            nombre_prediccion = "Yuca"
        elif valor == 18:
            nombre_prediccion = "Zanahoria"

        mensaje_ = "La predicción se ha realizado utilizando el modelo de Machine Learning con el algoritmo SVM"

        consequent, antecedent, confidence, mensaje  = Buscar_Reglas_Asociacion(nombre_prediccion, 4, Seleccionar_Mejores_Reglas(40, Reglas_Tabla_Ordenada(data_reglas)))

        df_reglas = crear_tabla_resultados(consequent, antecedent, confidence)

    return render_template("simulacion_final.html", df_table=df_reglas, val = nombre_prediccion, men=mensaje_)

@app.route('/data_original')
def data_original():
    return render_template("data_original.html", df_table=df_SPC_Original.head(1000))

@app.route('/data_prediccion')
def data_prediccion():
    return render_template("data_prediccion.html", df_table=df_SPC_FINAL.head(1000))

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

@app.route('/buscar_region', methods=['GET', 'POST'])
def buscar_region():

    tabla_region = pd.DataFrame()

    nombre_region = " "

    if request.method == 'POST':
        region = request.form['region']
        single_data = [region]
        region_ = single_data[0]

        nombre_region = region_

        tabla_region = items_por_region(dataset, region_)

    return render_template("buscar_region.html", df_resultados=tabla_region, nombre=nombre_region)

@app.route('/comparar_region', methods=['GET', 'POST'])
def comparar_region():

    tabla_comparacion = pd.DataFrame()

    nombre_region_1 = " "
    nombre_region_2 = " "

    if request.method == 'POST':
        region_1 = request.form['region_1']
        region_2 = request.form['region_2']

        single_data = [region_1, region_2]

        region_1_ = single_data[0]
        region_2_ = single_data[1]

        nombre_region_1 = region_1_
        nombre_region_2 = region_2_

        tabla_comparacion = emerging_comparacion_regiones(dataset, nombre_region_1, nombre_region_2)

    return render_template("comparar_region.html", df_resultados=tabla_comparacion, nombre_1=nombre_region_1, nombre_2=nombre_region_2)

if __name__ == '__main__':
    app.run(debug=True)
