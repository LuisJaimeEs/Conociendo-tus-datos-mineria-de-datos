import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder


sns.set()
df = pd.read_csv('mushrooms.csv')

#Informacion de los datos
df.info()

df.describe()

print(df[df.duplicated()])

print(df["class"].value_counts())

plt.title('Elementos por clase')

#Informacion estadistica por atributo
df.describe(include = 'all')

#Valores duplicados
print(df[df.duplicated()])

#Valores count
print(df["class"].value_counts())

#Petal
print(df["petal lenght"].value_counts())

plt.title('Elementos por clase')
print(sns.countplot(x=df["class"]))

sns.scatterplot(x = df["sepal length"], y = df["sepal width"], hue = df["class"])

sns.scatterplot(x = df["sepal length"], y = df["petal width"], hue = df["class"])

#Tarea con if separar clases
sns.pairplot(df, hue = "class")

plt.figure()
sns.heatmap(df.corr(), annot=True)

print(df.groupby("class").agg((["mean", "median"])))

plt.show()


fig, axes = plt.subplots(2, 2, figsize = (16, 9))
sns.boxplot(y = "petal width", x = "class", data = df, orient = "v", ax = axes[0, 0])
sns.boxplot(y = "petal length", x = "class", data = df, orient = "v", ax = axes[0, 1])
sns.boxplot(y = "petal width", x = "class", data = df, orient = "v", ax = axes[1, 0])
sns.boxplot(y = "petal length", x = "class", data = df, orient = "v", ax = axes[1, 1])

sns.FacetGrid(df, hue = "class", height = 5)\
    .map(sns.distplot, "petal width")\
        .add_legend()

sns.FacetGrid(df, hue = "class", height = 5)\
    .map(sns.distplot, "petal length")\
        .add_legend()
        
sns.FacetGrid(df, hue = "class", height = 5)\
    .map(sns.distplot, "sepal width")\
        .add_legend()
        
sns.FacetGrid(df, hue = "class", height = 5)\
    .map(sns.distplot, "petal length")\
        .add_legend()
        

#No se le coloca paretesis al metodo, no se el motivo
print(df.corr)
print(df.cov)

df
data = df.drop(["class"], axis = 1)
print(data)

data_normal = StandardScaler().fit_transform(data)
print(data_normal)


df_normal = pd.DataFrame(data_normal, columns = ["sepal length", "sepal width", "petal length", "petal width"])
print(data_normal)

df_normal["class"] = df["class"]
print(df_normal)


sns.pairplot(df_normal, hue = "class")

print(df_normal.corr)

print(df_normal.cov)

df_normal["petal width"].var()

fig, axes = plt.subplots(2, 2, figsize = (16, 9))
sns.boxplot(y = "petal width", x = "class", data = df_normal, orient = "v", ax = axes[0, 0])
sns.boxplot(y = "petal length", x = "class", data = df_normal, orient = "v", ax = axes[0, 1])
sns.boxplot(y = "sepal width", x = "class", data = df_normal, orient = "v", ax = axes[1, 0])
sns.boxplot(y = "sepal length", x = "class", data = df_normal, orient = "v", ax = axes[1, 1])
plt.show()

pca = PCA(n_components = 3)
principalComponents = pca.fit_transform(data_normal)

pca.get_covariance()

#No se le agrega parentesis en el metodo cov
df_normal.cov

principalComponents

principal_df = pd.DataFrame(principalComponents, columns = ["PCA1", "PCA2", "PCA3"])
final_df = pd.concat([principal_df, df[["class"]]], axis = 1)
print(final_df)


varianza = pca.explained_variance_ratio_
print(varianza)

suma = varianza [0] + varianza [1] 
print(suma)

plt.figure(figsize = (3, 2))
plt.bar(range(3), varianza, alpha = 0.5, align = "center")
plt.ylabel("varianza")
plt.xlabel("Principal components")
plt.show()

reducida_df = final_df.drop(["PCA3"], axis = 1)
print(reducida_df)

sns.pairplot(reducida_df, hue = "class", height = 4)

#Uso red neuronal

data_training = reducida_df.drop
label_encoder = LabelEncoder()
etiquetas = label_encoder.fit_transform(reducida_df["class"])
print(etiquetas)

data_training = reducida_df.drop(["class"], axis = 1)
clf = MLPClassifier(solver = "lbfgs", alpha = 1e-5, hidden_layer_sizes = (1000, 200, 50, 50), random_state = 1)
clf.fit(data_training, etiquetas)
puntuacion = clf.score(data_training, etiquetas)
print(puntuacion)

