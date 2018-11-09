import matplotlib.pyplot as plt
import pandas as pd
from sklearn import linear_model
import numpy as np

df = pd.read_csv("Irish.txt", "r", delimiter="\t")

Number = df.No
Sapal_L = df.Sepal_Length
Sapal_W = df.Sepal_Width
Petal_L = df.Petal_Length
Petal_W = df.Petal_Width
Species = df.Species

X = np.array([Sapal_L, Sapal_W]).transpose()
lr = linear_model.LogisticRegression()
lr.fit(X, Species)

prediction = lr.predict(np.array([Sapal_L[67], Sapal_W[67]]).reshape(1, -1))
print(prediction)

Setosa_S_L = Sapal_L[:45]
Versicolor_S_L = Sapal_L[50:95]
Virginica_S_L = Sapal_L[100:145]

Setosa_S_W = Sapal_W[:45]
Versicolor_S_W = Sapal_W[50:95]
Virginica_S_W = Sapal_W[100:145]

plt.scatter(Setosa_S_L, Setosa_S_W,label="Setosa")
plt.scatter(Versicolor_S_L, Versicolor_S_W,label="Versicolor")
plt.scatter(Virginica_S_L, Virginica_S_W,label="Virginica")
plt.scatter(Sapal_L[67], Sapal_W[67])

plt.legend()
plt.show()
