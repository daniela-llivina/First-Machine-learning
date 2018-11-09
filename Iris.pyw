import matplotlib.pyplot as plt
import pandas as pd
from sklearn import linear_model

df = pd.read_csv("Irish.txt","r",delimiter="\t")

Number = df.No
Sapal_L = df.Sepal_Length
Sapal_W = df.Sepal_Width
Petal_L = df.Petal_Length
Petal_W = df.Petal_Width
Species = df.Species 

Setosa_S_L = Sapal_L[:45]
Versicolor_S_L = Sapal_L[50:95]
Virginica_S_L = Sapal_L[100:145]

Setosa_S_W = Sapal_W[:45]
Versicolor_S_W = Sapal_W[50:95]
Virginica_S_W = Sapal_W[100:145]

x = [[Setosa_S_L,Setosa_S_W],[Versicolor_S_L,Versicolor_S_W],[Virginica_S_L,Virginica_S_W]]
y = ["Setosa","Versicolor","Virginica"]

lr = linear_model.LogisticRegression()
lr.fit(x,y)

print(lr.predict([Sapal_L[47],Sapal_W[47]]))

plt.scatter(Setosa_S_L,Setosa_S_W,label="Setosa")
plt.scatter(Versicolor_S_L,Versicolor_S_W,label="Versicolor")
plt.scatter(Virginica_S_L,Virginica_S_W,label="Virginica")
plt.scatter(Sapal_L[47],Sapal_W[47])

plt.legend()
plt.show()
