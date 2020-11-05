import pandas as pd 
import numpy as np
from sklearn.preprocessing import LabelEncoder
from keras.models import Sequential
from keras.layers import Dense

#tratando os dados, removendo lote e transformando fruta em numero
csv = pd.read_csv('dados.csv', sep = ",")
le = LabelEncoder()
csv['fruta'] = le.fit_transform(csv['fruta'])

csv = csv.drop(columns=['lote'])

dados = csv.values

print(dados)

classificadores = dados[:,0]
atributos = dados[:,1:]

#modelo o minimo seria 1,5 ai coloquei logo 10

modelo = Sequential()
modelo.add(Dense(units = 10, activation= 'relu'))
modelo.add(Dense(units = 1, activation='sigmoid'))

#treinando 

modelo.compile(optimizer="adam", loss="binary_crossentropy", metrics=["binary_accuracy"])
modelo.fit(atributos,classificadores, batch_size=20, epochs=100)

#predizendo 
novos = np.array([
    [50,4],
    [100,5]
])

resposta = modelo.predict(novos)
print (resposta)

if resposta < 0.5 :
    print ("Laranja")
else:
    print("Limão")

## eu ia fazer salvando também mas meu computador não ajudou 
## fiz sem executar mas é basicamente isso 
## 