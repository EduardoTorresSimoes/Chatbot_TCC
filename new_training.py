import random
import json
import pickle
import numpy as np

import nltk
from nltk.stem import WordNetLemmatizer

import pandas as pd

from sklearn.model_selection import train_test_split

from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

from tensorflow import keras

from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.optimizers.legacy import SGD

# Inicialização do lemmatizer (usado para normalizar as palavras).
lemmatizer = WordNetLemmatizer()

# Carregando os dados de intents do arquivo JSON.
intents = json.loads(open("intents.json", encoding="utf8").read())

# Inicialização das listas para armazenar palavras, classes e documentos.
words = []
classes = []
documents = []
ignore_letters = [".", "!", "?", ",", "_"]

# Iteração através dos intents e seus padrões para processar as palavras.
for intent in intents["intents"]:
    for pattern in intent["patterns"]:
        # Tokenização das palavras do padrão.
        word_list = nltk.word_tokenize(pattern)  # Tokeniza as "perguntas" do usuário
        words.extend(word_list)  # Adiciona a lista tokenizada word_list à "words"
        documents.append((word_list, intent["tag"]))
        # Adição da classe ao conjunto de classes, se ainda não estiver presente.
        if intent["tag"] not in classes:
            classes.append(intent["tag"])

# Normalização das palavras (lemmatization) e remoção de letras ignoradas.
words = [lemmatizer.lemmatize(word) for word in words if word not in ignore_letters]
words = sorted(set(words))

# Ordenação das classes.
classes = sorted(set(classes))

# Salvando as palavras e classes normalizadas em arquivos pickle para uso posterior.
pickle.dump(words, open('words.pkl', 'wb'))
pickle.dump(classes, open('classes.pkl', 'wb'))


# Inicialização das listas para armazenar os dados de treinamento.
training = []
output_empty = [0] * len(classes)

# Processamento dos documentos para criar os dados de treinamento (bag of words).
for document in documents:
    bag = []
    word_patterns = document[0]
    word_patterns = [lemmatizer.lemmatize(word.lower()) for word in word_patterns]
    for word in words:
        # Criação da bag of words para cada padrão.
        bag.append(1) if word in word_patterns else bag.append(0)

    # Criação do vetor de saída (output_row) com 1 na posição da classe correspondente e 0 nas outras posições.
    output_row = list(output_empty)
    output_row[classes.index(document[1])] = 1

    # Adição do par bag-of-words e vetor de saída ao conjunto de treinamento.
    training.append([bag, output_row])

# Embaralhamento dos dados de treinamento.
random.shuffle(training)

# Divisão dos dados em conjuntos de treinamento e teste (por exemplo, 80% para treinamento e 20% para teste).
train_data, test_data = train_test_split(training, random_state=42, test_size=0.2)

# Conversão dos conjuntos de treinamento e teste em arrays numpy.
train_x = np.array([item[0] for item in train_data])
train_y = np.array([item[1] for item in train_data])
test_x  = np.array([item[0] for item in test_data])
test_y  = np.array([item[1] for item in test_data])

model = Sequential()
model.add(Dense(128, input_shape=(len(train_x[0]),), activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(64, activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(len(train_y[0]), activation="softmax"))

# Compilação do modelo usando o otimizador SGD (Gradiente Descendente Estocástico).
sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss="categorical_crossentropy", optimizer=sgd, metrics=["accuracy"])

hist = model.fit(train_x, train_y, epochs=30, batch_size=5, verbose=1)

model.save("chatbot_new_modell.h5", hist)








# Avaliar o modelo
predictions = model.predict(test_x)
predicted_classes = np.argmax(predictions, axis=1)
true_classes = np.argmax(test_y, axis=1)

# Relatório de classificação
report = classification_report(true_classes, predicted_classes, target_names=classes, output_dict=True)

# Converta o relatório de classificação em um DataFrame pandas
report_df = pd.DataFrame(report).transpose()

# Exibir o relatório de classificação em formato de tabela
print(report_df)

# Exiba a matriz de confusão como um mapa de calor
cm = confusion_matrix(true_classes, predicted_classes)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
plt.xlabel('Classe Prevista')
plt.ylabel('Classe Verdadeira')
plt.title('Matriz de Confusão')
plt.show()

# Plotar curvas de aprendizado
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(hist.history['accuracy'], label='accuracy')
plt.plot(hist.history['loss'], label='loss')
plt.xlabel('Epoch')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(hist.history['accuracy'], label='accuracy')
plt.xlabel('Epoch')
plt.legend()

plt.show()