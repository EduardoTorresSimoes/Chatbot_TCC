import random
import json
import pickle
import numpy as np

import nltk
from nltk.stem import WordNetLemmatizer

from tensorflow import keras

from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.optimizers.legacy import SGD

# Inicialização do lemmatizer (usado para normalizar as palavras).
lemmatizer = WordNetLemmatizer()

# Carregando os dados de intents do arquivo JSON.
intents = json.loads(open('intents.json', encoding="utf8").read())

# Inicialização das listas para armazenar palavras, classes e documentos.
words = []
classes = []
documents = []
ignore_letters = ['.', '!', '?', ',', '_']

# Iteração através dos intents e seus padrões para processar as palavras.
for intent in intents['intents']: 
    for pattern in intent['patterns']:
        # Tokenização das palavras do padrão.
        word_list = nltk.word_tokenize(pattern) # Tokeniza as "perguntas" do usuário 
        words.extend(word_list) # Adiciona a lista tokenizada word_list à "words"
        documents.append((word_list, intent['tag']))
        # Adição da classe ao conjunto de classes, se ainda não estiver presente.
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

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

# Conversão do conjunto de treinamento para um array numpy.
training = np.array(training, dtype=object)

# Separação dos dados de treinamento (entradas e saídas) em listas distintas.
train_x = list(training[:, 0])
train_y = list(training[:, 1])

model = Sequential()
model.add(Dense(128, input_shape=(len(train_x[0]),), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(train_y[0]), activation='softmax'))

# Compilação do modelo usando o otimizador SGD (Gradiente Descendente Estocástico).
sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

hist = model.fit(np.array(train_x), np.array(train_y), epochs=50, batch_size=5, verbose=1)

model.save('chatbot_model.h5', hist)