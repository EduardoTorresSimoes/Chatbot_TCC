import random
import json
import pickle
import numpy as np
import extra_functions as ef

import nltk
from nltk.stem import WordNetLemmatizer

from tensorflow import keras
from keras.models import load_model

lemmatizer = WordNetLemmatizer()
intents = json.loads(open("intents.json", encoding="utf8").read())

words = pickle.load(open("words.pkl", "rb"))
classes = pickle.load(open("classes.pkl", "rb"))
model = load_model("chatbot_model.h5")


def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word) for word in sentence_words]
    return sentence_words


def bag_of_words(sentence):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for w in sentence_words:
        for i, word in enumerate(words):
            if word == w:
                bag[i] = 1
    return np.array(bag)


def handle_context(context):
    actions = {
        "CadastroPI": ef.protocolo_cadastro,
        "LoginPI": ef.protocolo_login
    }

    if context in actions:
        result = actions.get(context)  
        return result()
    else:
        return "Contexto não mapeado: {}".format(context)
    

def predict_class(sentence):
    bow = bag_of_words(sentence)
    res = model.predict(np.array([bow]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]

    results.sort(key=lambda x: [1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({"intents": classes[r[0]], "probability": str(r[1])})

    return return_list


def get_response(intents_list, intents_json):
    tag = intents_list[0]["intents"]
    list_of_intents = intents_json["intents"]
    
    for i in list_of_intents:
        if i["tag"] == tag:
            if "context" in i:
                context = i["context"]
                handle_context(context)
                return None  # Retorna None quando um contexto é encontrado
            result = random.choice(i["responses"])
            return result
    
    for i in list_of_intents:
        if i["tag"] == "NaoEntendi":
            result = random.choice(i["responses"])
            return result
    
    return "Desculpe, não entendi a mensagem. Poderia repetir de outra maneira?"


print("Go, bot is running!")
finish = False
while not finish:
    message = input("\n> ").lower()
    if message == "STOP":
        finish = True
    else:
        ints = predict_class(message)
        response = get_response(ints, intents)
        if response is not None:
            print(response)