import random
import json
import pickle
import numpy as np
import re

import nltk
from nltk.stem import WordNetLemmatizer

from tensorflow import keras
from keras.models import load_model

lemmatizer = WordNetLemmatizer()
intents = json.loads(open('intents.json', encoding="utf8").read())

words = pickle.load(open('words.pkl','rb'))
classes = pickle.load(open('classes.pkl','rb'))
model = load_model('chatbot_model.h5')

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

context = []

def predict_class(sentence):
    bow = bag_of_words(sentence)
    res = model.predict(np.array([bow]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]

    results.sort(key=lambda x:[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({'intents': classes[r[0]], 'probability': str(r[1])})

    context.append({'sentence': sentence, 'intents_list': return_list})
    return return_list

def get_response(intents_list, intents_json):
    tag = intents_list[0]['intents']

    for context_item in reversed(context):
        if context_item['sentence'] == message:
            tag = context_item['intents_list'][0]['intents']
            break

    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if i['tag'] == tag:
            result = random.choice(i['responses'])
            break
    return result


def extract_info(text):
    
    user_name    = None
    user_email   = None
    user_cpf     = None
    user_cnumber = None
    
    name_match = re.search(r"\b(?:[A-Z][a-zA-ZÀ-ÿ'-]*(?:\s(?:d[aeio]s?\s)?[A-Z][a-zA-ZÀ-ÿ'-]*)+)\b", text)
    #name_match = re.search(r"\b(?:[A-Z][a-zA-ZÀ-ÿ'-]*(?:\s(?:de\s)?[A-Z][a-zA-ZÀ-ÿ'-]*)+)\b", text)   //Funciona Nome completo e com "de" (mas não "da" "di" "do")
    if name_match:
        user_name = name_match.group()

    email_match = re.search(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b", text)
    if email_match:
        user_email = email_match.group()

    cpf_match = re.search(r"\b(?:\d{3}\.\d{3}\.\d{3}-\d{2}|\d{11}|\d{3}\ \d{3}\ \d{3} \d{2})\b", text)
    if cpf_match:
        user_cpf = cpf_match.group()
    
    cnumber_match = re.search(r"\b(?:\(\d{2}\)\s\d{5}-\d{4}|\d{2}\s\d{5}\s\d{4}|\d{2}\s\d{9})\b|\d{2}\s\d{5}-\d{4}", text)
    if cnumber_match:
        user_cnumber = cnumber_match.group()


    
    return user_name, user_email, user_cpf, user_cnumber


print('Go, bot is running!')
finish = False
while not finish:
    message = input("> ")
    if message == "STOP":
        finish = True
    else:
        name, email, cpf, number = extract_info(message)

        if name:
            print("Nome reconhecido:", name)
        if email:
            print("Email reconhecido:", email)  
        if cpf:
            print("CPF reconhecido:", cpf)  
        if number:
            print("Tel. reconhecido:", number)

        ints = predict_class(message)
        res = get_response(ints, intents)
        print (res)

