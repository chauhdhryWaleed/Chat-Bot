import numpy as np
import tensorflow as tf
import nltk
from nltk.stem import WordNetLemmatizer
import json
import pickle
import random


# Load the saved model
model = tf.keras.models.load_model('chatbot_model.keras')
intents = json.loads(open('data_set.json').read())
# Load preprocessed data (words and classes)
words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))

# Initialize lemmatizer
lemmatizer = WordNetLemmatizer()

def clean_up_sentence(sentence):
    sentence_Words=nltk.word_tokenize(sentence)
    sentence_Words=[lemmatizer.lemmatize(word) for word in sentence_Words]
    return sentence_Words

def bag_of_words(sentence):
    sentence_words=clean_up_sentence(sentence)
    bag=[0]*len(words)
    for w in sentence_words:
        for i,word in enumerate(words):
            if word==w :
                bag[i]=1
    return np.array(bag)

def predict_class(sentence):
    bow=bag_of_words(sentence)
    res=model.predict(np.array([bow]))[0]
    error_threshold= 0.33

    results =[[i,r] for i,r in enumerate(res) if r>error_threshold]
    results.sort(key=lambda  x : x[1],reverse=True)
    return_list=[]

    for r in results:
        return_list.append({'intent':classes[r[0]], 'probability':str(r[1])})

    return return_list


def get_response(intents_list, intents_json):

    tag=intents_list[0]['intent']
    list_of_intents=intents_json['intents']
    for i in list_of_intents:
        if i['tag']==tag:
            result=random.choice(i['responses'])
            break
    return result

while True:
    message=input("Human: ")
    print("\n")
    ints=predict_class(message)
    print(ints)
    res=get_response(ints,intents)
    print("Bot: ",res)
