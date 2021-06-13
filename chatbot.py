#creating a chatbot
#Import the library


import nltk
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
from nltk.tokenize import sent_tokenize, word_tokenize
import pickle
import numpy as np
import os


from keras.models import load_model
model = load_model('chatbot_model.h5')
import json
import random
intents = json.loads(open('intents.json').read())
words = pickle.load(open('words.pkl','rb'))
classes = pickle.load(open('classes.pkl','rb'))



def clean_up_sentence(sentence):
    # tokenize the pattern - split words into array
    sentence_words = nltk.word_tokenize(sentence)
    # stem each word - create short form for word
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

# return bag of words array: 0 or 1 for each word in the bag that exists in the sentence

def bow(sentence, words, show_details=True):
    # tokenize the pattern
    sentence_words = clean_up_sentence(sentence)
    # bag of words - matrix of N words, vocabulary matrix
    bag = [0]*len(words)  
    for s in sentence_words:
        for i,w in enumerate(words):
            if w == s: 
                # assign 1 if current word is in the vocabulary position
                bag[i] = 1
                if show_details:
                    print ("found in bag: %s" % w)
    return(np.array(bag))

def predict_class(sentence, model):
    # filter out predictions below a threshold
    p = bow(sentence, words,show_details=False)
    res = model.predict(np.array([p]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i,r] for i,r in enumerate(res) if r>ERROR_THRESHOLD]
    # sort by strength of probability
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
    return return_list

def getResponse(ints, intents_json):
    tag = ints[0]['intent']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if(i['tag']== tag):
            result = random.choice(i['responses'])
            break
    return result

def chatbot_response(msg):
    ints = predict_class(msg, model)
    res = getResponse(ints, intents)
    return res

from tkinter import *

def send():
    message= messageWindow.get("1.0",'end-1c').strip()
    messageWindow.delete("0.0",END)
    if  message != '':
        chatWindow.config(state=NORMAL)
        chatWindow.insert(END, "You: " + message + '\n\n')
        chatWindow.config(foreground="#0B0B0C", font=("Arial", 12 ))
        res = chatbot_response(message)
        chatWindow.insert(END, "Bot: " + res + '\n\n')
        chatWindow.config(state=DISABLED)
        chatWindow.yview(END)
        
root = Tk()
root.title("Covid-19 ChatBot")
root.geometry("400x500")
root.resizable(width=FALSE, height=FALSE)
#Creating a Menu
main_menu = Menu(root)
# Create the submenu 
file_menu = Menu(root)
# Add commands to submenu
file_menu.add_command(label="New..")
file_menu.add_command(label="Save As..")
file_menu.add_command(label="Exit")
main_menu.add_cascade(label="File", menu=file_menu)
#Add the rest of the menu options to the main menu
main_menu.add_command(label="Edit")
main_menu.add_command(label="Quit")
root.config(menu=main_menu)

#chat window    
chatWindow = Text(root, bd=1, bg="white",  width="50", height="8", font=("Arial"))
chatWindow.config(state=DISABLED)
chatWindow.place(x=6,y=6, height=385, width=370)
#scroll bar
scrollbar = Scrollbar(root, command=chatWindow.yview)
chatWindow['yscrollcommand'] = scrollbar.set
scrollbar.place(x=375,y=5, height=385)
#button
Button= Button(root, text="Send",  width="12", height=5, bd=0, bg="#0080ff", activebackground="#00bfff",foreground='#ffffff',font=("Arial", 12), command= send)
Button.place(x=6, y=400, height=88)
#message window
messageWindow = Text(root, bd=0, bg="white",width="30", height="4", font=("Arial", 16))
messageWindow.place(x=128, y=400, height=88, width=220)




root.mainloop()
