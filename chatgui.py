import json
import pickle
import random
import tkinter as tk
from tkinter import *

import nltk
import numpy as np
from keras import models
from nltk.stem import WordNetLemmatizer

# Download NLTK data
nltk.download("punkt")
nltk.download("wordnet")

# Load model, words, and classes
lemmatizer = WordNetLemmatizer()
model = models.load_model("chatbot_model.h5")
intents = json.loads(open("intents.json").read())
words = pickle.load(open("words.pkl", "rb"))
classes = pickle.load(open("classes.pkl", "rb"))


def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words


def bow(sentence, words, show_details=True):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s:
                bag[i] = 1
                if show_details:
                    print(f"found in bag: {w}")
    return np.array(bag)


def predict_class(sentence, model):
    p = bow(sentence, words, show_details=False)
    res = model.predict(np.array([p]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
    return return_list


def getResponse(ints, intents_json):
    tag = ints[0]["intent"]
    list_of_intents = intents_json["intents"]
    for i in list_of_intents:
        if i["tag"] == tag:
            result = random.choice(i["responses"])
            break
    return result


def chatbot_response(msg):
    ints = predict_class(msg, model)
    res = getResponse(ints, intents)
    return res


def send(event=None):
    msg = EntryBox.get("1.0", "end-1c").strip()
    EntryBox.delete("0.0", END)

    if msg != "":
        ChatLog.config(state=tk.NORMAL)
        ChatLog.insert(tk.END, "You: " + msg + "\n\n")
        ChatLog.config(foreground="#442265", font=("Verdana", 12))

        res = chatbot_response(msg)
        ChatLog.insert(tk.END, "Bot: " + res + "\n\n")

        ChatLog.config(state=tk.DISABLED)
        ChatLog.yview(tk.END)


# Create GUI with tkinter
base = tk.Tk()
base.title("Chatbot")
base.geometry("500x600")
base.resizable(width=False, height=False)

# Set up colors and fonts
BG_COLOR = "#f5f5f5"
TEXT_COLOR = "#444444"
FONT = "Helvetica 14"
FONT_BOLD = "Helvetica 13 bold"

# Create Chat window
ChatLog = tk.Text(base, bd=0, bg=BG_COLOR, fg=TEXT_COLOR, font=FONT, wrap=tk.WORD)
ChatLog.config(state=tk.DISABLED)

# Bind scrollbar to Chat window
scrollbar = tk.Scrollbar(base, command=ChatLog.yview, cursor="heart")
ChatLog["yscrollcommand"] = scrollbar.set

# Create Button to send message
SendButton = tk.Button(
    base,
    font=FONT_BOLD,
    text="Send",
    width="12",
    height=2,
    bd=0,
    bg="#32de97",
    activebackground="#3c9d9b",
    fg="#ffffff",
    command=send,
)

# Create the box to enter message
EntryBox = tk.Text(base, bd=0, bg="#ffffff", fg=TEXT_COLOR, font=FONT, height=2)

# Bind Enter key to send message
EntryBox.bind("<Return>", send)

# Place all components on the screen
scrollbar.place(x=476, y=6, height=486)
ChatLog.place(x=6, y=6, height=486, width=470)
EntryBox.place(x=6, y=501, height=90, width=380)
SendButton.place(x=390, y=501, height=90)

base.mainloop()
