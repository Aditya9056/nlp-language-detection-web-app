import sys
import string
import re
import numpy as np
import pickle


class Lang_detector:
    def detect(self, text):

        translate_table = dict((ord(char), None) for char in string.punctuation)
        # print(text)

        # Model Loading
        global lrLangDetectModel
        lrLangDetectFile = open("LRModel.pckl", "rb")
        lrLangDetectModel = pickle.load(lrLangDetectFile)
        lrLangDetectFile.close()

        text = " ".join(text.split())
        text = text.lower()
        text = re.sub(r"\d+", "", text)
        text = text.translate(translate_table)
        pred = lrLangDetectModel.predict([text])
        prob = lrLangDetectModel.predict_proba([text])

        # print(pred)
        return pred[0]


# Driver
# print(lang_detect(sys.argv[1]))
# print("Argument List:", str(sys.argv))