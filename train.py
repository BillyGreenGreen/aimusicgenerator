import io
import os
import re
import glob
import time
import keras
import shutil
import keras.utils
import numpy as np
import urllib.request
import keras.layers as layers

from keras import optimizers
from keras.engine import input_layer
from keras.engine.saving import model_from_json
from selenium import webdriver

class ModelTrain:
    def load_file(self, filename):
        file = open(filename, 'r')
        content = file.read()
        file.close()
        return content

    def sample(self, preds, temperature=1.0):
        preds = np.asarray(preds).astype("float64")
        preds = np.log(preds) / temperature
        exp_preds = np.exp(preds)
        preds = exp_preds / np.sum(exp_preds)
        probas = np.random.multinomial(1, preds, 1)
        return np.argmax(probas)
        
    def train(self, epochs, batch_size):
        content = self.load_file("data/ABC_cleaned/input.txt")
        chars = sorted(list(set(content)))
        char_indices = dict((c, i) for i, c in enumerate(chars))
        indices_char = dict((i, c) for i, c in enumerate(chars))

        maxlen = 40
        step = 3
        sentences = []
        next_chars = []
        for i in range(0, len(content) - maxlen, step):
            sentences.append(content[i:i+maxlen])
            next_chars.append(content[i+maxlen])

        x = np.zeros((len(sentences), maxlen, len(chars)), dtype=np.bool)
        y = np.zeros((len(sentences), len(chars)), dtype=np.bool)
        for i, sentence in enumerate(sentences):
            for t, char in enumerate(sentence):
                x[i, t, char_indices[char]] = 1
            y[i, char_indices[next_chars[i]]] = 1

        model = keras.Sequential()
        model.add(input_layer.InputLayer(input_shape=(maxlen, len(chars))))
        model.add(layers.LSTM(128))
        model.add(layers.Dense(len(chars), activation='softmax'))
        optimizer = optimizers.RMSprop(lr=0.01)
        model.compile(loss="categorical_crossentropy", optimizer=optimizer)
        for epoch in range(epochs):
            model.fit(x, y, batch_size=batch_size, epochs=1)
            print()
            print("Generating text after epoch %d" % epoch)

            start_index = np.random.randint(0, len(content) - maxlen - 1)
            for diversity in [0.2, 0.5, 1.0, 1.2]:
                print("...Diversity:", diversity)

                generated = ""
                sentence = content[start_index:start_index+maxlen]
                print('...Generating with seed: "' + sentence + '"')

                for i in range(400):
                    x_pred = np.zeros((1, maxlen, len(chars)))
                    for t, char in enumerate(sentence):
                        x_pred[0, t, char_indices[char]] = 1.0
                    preds = model.predict(x_pred, verbose=0)[0]
                    next_index = self.sample(preds, diversity)
                    next_char = indices_char[next_index]
                    sentence = sentence[1:] + next_char
                    generated += next_char
                print("...Generated: ", generated)
                print()

                topSeven = []
                contentSong = []
                fullAbc = ""
                count = 0
                if "X:" in generated:
                    index = generated.find("X:")
                    generated = generated[index:]
                    genList = generated.split('\n')
                    for line in genList:
                        if line.startswith(("X:", "T:", "%", "S:", "M:", "L:", "K:")):
                            topSeven.append(line)
                            count+=1
                        if count > 6:
                            if line and generated[count+1]:
                                contentSong.append(line)
                            else:
                                contentSong.append(line)
                                break
                        
                    if len(topSeven) == 7:
                        if "/" not in topSeven[1][2:] or '\\' not in topSeven[1][2:]:
                            for c in topSeven:
                                fullAbc += c + "\n"
                            for c in contentSong:
                                fullAbc += c + "\n"
                            with open("good_reels.txt", 'a') as f:
                                f.write("\n" + fullAbc)
                                f.close()

                            driver = webdriver.Chrome()
                            driver.get("https://www.mandolintab.net/abcconverter.php")
                            textarea = driver.find_element_by_name("abc")
                            submit = driver.find_element_by_name("submit")
                            textarea.clear()
                            textarea.send_keys(fullAbc)
                            submit.click()
                            time.sleep(2)
                            midi = driver.find_element_by_xpath("//div[contains(@id, 'converter_display')]/a")
                            midi.click()
                            picture = driver.find_element_by_xpath("//div[contains(@id, 'converter_display')]/a/img").get_attribute('src')
                            filePath = "F:/Uni/Year_3/FYP/gen/Songs/" + topSeven[1][2:] + "/"
                            if os.path.exists(filePath):
                                urllib.request.urlretrieve(picture, filePath + topSeven[1][2:] + ".png")
                                list_of_files = glob.glob("C:/Users/theca/Downloads/" + '/*mid')
                                latest_file = max(list_of_files, key=os.path.getctime)
                                shutil.move(latest_file, filePath + topSeven[1][2:] + ".mid")
                                driver.close()
                            else:
                                os.mkdir(filePath)
                                urllib.request.urlretrieve(picture, filePath + topSeven[1][2:] + ".png")
                                list_of_files = glob.glob("C:/Users/theca/Downloads/" + '/*mid')
                                latest_file = max(list_of_files, key=os.path.getctime)
                                shutil.move(latest_file, filePath + topSeven[1][2:] + ".mid")
                                driver.close()
                        break     