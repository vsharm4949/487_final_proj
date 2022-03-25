import json

import requests
import os
import pandas as pd
import time
from random import randrange

# from translate import Translator

from googletrans import Translator


# API_TOKEN = "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"

# headers = {"Authorization": f"Bearer {API_TOKEN}"}
# API_URL = "https://api-inference.huggingface.co/models/Helsinki-NLP/opus-mt-zh-en"
# headers = {"Authorization": f"Bearer {API_TOKEN}"}
#
# def query(payload):
#     data = json.dumps(payload)
#     response = requests.request("POST", API_URL, headers=headers, data=data)
#     return json.loads(response.content.decode("utf-8"))

def load_data(train_path):
    df = pd.read_csv(train_path)

    return df

def load_dataset(train_path):
    data = load_data(train_path)
    data_list = data['joke'].tolist()

    return data_list, data['id'].tolist(), data['label'].tolist()

# def translate_to_English():
    # translator = Translator(to_lang="en")
    # translation = translator.translate("This is a pen.")

# def clean_list(data):
#     list_output = []
#     for item in data:
#         x = item['translation_text']
#         # print(type(x))
#         list_output.append(x)
#
#     return list_output



if __name__ == '__main__':
    # X = ["我是美女", "你是苹果",]

    cur = '/'.join(os.path.abspath(__file__).split('/')[:-1])
    train_path = os.path.join(cur, 'data/balanced_test.csv')

    list_X, list_id, list_label = load_dataset(train_path)

    total_round = int(len(list_X) / 500)
    # last_round = len(list_X) % 500
    # print(len(list_X), total_round, last_round)


    #
    # # sentences = ['你是谁', '我和你都很不错', '好困想要睡觉，啊啊啊困']
    #
    # sentences = list_X[:2]
    # list_id = list_id[:400]
    # list_label = list_label[:400]

    translator = Translator()

    last_index = 0
    all_result = []

    for i in range(total_round):
        index = (i + 1) * 500
        print("index: ",index)
        sentences = list_X[last_index:index]

        print("lenth of dataset now: ",len(sentences))
        print("starting to translate ---- ")
        result = translator.translate(sentences, src='zh-cn', dest='en')
        print("--- end of translation")

        #add to all result
        all_result.extend(result)

        last_index = index

        print("start sleeping: ")
        time.sleep(randrange(5))
        print("end sleeping: ",time.time())


        print("-------end of index: ", i, "-------")
        print("")

    translated_list = []
    original_list = []
    for trans in all_result:
        translated_list.append(trans.text)
        original_list.append(trans.origin)

    try:
        assert(len(translated_list) == len(list_id))
        assert(len(translated_list) == len(list_label))

        print("equal")
    #
        df =  pd.DataFrame({'id':list_id,'joke':translated_list, 'label':list_label})
        df.to_csv('translate_to_English_test.csv')
    except:
        df = pd.DataFrame({'joke': translated_list, 'original': original_list})
        df.to_csv('output.csv')


    #[{'translation_text': "I'm a pretty girl."}, {'translation_text': "You're an apple."}]
# <class 'dict'>
#
#     data = query(X)
#     print((data))
#
#     print(type(data[0]))