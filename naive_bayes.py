import pandas as pd
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
from IPython.display import display
import os
from collections import defaultdict
from sklearn.model_selection import train_test_split

import jieba
from jieba import posseg


def load_data(train_path):
    df = pd.read_csv(train_path)
    # df['label'] = np.where(df["label"] == 1, 0, df["label"])
    # df['label'] = np.where(df["label"] == 3, 1, df["label"])
    # df['label'] = np.where(df["label"] == 5, 2, df["label"])

    return df

class NaiveBayes():
    """Naive Bayes classifier."""

    def __init__(self):
        super().__init__()
        cur = '/'.join(os.path.abspath(__file__).split('/')[:-1])
        self.train_path = os.path.join(cur, 'data/task2_train.csv')

        self.ngram_count = []
        self.total_count = []
        self.category_prob = []
        self.vectorizer = None

        self.ngram_count_dic = defaultdict(int)
        self.totoal_data = 0
        self.data_frame = None


    '''fit the data'''
    def fit(self, data):
        
        # for index, row in data.iterrows():
        #     text = row["joke"]
        #     words = posseg.cut(text)
        #     for w in words:
        #         w.word
        #     break
    
        self.vectorizer = CountVectorizer(min_df=3, max_df=0.8, ngram_range=(1, 2))
        print(self.vectorizer)
        
        
        occurence_matrix = self.vectorizer.fit_transform(data['joke'])
        # print(self.vectorizer.get_feature_names_out()[20:70])

        label_array = np.array(data['label'])
        # print("occurence_matrix", occurence_matrix)

        for i in range(4):
            # this gives us the lable meets current lable

            self.ngram_count.append(np.sum(occurence_matrix[label_array == i], axis=0))
            self.total_count.append(np.sum(occurence_matrix[label_array == i]))

        # prior probability is p(word|category)
        article_total = np.shape(data)[0]
        for i in range(4):
            category_num = sum(data['label'] == i)
            self.category_prob.append(category_num / article_total)


    def calculate_prob(self, docs, c_i, alpha):
        prob = np.zeros(len(docs))

        doc_total_count = self.total_count[c_i]

        #X array is words in the docs and in the original trainning set
        X = self.vectorizer.transform(docs)

        X = X.toarray()

        #column-wise is the number of unique words in docs
        V = np.shape(X)[1]

        for row in range(np.shape(X)[0]):
            curr_x_row = X[row,:]
            index = np.where(curr_x_row > 0)

            frq = curr_x_row[curr_x_row > 0]
            testcol = np.shape(frq)[0]

            nk = self.ngram_count[c_i][0,index]
            probability_log = np.log((nk+ alpha) / (doc_total_count + alpha * V))

            probability_log = np.array(probability_log[0])
            probability_final = frq * probability_log


            probability_final = np.sum(probability_final)
            prob[row] = (np.log(self.category_prob[c_i]) + probability_final)

        return prob



    def predict(self, docs, alpha):
        prediction = [None] * len(docs)

        #used to be 4 for final one
        prob_arr = np.ones((len(self.category_prob),len(docs)))
        # print(prob_arr)
        for i in range(len(self.category_prob)):
            # print(i)
            probability = self.calculate_prob(docs,i,alpha)
            prob_arr[i] = probability


        prob_arr = np.transpose(prob_arr)
        for i in range(len(prediction)):
            prediction[i] = np.argmax(prob_arr[i])

        return prediction


def evaluate(predictions, labels):
    accuracy, mac_f1, mic_f1 = None, None, None

    table = np.zeros((np.max(predictions) + 1,np.max(labels) + 1))

    # col is label
    # row is predict
    for pred in range(len(labels)):
        row = predictions[pred]
        col = labels[pred]
        # print(row,col)
        table[row][col] += 1

    F1 = 0
    upper = 0
    transpose_table = np.transpose(table)

    for row in range(len(table)):
        precision = (table[row][row]) / (np.sum(table[row]))
        recall = (transpose_table[row][row]) / (np.sum(transpose_table[row]))
        w = 2.0 * (precision * recall) / (precision + recall)
        if np.isnan(w) ==  False:
            F1+=w
        upper += table[row][row]
    mic_f1 = upper / np.sum(table)


    mac_f1 = F1 / (np.max(labels) + 1)

    accuracy = mic_f1
    print("precision", precision)
    print("recall", recall)
    return accuracy, mac_f1, mic_f1


if __name__ == '__main__':
    '''load data'''
    cur = '/'.join(os.path.abspath(__file__).split('/')[:-1])
    train_path = os.path.join(cur, 'data/task1_train.csv')

    data = load_data(train_path)
    train_data, test_data = train_test_split(data, test_size=0.3, random_state=42)
    print(test_data.label.value_counts())
    print(train_data.label.value_counts())
    
    # for index, row in train_data.iterrows():
    #     text = row["joke"]

    #     words = posseg.cut(text)
    #     for w in words:
    #         print('%s %s' % (w.word, w.flag))
       
    #     print(words)
    #     break
    # words = posseg.cut(train_data)
    



    naive_bayes = NaiveBayes()
    naive_bayes.fit(train_data)
    
    
    alpha = 0.4
    predictions = naive_bayes.predict(test_data.joke.tolist(), alpha)
    labels = test_data['label'].tolist()
    accuracy, mac_f1, mic_f1 = evaluate(predictions, labels)
    print(f"Accuracy: {accuracy}")
    print(f"Macro f1: {mac_f1}")
    print(f"Micro f1: {mic_f1}")
    #
