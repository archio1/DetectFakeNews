
import nltk
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import cross_val_score
from keras.preprocessing.text import Tokenizer
from keras_preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import LSTM, Dropout, Dense, Embedding
from sklearn.model_selection import train_test_split
from keras.models import load_model
from joblib import dump, load
import pickle
from nltk.corpus import stopwords
nltk.download('stopwords')


class DetectFake:

    def __init__(self, model_name, data=None, model_path=None):
        self.model_name = self.check_model_name(model_name)
        self.init_vectorizer = self.initialization_vector()
        if model_path:
            self.load_model(model_path)
        else:
            self.data = data
            self.model = None
            self.data_collection = None
            self.vectorization_data = None
            self.word_index = None
            self.vocab_size = None
            self.padded_seq = None
            self.embedding_matrix = None

    def initialization_vector(self):
        if self.model_name == 'passive_aggressive':
            return TfidfVectorizer(stop_words='english', max_df=0.7)
        elif self.model_name == 'neural_network':
            return Tokenizer()# have parameter of length size


    def check_model_name(self, name):
        """ADD HERE DESCRIPTION"""

        models_tuple = ('passive_aggressive', 'neural_network')
        if name in models_tuple:
            return name
        else:
            raise Exception('Model name not found')

    def train(self):
        """ADD HERE DESCRIPTION"""

        if self.model_name == 'passive_aggressive':
            self.passive_aggressive_classifier()
        elif self.model_name == 'neural_network':
            self.neural_network()

    def prepare_data(self):
        """ADD HERE DESCRIPTION"""

        self.data['label'] = self.data['label'].replace({0: 'Real', 1: 'Fake'})

        x_train, x_test, y_train, y_test = train_test_split(
            self.data['text'], self.data['label'], test_size=0.2, random_state=7
        )

        self.data_collection = {'x_train': x_train, 'x_test': x_test, 'y_train': y_train, 'y_test': y_test}

    def vectorization_of_text(self):
        """ADD HERE DESCRIPTION"""

        self.vectorization_data = {
            'vec_train': self.init_vectorizer.fit_transform(self.data_collection['x_train'].values.astype('U')),
            'vec_test': self.init_vectorizer.transform(self.data_collection['x_test'].values.astype('U'))
        }

    def passive_aggressive_classifier(self):
        """ADD HERE DESCRIPTION"""

        pac = PassiveAggressiveClassifier(max_iter=50)
        pac.fit(self.vectorization_data['vec_train'], self.data_collection['y_train'])
        self.model = pac
        self.model.init_vectorizer = self.init_vectorizer

    def accuracy(self):
        """ADD HERE DESCRIPTION"""

        y_pred = self.model.predict(self.vectorization_data['vec_test'])
        score = accuracy_score(self.data_collection['y_test'], y_pred)
        print(f'Accuracy: {round(score * 100, 2)}%')
        print(confusion_matrix(self.data_collection['y_test'], y_pred, labels=['Real', 'Fake']))

    def save_model(self, path_dir='../resources'):
        if self.model_name == 'passive_aggressive':
            dump(self.model, path_dir+'/passive_aggressive_model')
        elif self.model_name == 'neural_network':
            self.model.save(path_dir+'/neural_model')

    def load_model(self, path_to_model):
        if self.model_name == 'passive_aggressive':
            self.model = load(path_to_model)
        elif self.model_name == 'neural_network':
            self.model = load_model(path_to_model)

    def cleaning_news_for_neural_net(self):
        self.data = self.data.drop(columns=['id', 'title', 'author'], axis=1)# unite or delete columns
        self.data = self.data.dropna(axis=0)
        self.data['clean_news'] = self.data['text'].str.lower()
        self.data['clean_news'] = self.data['clean_news'].str.replace(r'[^A-Za-z0-9\s]', '')
        self.data['clean_news'] = self.data['clean_news'].str.replace(r'\n', '')
        self.data['clean_news'] = self.data['clean_news'].str.replace(r'\s+', ' ')
        # self.data['clean_news'] = self.data['text'].lower()
        # self.data['clean_news'] = self.data['clean_news'].replace(r'[^A-Za-z0-9\s]', '')
        # self.data['clean_news'] = self.data['clean_news'].replace(r'\n', '') #str
        # self.data['clean_news'] = self.data['clean_news'].replace(r'\s+', ' ')

    def vector_text_for_neural_network(self):
        # self.tokenizer = Tokenizer()
        self.init_vectorizer.fit_on_texts(self.data['clean_news'])
        self.word_index = self.init_vectorizer.word_index
        self.vocab_size = len(self.word_index)
        sequences = self.init_vectorizer.texts_to_sequences(self.data['clean_news'])
        self.padded_seq = pad_sequences(sequences, maxlen=600, padding='post', truncating='post')

    def create_matrix(self):
        embedding_index = {}
        with open('../resources/glove.6B.100d.txt', encoding='utf-8') as f:
            for line in f:
                values = line.split()
                word = values[0]
                coefs = np.asarray(values[1:], dtype='float32')
                embedding_index[word] = coefs

        self.embedding_matrix = np.zeros((self.vocab_size + 1, 100))
        for word, i in self.word_index.items():
            embedding_vector = embedding_index.get(word)
            if embedding_vector is not None:
                self.embedding_matrix[i] = embedding_vector

    def init_test_train_split(self):
        x_train, x_test, y_train, y_test = train_test_split(
            self.padded_seq, self.data['label'], test_size=0.20, random_state=42, stratify=self.data['label'])
        self.data_collection = {'x_train': x_train, 'x_test': x_test, 'y_train': y_train, 'y_test': y_test}

    def neural_network(self):
        model = Sequential([
            Embedding(self.vocab_size + 1, 100, weights=[self.embedding_matrix], trainable=False),
            Dropout(0.2),
            LSTM(128),
            Dropout(0.2),
            Dense(256),
            Dense(1, activation='sigmoid')
        ])
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics='accuracy')
        print(model.summary())
        model.fit(
            self.data_collection['x_train'], self.data_collection['y_train'],
            epochs=10, batch_size=256, validation_data=(self.data_collection['x_test'], self.data_collection['y_test'])
        )
        self.model = model

    def predict(self, newtext):
        vec_newtest = self.model.init_vectorizer.transform([newtext])
        y_pred1 = self.model.predict(vec_newtest)
        return y_pred1[0]

    def predict_by_neural_network(self):
        # self.data = self.data.drop(columns=['subject', 'data'], axis=1)# unite or delete columns
        self.cleaning_news_for_neural_net()
        self.vector_text_for_neural_network()
        self.create_matrix()
        y_pred1 = self.model.predict(np.array(self.padded_seq))
        return y_pred1[0]

if __name__ == '__main__':
    data_frame = pd.read_csv('../resources/train.csv')
    data_frame_true = pd.read_csv('../True.csv')
    # data_frame_true['label'] = 'Real'
    # model_passive_aggressive = DetectFake('passive_aggressive', data=data_frame)
    # model_passive_aggressive.prepare_data()
    # model_passive_aggressive.vectorization_of_text()
    # model_passive_aggressive.passive_aggressive_classifier()
    # model_passive_aggressive.train()
    # model_passive_aggressive.save_model()
    model_neural_network = DetectFake('neural_network', data_frame)




