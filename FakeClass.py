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
from nltk.corpus import stopwords
nltk.download('stopwords')


class DetectFake:

    def __init__(self, model_name, data):
        self.model_name = self.check_model_name(model_name)
        self.data = data
        self.model = None
        self.data_collection = None
        self.vectorization_data = None
        self.word_index = None
        self.vocab_size = None
        self.padded_seq = None
        self.embedding_matrix = None


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
        if self.model_name == 'neural_network':
            self.model = self.neural_network()

    def prepare_data(self):
        """ADD HERE DESCRIPTION"""

        self.data['label'] = self.data['label'].replace({0: 'Real', 1: 'Fake'})

        x_train, x_test, y_train, y_test = train_test_split(
            self.data['text'], self.data['label'], test_size=0.2, random_state=7
        )

        self.data_collection = {'x_train': x_train, 'x_test': x_test, 'y_train': y_train, 'y_test': y_test}

    def vectorization_of_text(self):
        """ADD HERE DESCRIPTION"""

        tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
        self.vectorization_data = {
            'vec_train': tfidf_vectorizer.fit_transform(self.data_collection['x_train'].values.astype('U')),
            'vec_test': tfidf_vectorizer.transform(self.data_collection['x_test'].values.astype('U'))
        }

    def passive_aggressive_classifier(self):
        """ADD HERE DESCRIPTION"""

        pac = PassiveAggressiveClassifier(max_iter=50)
        pac.fit(self.vectorization_data['vec_train'], self.data_collection['y_train'])
        self.model = pac

    def accuracy(self):
        """ADD HERE DESCRIPTION"""

        y_pred = self.model.predict(self.vectorization_data['vec_test'])
        score = accuracy_score(self.data_collection['y_test'], y_pred)
        print(f'Accuracy: {round(score * 100, 2)}%')
        print(confusion_matrix(self.data_collection['y_test'], y_pred, labels=['Real', 'Fake']))

    def cleaning_news_for_neural_net(self):
        self.data = self.data.drop(columns=['id', 'title', 'author'], axis=1)
        self.data = self.data.dropna(axis=0)
        self.data['clean_news'] = self.data['text'].str.lower()
        self.data['clean_news'] = self.data['clean_news'].str.replace(r'[^A-Za-z0-9\s]', '')
        self.data['clean_news'] = self.data['clean_news'].str.replace(r'\n', '')
        self.data['clean_news'] = self.data['clean_news'].str.replace(r'\s+', ' ')

    def vector_text_for_neural_network(self):
        tokenizer = Tokenizer()
        tokenizer.fit_on_texts(self.data['clean_news'])
        self.word_index = tokenizer.word_index
        self.vocab_size = len(self.word_index)
        sequences = tokenizer.texts_to_sequences(self.data['clean_news'])
        self.padded_seq = pad_sequences(sequences, maxlen=600, padding='post', truncating='post')

    def create_matrix(self):
        embedding_index = {}
        with open('glove.6B.100d.txt', encoding='utf-8') as f:
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
        history_model = model.fit(
            self.data_collection['x_train'], self.data_collection['y_train'],
            epochs=10, batch_size=256, validation_data=(self.data_collection['x_test'], self.data_collection['y_test']))
        return history_model




if __name__ == '__main__':
    data_frame = pd.read_csv('train.csv')
    # model_passive_aggressive = DetectFake('passive_aggressive', data_frame)
    # model_passive_aggressive.prepare_data()
    # model_passive_aggressive.vectorization_of_text()
    # model_passive_aggressive.train()
    # model_passive_aggressive.accuracy()
    model_neural_network = DetectFake('neural_network', data_frame)
    model_neural_network.cleaning_news_for_neural_net()
    model_neural_network.vector_text_for_neural_network()
    model_neural_network.create_matrix()
    model_neural_network.init_test_train_split()
    model_neural_network.neural_network()

