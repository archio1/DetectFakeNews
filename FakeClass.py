import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import cross_val_score


class DetectFake:

    def __init__(self, model_name, data):
        self.model_name = self.check_model_name(model_name)
        self.data = data
        self.model = None
        self.data_collection = None
        self.vectorization_data = None

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
            self.model = self.NT()

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

    def NT(self):
        pass

if __name__ == '__main__':
    data_frame = pd.read_csv('train.csv')
    model_passive_aggressive = DetectFake('passive_aggressive', data_frame)
    model_passive_aggressive.prepare_data()
    model_passive_aggressive.vectorization_of_text()
    model_passive_aggressive.train()
    model_passive_aggressive.accuracy()
