import nltk
import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import RegexpTokenizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score


class SpamFilter:
    def __init__(self, data_path):
        self.data_path = data_path
        self.stop_words = set(stopwords.words("english"))
        self.tokenizer = RegexpTokenizer(r'\w+')
        self.stemmer = PorterStemmer()
        self.tfidf_converter = TfidfVectorizer(max_features=2000, min_df=5, max_df=0.7)
        self.text_classifier = RandomForestClassifier(n_estimators=100, random_state=0)

    def preprocess(self, text):
        text = re.sub('[^a-zA-Z]', ' ', text)
        text = text.lower()
        text = self.tokenizer.tokenize(text)
        text = [self.stemmer.stem(word) for word in text if not word in self.stop_words]
        return " ".join(text)

    def train_model(self):
        df = pd.read_csv('/Users/hyunerickang/Desktop/ML Email Spam Filter/spam_ham_dataset.csv')
        df['text'] = df['text'].apply(lambda x: self.preprocess(x))

        features = self.tfidf_converter.fit_transform(df['text']).toarray()

        labels = df['label']
        features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size=0.2,
                                                                                    random_state=0)

        self.text_classifier.fit(features_train, labels_train)

        predictions = self.text_classifier.predict(features_test)

        print(confusion_matrix(labels_test, predictions))
        print(classification_report(labels_test, predictions))
        print(accuracy_score(labels_test, predictions))

    def predict(self, text):
        text = self.preprocess(text)
        features = self.tfidf_converter.transform([text]).toarray()
        return self.text_classifier.predict(features)[0]

spam_filter = SpamFilter('/Users/hyunerickang/Desktop/ML Email Spam Filter/spam_ham_dataset.csv')
spam_filter.train_model()

while True:
    text = input("Enter a message to classify: ")
    if text == "quit":
        print("Program ended.")
        break
    else:
        prediction = spam_filter.predict(text)
        print("This message is likely:", prediction)
