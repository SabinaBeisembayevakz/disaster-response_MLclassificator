# import libraries
import re
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sqlalchemy import create_engine
import nltk
nltk.download(['punkt', 'wordnet'])
import numpy as np
from nltk.stem import WordNetLemmatizer
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import classification_report
from sklearn.multioutput import MultiOutputClassifier, MultiOutputRegressor
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import GridSearchCV
import pickle
import sys
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
nltk.download('stopwords')


def load_data(database_filepath):
    """
    load data from database
    """
    engine = create_engine(f'sqlite:///{database_filepath}')
    df = pd.read_sql_table('disasterresponse', engine) 
    X = df.message.values
    y = df.iloc[:,5:]
    category_names = y.columns
    return X, y, category_names


def tokenize(text):
    """
    tokenizes, normalizes, lemmatizes text
    """
    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    detected_urls = re.findall(url_regex, text)
    for url in detected_urls:
        text = text.replace(url, "urlplaceholder")

    tokenizer = RegexpTokenizer(r'\w+')
    tokens_np = tokenizer.tokenize(text)
    stop_words = set(stopwords.words('english'))
    filtered_text = []
    for w in tokens_np:
        if w not in stop_words:
            filtered_text.append(w)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for item in filtered_text:
        clean_tok = lemmatizer.lemmatize(item).strip()
        clean_tokens.append(clean_tok)
    return clean_tokens


def build_model():
    """
    builds ML classification model
    """
    clf = RandomForestClassifier()
    pipeline = Pipeline([
            ('vect', CountVectorizer(tokenizer=tokenize)),
            ('tfidf', TfidfTransformer()),
            ('clf', MultiOutputClassifier(clf))
        ])
    model = pipeline
    return model
   

def evaluate_model(model, X_test, Y_test, category_names):
    """
    writes classification report of a model
    """
    y_pred = model.predict(X_test)
    result = classification_report(Y_test, y_pred, target_names=category_names)
    print(result)


def save_model(model, model_filepath):
    """
    saves ML model to a pickle file
    """
    pickle.dump(model, open(f'{model_filepath}', 'wb'))


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print(database_filepath, model_filepath)
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, y, category_names = load_data(database_filepath)
        #print(y)
        X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.2)
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()
Footer
© 2023 GitHub, Inc.
Footer navigation
Terms
Privacy
Security
Status
Docs
Contact GitHub
Pricing
API
Training
Blog
About
