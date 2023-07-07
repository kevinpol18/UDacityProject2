import pickle
import re
import sys

import nltk

nltk.download('punkt')
nltk.download('stopwords')
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import word_tokenize
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import accuracy_score, classification_report, f1_score, precision_score, recall_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import Pipeline
from sqlalchemy import create_engine

#set random state
RANDOM_STATE = 42



def load_data(database_filepath):
    
    """Load dataset and split into X and Y
    
    Args:
    database_filename: string. Filename for SQLite database containing cleaned message data.
       
    Returns:
    X: pandas dataframe. Dataframe containing features dataset.
    Y: pandas dataframe. Dataframe containing labels dataset.
    category_names: list. List containing category names.
    """
    # Load data from database
    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql("SELECT * FROM messages", engine)
    
    # Create X and Y datasets
    X = df['message']
    Y = df.drop(['id', 'message', 'original', 'genre'], axis = 1)
    
    # Create list containing all category names
    category_names = list(Y.columns.values)
    
    return X, Y, category_names


def tokenize(text):
    
    """Remove punctuation, lower case, tokenize and stem text string
    
    Args:
    text: string. String containing message for processing
       
    Returns:
    stemmed: list of strings. List containing cleaned and stemmed word tokens
    """
    global stop_words_english

    #Remove punctuation and convert to lower case
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
    
    #Tokenize
    tokens = word_tokenize(text)
    
    #Stem words
    stemmer = PorterStemmer()
    
    #Remove stop words
    stop_words_english = stopwords.words("english")
    
    stemmed = [stemmer.stem(word) for word in tokens if word not in stop_words_english]
    
    return stemmed


def build_model():
    
    """Build a machine learning pipeline to build the model.
    
    Args:
    None
       
    Returns:
    classifier: multi output classifier. Best model from the grid search.
    """
    # Create pipeline
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer = tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])
    
    # Create parameters dictionary
    clf_parameters = {
        'vect__ngram_range': [(1, 1), (1, 2)],
        'tfidf__use_idf': (True, False),
        'clf__estimator__n_estimators': [25, 50, 100],
        'clf__estimator__min_samples_split': [2, 3, 4]
    }
 
    grid_search = GridSearchCV(pipeline, param_grid = clf_parameters, cv = 3, verbose = 3, n_jobs=1, error_score='raise')
    return grid_search

def evaluate_model(model, X_test, Y_test, category_names):
    """Returns test accuracy, precision, recall and F1 score for fitted model
    
    Args:
    model: model object. Fitted model object.
    X_test: dataframe. Dataframe containing test features dataset.
    Y_test: dataframe. Dataframe containing test labels dataset.
    category_names: list of strings. List containing category names.
    
    Returns:
    None
    """
    # Predict labels for test dataset
    Y_pred = model.predict(X_test)
    metrics = []
    
    # Calculate evaluation metrics for each set of labels
    for i in range(len(category_names)):
        accuracy = accuracy_score(Y_test.iloc[:, i], Y_pred[:, i])
        precision = precision_score(Y_test.iloc[:, i], Y_pred[:, i], average = 'weighted')
        recall = recall_score(Y_test.iloc[:, i], Y_pred[:, i], average = 'weighted')
        f1 = f1_score(Y_test.iloc[:, i], Y_pred[:, i], average = 'weighted')
        
        metrics.append([accuracy, precision, recall, f1])
    
    # Create dataframe containing metrics
    metrics = np.array(metrics)
    metrics_df = pd.DataFrame(data = metrics, index = category_names, columns = ['Accuracy', 'Precision', 'Recall', 'F1'])
    
    print(metrics_df)


def save_model(model, model_filepath):
    """Save pickled fitted model
    
    Args:
    model: model object. Fitted model object.
    model_filepath: string. Filepath for where fitted model should be saved
    
    Returns:
    None
    """
    with open(model_filepath, 'wb') as f:
        pickle.dump(model.best_estimator_, f)


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
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