import json
import plotly
import pandas as pd

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar, Layout
from sklearn.externals import joblib
from sqlalchemy import create_engine


app = Flask(__name__)

def tokenize(text):
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

# load data
engine = create_engine('sqlite:///../data/DisasterResponse.db')
df = pd.read_sql_table('messages', engine)

# load model
model = joblib.load("../models/classifier.pk1")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    # extract data needed for visuals
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)
    
    # Top 10 categories
    category_counts = df.iloc[:, 4:].sum().sort_values(ascending=False).head(10)
    category_names = list(category_counts.index)

    # create visuals
    graphs = [
        # First visualization: Distribution of Message Genres
        {
            'data': [
                Bar(
                    x=genre_names,
                    y=genre_counts,
                    marker=dict(color='rgba(58, 71, 80, 0.6)'),
                    text=['{} messages in {}'.format(y, x) for x, y in zip(genre_names, genre_counts)],
                    hoverinfo='text'
                )
            ],
            'layout': Layout(
                title='Distribution of Message Genres',
                yaxis=dict(title="Count"),
                xaxis=dict(title="Genre")
            )
        },
        # Second visualization: Top 10 Categories
        {
            'data': [
                Bar(
                    x=category_counts,
                    y=category_names,
                    orientation='h',  # horizontal bar chart
                    marker=dict(color='rgba(80, 58, 71, 0.6)'),
                    text=['{} messages in {}'.format(x, y) for x, y in zip(category_counts, category_names)],
                    hoverinfo='text'
                )
            ],
            'layout': Layout(
                title='Top 10 Message Categories',
                xaxis=dict(title="Count"),
                yaxis=dict(title="Category"),
                margin=dict(l=150)  # Adjust left margin for better visibility
            )
        }
    ]
    
    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '') 

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='0.0.0.0', port=3000, debug=True)


if __name__ == '__main__':
    main()
