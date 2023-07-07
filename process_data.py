import sys
import pandas as pd
import numpy as np
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    """Load messages and categories datasets and merge on id
    
    Args:
    messages_filepath: string. Filepath for csv file containing messages    dataset.
    categories_filepath: string. Filepath for csv file containing           categories dataset.
       
    Returns:
    df: pandas dataframe. Dataframe containing merged df.
    """
    
    # Load messages dataset
    messages = pd.read_csv(messages_filepath)
    
    # Load categories dataset
    categories = pd.read_csv(categories_filepath)
    
    # Merge datasets
    df = messages.merge(categories, how = 'left', on = ['id'])
    
    return df


def clean_data(df):
    
    """Clean the dataframe by removing duplicates and converting categories from strings to binary values.
    
    Args:
    df: pandas dataframe. Dataframe containing merged df.
       
    Returns:
    df: pandas dataframe. Dataframe containing cleaned version of input     dataframe.
    """
    
    # create a dataframe of the 36 individual category columns
    categories = df['categories'].str.split(';',expand=True)
    
    # select the first row of the categories dataframe
    row = categories.iloc[0]
    
    # use this row to extract a list of new column names for categories.
    category_colnames = [x[0:-2] for x in row]

    # rename the columns of `categories`
    categories.columns = category_colnames
    
    for column in categories:
        
        # set each value to be the last character of the string
        categories[column] = categories[column].transform(lambda x: x[-1:])
    
        # convert column from string to numeric
        categories[column] = pd.to_numeric(categories[column])
        
    # drop the original categories column from `df`
    df = df.drop(columns = ['categories'])
    
    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df, categories], axis = 1)
    
    # drop duplicates
    df = df.drop_duplicates()
    
    # return dataframe
    return df

def save_data(df, database_filename):
    """Save cleaned dataframe into an SQLite database.
    
    Args:
    df: pandas dataframe. Dataframe containing cleaned df.
    database_filename: string. Filename for output database.
       
    Returns:
    None
    """
    engine = create_engine('sqlite:///' + database_filename)
    df.to_sql('messages', engine, index=False, if_exists='replace')  


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()