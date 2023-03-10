import sys
import pandas as pd
from sqlalchemy import create_engine



def load_data(messages_filepath, categories_filepath):
    ''' 
    Input:
    messages_filepath - path to messages csv file 
    categories_filepath - path to categories csv file
    
    Output:
    df - dataframe
     '''
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = messages.merge(categories, how='inner', on='id')
    return df


def clean_data(df):
    
    '''
    Input:
    df - dataframe 
    
    Output:
    df - cleaned datadrame 
    
    '''
    
    # creating a df of all the categories by splitting them up.
    categories = df['categories'].str.split(';', expand=True)
    
    # select the first row of the categories dataframe
    row = categories.iloc[0]
    category_colnames = row.apply(lambda x: x[:-2])
    # use this row to extract a list of new column names for categories 
    # one way is to apply a lambda function that takes everything 
    # up to the second to last character of each string with slicing
    
    # rename the columns of `categories`
    categories.columns = category_colnames
    
    for column in categories:
        categories[column] = categories[column].str.slice(-1)  # set each value to be the last character of the string
        categories[column] = categories[column].astype(int) # convert column from string to numeric
        categories_df = pd.DataFrame(data = categories)
    
    #Drop the no longer needed categories column
    df = df.drop('categories', axis = 1)
    
    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df,categories_df], axis = 1)
    
    # drop duplicates
    df = df.drop_duplicates()
    
    return df
    
    
    
    



def save_data(df, database_filename):
    
    '''
    Input:
    df - cleaned dataframe
    database_filename - database filename for sqlite database
    
    Output:
    None
    
   '''
    
    engine = create_engine('sqlite:///'+ database_filename)
    df.to_sql('DisasterResponseTable', engine, index = False, if_exists="replace")
 
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