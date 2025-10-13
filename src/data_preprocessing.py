import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
import logging
import os
from sklearn.preprocessing import LabelEncoder
import string
from nltk.stem.porter import PorterStemmer





log_dir = 'logs'
os.makedirs(log_dir, exist_ok=True)

# Logger

logger = logging.getLogger('data_preprocessing')
logger.setLevel("DEBUG")

# Handler
console_handler = logging.StreamHandler()
console_handler.setLevel("DEBUG")

file_path = os.path.join(log_dir, "data_preprocessing.log")
file_handler = logging.FileHandler(file_path)
file_handler.setLevel("DEBUG")

# Format of log message
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)


# Function to transform text
# Lowercase transformation and text preprocessing function

def transform_text(text: str) -> str: 

    # make text to lower
    text = text.lower()
    
    text = nltk.word_tokenize(text)
    y = []

    # Remove special character
    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:]
    y.clear()

    # Loop through the tokens and remove stopwords and puctuation
    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    text = y.copy()
    y.clear()


    # transform each word to each root word using stemming technique
    ps = PorterStemmer()
    for i in text:
        y.append(ps.stem(i))
    
    # join all the word into single string
    return  " ".join(y)




def preprocessing(df: pd.DataFrame, target_column="target", text_column="text") -> pd.DataFrame:
    """Preprocessing of data, drop duplicates, encode target variable, and normalize the text"""
    
    logger.debug("Start Preprocessing for Dataframe")

    try:
        # Drop Duplicates
        df.drop_duplicates(inplace=True)
        logger.debug("Dropped Duplicates")


        # Encode Target Variable
        encoder = LabelEncoder()
        df[target_column] = encoder.fit_transform(df[target_column])
        logger.debug("Target Column Encoded")

        # transform and clean text 
        df[text_column] = df[text_column].apply(transform_text)
        logger.debug("Text column Transformed")
        return df
    
    except KeyError as e:
        logger.error('Column not found: %s', e)
        raise
    except Exception as e:
        logger.error('Error during text normalization %s', e)



def main(text_column='text', target_column='target'):

    try:
        train_data = pd.read_csv("data/raw/train.csv")
        test_data = pd.read_csv("data/raw/test.csv")
        logger.debug("data loaded for preprocessing")

        train_processed_data = preprocessing(train_data, target_column, text_column)
        test_processed_data = preprocessing(test_data, target_column, text_column)
        logger.debug("Data processed successfully")

        data_dir = 'data'
        data_path = os.path.join(data_dir, 'interim')
        os.makedirs(data_path, exist_ok=True)

        train_processed_data.to_csv(os.path.join(data_path, 'train_processed.csv'), index=False)
        test_processed_data.to_csv(os.path.join(data_path, 'test_processed.csv'), index=False)
        logger.debug("Data saved to %s",data_path)
    except FileNotFoundError as e:
        logger.error('File not found: %s', e)
    except pd.errors.EmptyDataError as e:
        logger.error('No data: %s', e)
    except Exception as e:
        logger.error('Failed to complete the data transformation process: %s', e)
        print(f"Error: {e}")

if __name__ == '__main__':
    main()




