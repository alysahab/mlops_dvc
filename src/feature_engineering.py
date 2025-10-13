from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import os
import logging


logger = logging.getLogger('feature_engineering')
logger.setLevel("DEBUG")

# Handlers
console_handler = logging.StreamHandler()
console_handler.setLevel("DEBUG")

log_dir = 'logs'
os.makedirs(log_dir, exist_ok=True)
file_path = os.path.join(log_dir, 'feature_engineering.log')

file_handler = logging.FileHandler(file_path)
file_handler.setLevel("DEBUG")

# Format for log message
formatter =  logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

# add handler
logger.addHandler(console_handler)
logger.addHandler(console_handler)

def load_data(file_path:str) -> pd.DataFrame:
    """Load processed data"""
    try:
        data = pd.read_csv(file_path)
        data.fillna('', inplace=True)
        logger.debug("Data Loaded and NaNs are filled from %s", file_path)
        return data
    
    except pd.errors.ParserError as e:
        logger.error('Failed to parse the CSV file: %s', e)
        raise
    except Exception as e:
        logger.error('Unexpected error occurred while loading the data: %s', e)
        raise

def apply_tfidf(data: pd.DataFrame, text_column:str, max_feature:int) -> pd.DataFrame:
    """Most repeated words to numericals representation"""
    try:
        tfid = TfidfVectorizer(max_features = max_feature)
        data = pd.concat([data, pd.DataFrame(tfid.fit_transform(data[text_column]).toarray(), columns=tfid.get_feature_names_out())])
        data.drop(columns=text_column, inplace=True)
        logger.debug("Tfidf applied to %s", text_column)
        return data
    except Exception as e:
        logger.error('Error during Bag of Words transformation: %s', e)
        raise


def save_data(df: pd.DataFrame, file_path: str) -> None:
    """Save the dataframe to a CSV file."""
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        df.to_csv(file_path, index=False)
        logger.debug('Data saved to %s', file_path)
    except Exception as e:
        logger.error('Unexpected error occurred while saving the data: %s', e)
        raise


def main(text_column = 'text'):

    try:
        train_data = load_data('data/interim/train_processed.csv')
        test_data = load_data('data/interim/test_processed.csv')

        train_df = apply_tfidf(train_data,text_column,600)
        test_df = apply_tfidf(test_data,text_column,600)

        save_data(train_df, os.path.join('data', 'processed', 'train_tfidf.csv'))
        save_data(test_df, os.path.join('data', 'processed', 'test_tfidf.csv'))

        logger.debug("Data Engineering Completed and Final Dataframe Saved")
    except Exception as e:
        logger.error('Failed to complete the feature engineering process: %s', e)
        print(f"Error: {e}")

if __name__ == '__main__':
    main()