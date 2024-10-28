import pandas as pd

def clean_n_transform(train):
    # TODO: Clean and transform the dataset
        
    clean_data_path = "./data/clean_train.csv"
    
    # Save the cleaned training data
    train.to_csv(clean_data_path, index=False)
    
    return train, clean_data_path