import re
import pandas as pd
from sklearn.model_selection import train_test_split

def preprocess_text(text):
    text = text.lower()  # Convert to lowercase
    text = re.sub(r'http\S+', '', text)  # Remove URLs
    text = re.sub(r'[^a-z0-9\s]', '', text)  # Remove special characters
    text = re.sub(r'\s+', ' ', text).strip()  # Remove extra spaces
    return text

def normalize_class(class_name):
    class_name = class_name.lower()
    if class_name == 'pi':
        return 'yes'
    return class_name

def process_data(file_path, updated_file_path, output_train_path, output_val_path):
    # Load the data
    data = pd.read_csv(file_path)

    # Preprocessing
    # 1. Remove unnecessary columns
    data = data.drop(columns=['Unnamed: 2'])

    # 2. Handle missing values
    data = data.dropna(subset=['text'])

    # 3. Standardize text (lowercase, remove special characters)
    data['text'] = data['text'].apply(preprocess_text)

    # Normalizing the class names
    data['class'] = data['class'].apply(normalize_class)

    undefined_df = data[data['class'] == "undefined"]
    undefined_df.to_csv('undefined_df.csv', index=False)

    updated_undefined_df = pd.read_csv(updated_file_path)

    pred_no = updated_undefined_df[updated_undefined_df["class"] == "no"]
    pred_yes = updated_undefined_df[updated_undefined_df["class"] == "yes"]

    # Combine dataframes vertically
    combined_df = pd.concat([data[data["class"] != "undefined"], pred_no[0:337], pred_yes], axis=0)

    combined_df.rename(columns={'class': 'label'}, inplace=True)
    combined_df['label'] = combined_df['label'].apply(lambda x: 1 if x =='yes' else 0)

    dataset = combined_df[['label', 'text']]

    # Perform stratified split
    X = dataset.drop(columns=['label'])
    y = dataset['label']

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.25, stratify=y, random_state=42)

    # Combine X and y back into DataFrames for training and validation sets
    train_roberta = X_train.copy()
    train_roberta['label'] = y_train

    validation_roberta = X_val.copy()
    validation_roberta['label'] = y_val

    # Save the datasets
    train_roberta.to_csv(output_train_path, index=False)
    validation_roberta.to_csv(output_val_path, index=False)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--file_path', type=str, required=True)
    parser.add_argument('--updated_file_path', type=str, required=True)
    parser.add_argument('--output_train_path', type=str, required=True)
    parser.add_argument('--output_val_path', type=str, required=True)
    args = parser.parse_args()

    process_data(args.file_path, args.updated_file_path, args.output_train_path, args.output_val_path)
