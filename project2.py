import json
import argparse
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import pandas as pd
from sklearn.model_selection import train_test_split

from sklearn.feature_extraction.text import CountVectorizer
from sklearn import preprocessing
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC
# Function to read the data from a JSON file
def read_data(data):
    df = pd.read_json(data)
    return df
# Function to preprocess the data by tokenizing and removing stop words
def preprocess_data(df, ingredients):
    ing_list = []
    x = []
    # Get a list of English stop words
    stop_words = set(stopwords.words("english"))
    # Loop over the ingredients in the dataset
    for i in df['ingredients']:
        i = " ".join(i)
        x.append(i)
    # Add the input ingredients to the list
    ingredients = " ".join(ingredients)
    x.insert(0, ingredients)
    # Loop over the ingredients, lowercase them, remove digits, and tokenize
    for j in x:
        j = j.lower()
        j = ''.join('' if j.isdigit() else c for c in j)
        token = nltk.word_tokenize(j)
        # Remove stop words from the tokenized ingredients
        my_tokens = " ".join([word for word in token if word not in stop_words])
        ing_list.append(my_tokens)
    
    return ing_list

# Function to vectorize the data using a CountVectorizer
def vectorize_data(data):
    vectorizer = CountVectorizer()
    vector = vectorizer.fit_transform(data)
    headings = vector[0]
    vector = vector[1:]
    return headings, vector
# Function to build the SVM model
def build_model(headings, vector, df):
    label_encoder = preprocessing.LabelEncoder()
    # Fit the label encoder to the cuisine column
    label_encoder.fit(df['cuisine'])
    # Create a pipeline for the SVM model
    clf = make_pipeline(SVC())
    # Split the data into training and testing sets
    x_train, x_test, y_train, y_test = train_test_split(vector, label_encoder.transform(df['cuisine']), test_size=0.3)
    # Fit the model to the training data
    clf.fit(x_train, y_train)
    # Predict the cuisine of the input ingredients
    input_predict = clf.predict(headings)
    cuisine = label_encoder.inverse_transform(input_predict)
    cuisine_dict = {'cuisine': cuisine[0]}
    return cuisine_dict
# Function to find the top N recipes with the highest cosine similarity
def find_top_n(headings, vector, df, N):
    # Calculate the cosine similarity between the input ingredients and the other recipes
    df['Scores'] = cosine_similarity(headings, vector).transpose()
    closest_recipe = df[['id','Scores']].nlargest(int(N)+1, ['Scores'])
    closest_recipes = closest_recipe.to_dict('records')
    score = closest_recipes[0]['Scores']
    return score, closest_recipes[1:]
#Function to print the final results in JSON format
def print_final_output(final_dict):
    json_object = json.dumps(final_dict, indent=4)
    print(json_object)
#Function to write the final results to a JSON file
def write_to_file(final_dict):
    with open('output.json', 'w') as json_file:
        json.dump(final_dict, json_file)
#Main function that runs the script
def main(args):
    if args.N and args.ingredient:
        # Read the data from the JSON file
        df = read_data("yummly.json")
        # Preprocess the data
        data = preprocess_data(df, args.ingredient)
        # Vectorize the data
        headings, vector = vectorize_data(data)
        # Build the SVM model
        cuisine_dict = build_model(headings, vector, df)
        # Find the top N recipes
        score, close_recipes = find_top_n(headings, vector, df, args.N)
        # Print the final results
        final_dict = {**cuisine_dict, 'score': score, 'closest': close_recipes}
        print_final_output(final_dict)
        # Write the final results to a JSON file
        write_to_file(final_dict)
#Parse the command line arguments
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--N', action='store', help="Top N Values", required=True, type=int)
    parser.add_argument('--ingredient', action='append', help="Input Ingredients", required=True)
    args = parser.parse_args()
    main(args)

