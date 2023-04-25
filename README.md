# cs5293sp23-project2
# Cuisine Predictor
*Vamsi Thokala*

This code provides a recipe recommendation system based on the ingredients provided by the user. The system uses a Support Vector Machine (SVM) model to predict the cuisine of the input ingredients and returns the top N closest matching recipes based on cosine similarity.
Libraries Imported

    json: to parse and manipulate JSON data
    argparse: to parse command-line arguments
    nltk: Natural Language Processing (NLP) library to perform text preprocessing
    pandas: to read and manipulate data stored in a JSON file
    sklearn: machine learning library to build the SVM model and perform data preprocessing and feature extraction

Functions
read_data(data)

    Input: path of the JSON file containing the recipe data
    Output: a pandas DataFrame containing the recipe data

preprocess_data(df, ingredients)

    Input:
        df: pandas DataFrame containing the recipe data
        ingredients: list of input ingredients
    Output: a list of preprocessed ingredients, where each ingredient is a string of tokens obtained after tokenizing and removing stop words from the ingredient

vectorize_data(data)

    Input: list of preprocessed ingredients
    Output: headings and vectorized data, where headings are the vectorized form of the input ingredients and vector is the vectorized form of the rest of the ingredients in the dataset

build_model(headings, vector, df)

    Input:
        headings: vectorized form of the input ingredients
        vector: vectorized form of the rest of the ingredients in the dataset
        df: pandas DataFrame containing the recipe data
    Output: a dictionary with the key 'cuisine' and the value being the cuisine predicted for the input ingredients using the SVM model

find_top_n(headings, vector, df, N)

    Input:
        headings: vectorized form of the input ingredients
        vector: vectorized form of the rest of the ingredients in the dataset
        df: pandas DataFrame containing the recipe data
        N: number of closest matching recipes to be returned
    Output: a tuple containing the cosine similarity score of the input ingredients with the closest matching recipe and a list of dictionaries with the keys 'id' and 'Scores' representing the id of the closest matching recipes and their cosine similarity scores with the input ingredients, respectively.

print_final_output(final_dict)

    Input: a dictionary containing the predicted cuisine and top N closest matching recipes
    Output: None. Prints the final dictionary in a formatted JSON string

write_to_file(final_dict)

    Input: a dictionary containing the predicted cuisine and top N closest matching recipes
    Output: None. Writes the final dictionary to a JSON file 'output.json'

main(args)

    Input: command-line arguments in the form of an argparse Namespace object
    Output: None. Calls the necessary functions to run the script and produce the final output in a JSON string and a JSON file

Command Line Arguments

    --N: the number of closest matching recipes to be returned
    --ingredient: list of input ingredients

How to Run the Code

The code can be run using the following command in the terminal:

    python recipe_recommendation_system.py --N <value> --ingredient <ingredient1> --ingredient <ingredient2> ...

where <value> is the desired number of closest matching recipes to be returned, and <ingredient1>, <ingredient2>, ... are the input ingredients.

For example, to return the top 3 closest matching recipes for ingredients 'garlic', 'onion', and 'potatoes', the command would be:
python recipe_recommendation_system.py --N 3 --ingredient garlic --ingredient onion --ingredient potatoes



https://user-images.githubusercontent.com/115323632/234424140-a2ed1f7f-29e0-40a9-97a6-1a305e1f301e.mov


