import project2
import pytest
import pandas as pd
import scipy
import numpy as np

data_file = "yummly.json"
ingredients = ["water", "vegetable oil", "wheat", "salt"]

def test_read_data():
    result = project2.read_data(data_file)
    assert isinstance(result, pd.DataFrame)
    assert not result.empty

def test_preprocess_data():
    df = project2.read_data(data_file)
    result = project2.preprocess_data(df, ingredients)
    assert isinstance(result, list)
    assert len(result) > 0

def test_vectorize_data():
    df = project2.read_data(data_file)
    data = project2.preprocess_data(df, ingredients)
    result = project2.vectorize_data(data)
    assert isinstance(result, tuple)
    assert len(result) == 2
    assert isinstance(result[0], (scipy.sparse.csr.csr_matrix, np.ndarray))
    assert isinstance(result[1], (scipy.sparse.csr.csr_matrix, np.ndarray))


def test_build_model():
    df = project2.read_data(data_file)
    data = project2.preprocess_data(df, ingredients)
    headings, vector = project2.vectorize_data(data)
    result = project2.build_model(headings, vector, df)
    assert isinstance(result, dict)
    assert "cuisine" in result

def test_find_top_n():
    df = project2.read_data(data_file)
    data = project2.preprocess_data(df, ingredients)
    headings, vector = project2.vectorize_data(data)
    result = project2.find_top_n(headings, vector, df, 3)
    assert isinstance(result, tuple)
    assert len(result) == 2
    assert isinstance(result[0], float)
    assert isinstance(result[1], list)
