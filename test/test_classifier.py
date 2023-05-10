import numpy as np
import pandas as pd
from sklearn import linear_model, datasets, metrics
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras.utils import to_categorical
import logging
import pytest


@pytest.fixture(scope="module")
def iris_data():
    # Load the Iris dataset
    iris = datasets.load_iris()
    X = iris.data
    y = iris.target
    yield X, y

@pytest.fixture(scope="module")
def split_data(iris_data):
    X, y = iris_data
    # Split the dataset into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42)
    yield X_train, X_test, y_train, y_test


@pytest.fixture(scope="module")
def model():
    # Create a neural network model
    model = Sequential()
    model.add(Dense(16, input_shape=(4,)))
    model.add(Activation('sigmoid'))
    model.add(Dense(3))
    model.add(Activation('softmax'))
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy', metrics=["accuracy"])
    yield model

def test_data_shape(iris_data):
    X, y = iris_data
    assert X.shape == (150, 4)
    assert y.shape == (150,)

def test_model_accuracy(model, split_data):
    X_train, X_test, y_train, y_test = split_data
    # Convert target labels to one-hot encoding
    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)
    # Train the model
    model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=0)
    # Evaluate the accuracy on the test set
    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
    assert accuracy >= 0.1


def test_data_leakage(model, iris_data, split_data):
    X_train, X_test, y_train, y_test = split_data
    # Convert target labels to one-hot encoding
    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)
    # Train the model on the entire dataset
    model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=0)
    # Evaluate the accuracy on the training set (should be very high)
    _, train_accuracy = model.evaluate(X_train, y_train, verbose=0)
    # Evaluate the accuracy on the test set (should be lower)
    _, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
    assert test_accuracy < train_accuracy


def test_model_predictions(model, split_data):
    X_train, X_test, y_train, y_test = split_data
    y_pred = model.predict(X_test)
    assert y_pred.shape == (45, 3)
    assert np.allclose(np.sum(y_pred, axis=1), np.ones(45), rtol=1e-5)

def test_model_architecture(model): # test layers number
    assert len(model.layers) >= 1
    assert len(model.layers) <= 10
