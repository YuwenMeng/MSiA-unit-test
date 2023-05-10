import numpy as np
import pandas as pd
from sklearn import linear_model, datasets, metrics
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras.utils import to_categorical
import logging

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger("classifier")

def iris_data():
    # Load the Iris dataset
    iris = datasets.load_iris()
    X = iris.data
    y = iris.target
    return X, y


def split_data(X, y, test_size=0.3, random_state=42):
    # Split the dataset into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state)
    return X_train, X_test, y_train, y_test


def create_model(input_shape, num_classes):
    # Create a neural network model
    model = Sequential()
    model.add(Dense(16, input_shape=input_shape))
    model.add(Activation('sigmoid'))
    model.add(Dense(num_classes))
    model.add(Activation('softmax'))
    return model


def train_model(model, X_train, y_train, epochs=10, batch_size=5):
    # Categorical data must be converted to a numerical form
    y_train_ohe = to_categorical(y_train)

    # Compile and train the model
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy', metrics=["accuracy"])
    model.fit(X_train, y_train_ohe, epochs=epochs,
              batch_size=batch_size, verbose=0)


def evaluate_model(model, X_test, y_test):
    # Categorical data must be converted to a numerical form
    y_test_ohe = to_categorical(y_test)

    # Evaluate the model
    loss, accuracy = model.evaluate(X_test, y_test_ohe, verbose=0)
    return accuracy

def main():
    #setup loggin
    try:
        X, y = iris_data()

        X_train, X_test, y_train, y_test = split_data(X, y)
        logger.info('Split data done. ')
        
        model = create_model(input_shape=(
            X.shape[1],), num_classes=len(np.unique(y)))
        train_model(model, X_train, y_train)
        logger.info('Model training done')
        
        accuracy = evaluate_model(model, X_test, y_test)

    except Exception as e:
        logger.error(
            "An exception occurred during model training and evaluation.")
        logger.error(str(e))

if __name__ == "__main__":
    main()
