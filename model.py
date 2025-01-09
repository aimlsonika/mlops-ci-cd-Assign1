"""
This script trains a logistic regression model on the Iris dataset.
"""
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def train_model():
    """
    Trains a logistic regression model on the Iris dataset.
    Returns the accuracy of the model on the test data.
    """
    data = load_iris() # pylint: disable=no-member
    x_train, x_test, y_train, y_test = train_test_split(
        data.data, data.target, test_size=0.2, random_state=42)
    model = LogisticRegression(max_iter=200)
    model.fit(x_train, y_train)
    predictions = model.predict(x_test)
    acc = accuracy_score(y_test, predictions)
    return acc

if __name__ == "__main__":
    print(f"Model Accuracy: {train_model():.3f}")
