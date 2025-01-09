"""
This script tests a logistic regression model on the Iris dataset.
"""
import sys
import os
from ..model import train_model
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def test_train_model():
    """
    Trains a logistic regression model on the Iris dataset.
    Asserts if the accuracy of the model on the test data is < 80%.
    """
    acc = train_model()
    assert acc > 0.8  # Simple test for accuracy threshold
