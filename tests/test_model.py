from model import train_model

def test_train_model():
    acc = train_model()
    assert acc > 0.5  # Simple test for accuracy threshold