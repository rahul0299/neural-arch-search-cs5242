from CNN_NAS.ChildCNNModel import ChildCNNModel


def test_accuracy_reward(model: ChildCNNModel, test_data, **kwargs):
    test_x, test_y = test_data
    accuracy = model.evaluate_model(test_x, test_y, **kwargs)

    return (accuracy, accuracy) # accuracy, reward value
