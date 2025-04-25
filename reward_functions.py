import math
import torch

from CNN_NAS.ChildCNNModel import ChildCNNModel


def test_accuracy_reward(model: ChildCNNModel, test_data, **kwargs):
    test_x, test_y = test_data
    accuracy, validation_loss = model.evaluate_model(test_x, test_y, **kwargs)

    return accuracy, validation_loss, accuracy  # accuracy, reward value



def test_accuracy_with_model_complexity_reward(model, test_data, w1=1, w2=0.05, **kwargs):
    test_x, test_y = test_data
    accuracy, validation_loss = model.evaluate_model(test_x, test_y, **kwargs)

    model_complexity = sum(p.numel() for p in model.parameters())
    reward = w1 * accuracy - w2 * math.log(model_complexity)

    return accuracy, validation_loss, reward
