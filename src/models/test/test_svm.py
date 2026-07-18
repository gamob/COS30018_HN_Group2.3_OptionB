import numpy as np
from sklearn.svm import SVC

from src.models.svm_model import build_svm_model, predict_digit, stratified_sample


def test_stratified_sample_respects_requested_size():
    x = np.arange(100).reshape(20, 5)
    y = np.repeat([0, 1], 10)
    sampled_x, sampled_y = stratified_sample(x, y, 10)
    assert sampled_x.shape == (10, 5)
    assert np.bincount(sampled_y).tolist() == [5, 5]


def test_predict_digit_accepts_uint8_image():
    x = np.vstack([np.zeros(784), np.ones(784)])
    y = np.array([0, 1])
    model = SVC(kernel="linear").fit(x, y)
    assert predict_digit(model, np.zeros((28, 28), dtype=np.uint8)) == 0


def test_build_svm_model_uses_rbf_kernel():
    model = build_svm_model()
    assert model.named_steps["svm"].kernel == "rbf"
