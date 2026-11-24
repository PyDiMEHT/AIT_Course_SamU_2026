from builtins import range
import numpy as np
from random import shuffle

def softmax_loss_naive(W, X, y, reg):
    """
    Softmax loss function, naive implementation (with loops)
    """

    loss = 0.0
    dW = np.zeros_like(W)

    num_train = X.shape[0]
    num_classes = W.shape[1]

    for i in range(num_train):

        # 1. Scores для текущего изображения
        scores = X[i].dot(W)

        # 2. Численная стабилизация
        scores -= np.max(scores)

        # 3. Softmax
        exp_scores = np.exp(scores)
        probs = exp_scores / np.sum(exp_scores)

        # 4. Loss
        loss += -np.log(probs[y[i]])

        # 5. Градиент
        for j in range(num_classes):

            if j == y[i]:
                dW[:, j] += (probs[j] - 1) * X[i]
            else:
                dW[:, j] += probs[j] * X[i]

    # Усредняем по количеству объектов
    loss /= num_train
    dW /= num_train

    # Регуляризация
    loss += reg * np.sum(W * W)
    dW += 2 * reg * W

    return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
    """
    Softmax loss function, vectorized version.
    """

    num_train = X.shape[0]

    # 1. Scores для всех изображений
    scores = X.dot(W)

    # 2. Численная стабилизация
    scores -= np.max(scores, axis=1, keepdims=True)

    # 3. Softmax
    exp_scores = np.exp(scores)

    probs = exp_scores / np.sum(
        exp_scores,
        axis=1,
        keepdims=True
    )

    # 4. Loss
    correct_class_probs = probs[np.arange(num_train), y]

    loss = -np.sum(np.log(correct_class_probs))
    loss /= num_train

    # 5. Градиент
    dscores = probs.copy()

    dscores[np.arange(num_train), y] -= 1

    dW = X.T.dot(dscores)
    dW /= num_train

    # 6. Регуляризация
    loss += reg * np.sum(W * W)
    dW += 2 * reg * W

    return loss, dW