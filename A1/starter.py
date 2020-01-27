from warnings import simplefilter
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# ignore all future warnings
simplefilter(action='ignore', category=FutureWarning)


def loadData():
    with np.load('notMNIST.npz') as data:
        Data, Target = data['images'], data['labels']
        posClass = 2
        negClass = 9
        dataIndx = (Target == posClass) + (Target == negClass)
        Data = Data[dataIndx] / 255.
        Target = Target[dataIndx].reshape(-1, 1)
        Target[Target == posClass] = 1
        Target[Target == negClass] = 0
        np.random.seed(421)
        randIndx = np.arange(len(Data))
        np.random.shuffle(randIndx)
        Data, Target = Data[randIndx], Target[randIndx]
        trainData, trainTarget = Data[:3500], Target[:3500]
        validData, validTarget = Data[3500:3600], Target[3500:3600]
        testData, testTarget = Data[3600:], Target[3600:]
    return trainData, validData, testData, trainTarget, validTarget, testTarget


# reg = lambda
# I'm assuming x is the matrix version
# I agree with this interpretation, but I changed input W to w since it's a vector and not matrix -Dev
# The dimensions of w are unclear in the assignment, but I think w is a column vector, x is an n by d matrix. lmk if you
#   think otherwise.
def MSE(w, b, x, y, reg):
    return np.linalg.norm(x.dot(w) + b - y) ** 2 + reg / 2 * np.linalg.norm(w) ** 2


# Je suis confused. The document says 'return the gradient with respect to the weights and the gradient with respect to
#   the bias. Which one do we return? Or, are we supposed to return both in a list or something?
def gradMSE(w, b, x, y, reg):
    return 2 * x.T.dot(x.dot(w) + b - y) + reg * w, 2 * x.T.dot(x.dot(w) + b - y) * 1


# TODO: This function is untested!!
def grad_descent(w, b, x, y, alpha, epochs, reg, error_tol, val_data, val_label, test_data, test_label, printing=True):
    prev_loss = 0
    for i in range(epochs):
        grad_w, grad_b = gradMSE(w, b, x, y, reg)
        w -= alpha * grad_w
        b -= alpha * grad_b
        loss = MSE(w, b, x, y, reg)

        # Print Losses if printing is on
        if printing:
            print("Training loss is", loss)
            print("Validation loss is ", MSE(w, b, val_data, val_label, reg))

        # Check stopping condition
        if np.abs(prev_loss - loss) <= error_tol:
            break
        else:
            prev_loss = loss
    print("Test loss is ", MSE(w, b, test_data, test_label, reg))
    return w, b


def crossEntropyLoss(W, b, x, y, reg):
    # Your implementation here
    pass


def gradCE(W, b, x, y, reg):
    # Your implementation here
    pass


def buildGraph(loss="MSE"):
    # Initialize weight and bias tensors
    tf.set_random_seed(421)

    if loss == "MSE":
        # Your implementation
        pass

    elif loss == "CE":
        # Your implementation here
        pass


if __name__ == "__main__":
    trainData, validData, testData, trainTarget, validTarget, testTarget = loadData()
    print("hi")
