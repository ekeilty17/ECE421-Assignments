# ignore all future warnings
from warnings import simplefilter
simplefilter(action='ignore', category=FutureWarning)

# imports
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import layers, models, datasets

# ignore tensorflow depreciation warnings
import tensorflow.python.util.deprecation as deprecation
deprecation._PRINT_DEPRECATION_WARNINGS = False

"""
    Bunch of Helper functions
"""

# given by the assignment
def loadData():
    with np.load("notMNIST.npz") as data:
        Data, Target = data ["images"], data["labels"]
        np.random.seed(521)
        randIndx = np.arange(len(Data))
        np.random.shuffle(randIndx)
        Data = Data[randIndx]/255.
        Target = Target[randIndx]
        trainData, trainTarget = Data[:15000], Target[:15000]
        validData, validTarget = Data[15000:16000], Target[15000:16000]
        testData, testTarget = Data[16000:], Target[16000:]
    return trainData, validData, testData, trainTarget, validTarget, testTarget
trainData, validData, testData, trainTarget, validTarget, testTarget = loadData()

# given by the assignment
def convertOneHot(trainTarget, validTarget, testTarget):
    newtrain = np.zeros((trainTarget.shape[0], 10))
    newvalid = np.zeros((validTarget.shape[0], 10))
    newtest = np.zeros((testTarget.shape[0], 10))
    for item in range(0, trainTarget.shape[0]):
        newtrain[item][trainTarget[item]] = 1
    for item in range(0, validTarget.shape[0]):
        newvalid[item][validTarget[item]] = 1
    for item in range(0, testTarget.shape[0]):
        newtest[item][testTarget[item]] = 1
    return newtrain, newvalid, newtest

def plot_loss(x, train_loss=None, valid_loss=None, test_loss=None, title=None, ax=None):
    ax = plt.gca() if ax == None else ax
    if train_loss != None:
        ax.plot(x, train_loss, label="Training Loss")
    if valid_loss != None:
        ax.plot(x, valid_loss, label="Validation Loss")
    if test_loss != None:
        ax.plot(x, test_loss, label="Testing Loss")

    ax.set_title("Loss" if title == None else title)

    ax.set_xlabel("Iterations")
    ax.set_xlim(left=0)
    ax.set_ylabel("Loss")
    ax.set_ylim(bottom=0)
    ax.legend(loc="upper right")

def plot_accuracy(x, train_accuracy=None, valid_accuracy=None, test_accuracy=None, title=None, ax=None):
    ax = plt.gca() if ax == None else ax
    if train_accuracy != None:
        ax.plot(x, train_accuracy, label="Training Accuracy")
    if valid_accuracy != None:
        ax.plot(x, valid_accuracy, label="Validation Accuracy")
    if test_accuracy != None:
        ax.plot(x, test_accuracy, label="Testing Accuracy")

    ax.set_title("Accuracy" if title == None else title)

    ax.set_xlabel("Iterations")
    ax.set_xlim(left=0)
    ax.set_ylabel("Accuracy")
    ax.set_yticks(np.arange(0, 1.1, step=0.1))
    ax.grid(linestyle='-', axis='y')
    ax.legend(loc="lower right")


def display_statistics(train_loss=None, train_acc=None, valid_loss=None, valid_acc=None, 
                       test_loss=None, test_acc=None):
    
    tl == "-" if train_loss is None else round(train_loss[-1], 4)
    ta == "-" if train_acc is None else round(train_acc[-1]*100, 2)
    vl == "-" if valid_loss is None else round(valid_loss[-1], 4)
    va == "-" if valid_acc is None else round(valid_acc[-1]*100, 2)
    sl == "-" if test_loss is None else round(test_loss[-1], 4)
    sa == "-" if test_acc is None else round(test_acc[-1]*100, 2)
    
    print(f"Training loss: {tl}{'':.20s}\t\tTraining acc: {ta}%")
    print(f"Validation loss: {vl}{'':.20s}\tValidation acc: {va}%")
    print(f"Testing loss: {sl}{'':.20s}\tTesting acc: {sa}%")
    
    fig, ax = plt.subplots(1, 2, figsize=(18, 6))
    plot_loss(np.arange(0, len(train_loss), 1), train_loss, valid_loss, test_loss, ax=ax[0])
    plot_accuracy(np.arange(0, len(train_loss), 1), train_acc, valid_acc, test_acc, ax=ax[1])
    plt.show()
    plt.close()

TINY = 1e-20
VTDatasets = {"validData" : validData, "validTarget" : validTarget,
              "testData" : testData, "testTarget" : testTarget}

N = trainData.shape[0]
d = trainData.shape[1] * trainData.shape[2]
K = 10

"""
    Neural Network Helper Functions
"""
def relu(x):
    return np.maximum(0, x)

def softmax(x):
    return np.exp(x) / np.exp(x).sum()

def softmax_batch(X):
    return np.exp(X) / np.exp(X).sum(axis=1, keepdims=True)

def computeLayer(X, W, b):
    # W might need to get transposed depending on how we define it
    return X.dot(W) + b

# target is one-hot encoded
# prediction is the output of the softmax function
def CE(target, prediction):
    return -(target * np.log(prediction+TINY)).sum(axis=1).mean()

# target is one-hot encoded
# activation is the value before softmaxing
def gradCE(target, activation):
    return target - activation


"""
    BackProp
"""
# Making a fake neural network
class NN(object):

    """
    3 layers:
        input: x
        hidden: h
        output: o

    hidden has ReLU activation function
    output is softmaxed
    """

    def __init__(self, inp_size, hid_size, out_size):
        self.inp_size = inp_size
        self.hid_size = hid_size
        self.out_size = out_size

        # getting random parameters
        self.W_h = 2*np.random.random_sample((inp_size, hid_size)) - 1
        self.b_h = 2*np.random.random_sample(hid_size) - 1
        self.W_o = 2*np.random.random_sample((hid_size, out_size)) - 1
        self.b_o = 2*np.random.random_sample(out_size) - 1

    def Loss(self, y, y_pred):
        return CE(y, y_pred)

    def dLoss(self, y, y_pred):
        return gradCE(y, y_pred)

    def feedforward(self, X):
        H = computeLayer(X, self.W_h, self.b_h)
        H = relu(H)
        O = computeLayer(H, self.W_o, self.b_o)
        P = softmax_batch(O)

        # do it for each batch
        #P = O.copy()
        #for i in range(P.shape[0]):
        #    P[i, :] = softmax(O[i, :])
        return P
    
    def backprop(self, X, y):
        
        G = computeLayer(X, self.W_h, self.b_h)
        H = relu(G)
        O = computeLayer(H, self.W_o, self.b_o)
        P = softmax_batch(O)


        dL_do = self.dLoss(y, P)

        dL_dWo = H.T @ dL_do
        dL_dbo = dL_do

        dL_dh = dL_do @ self.W_o.T
        dL_dg = (dL_dh > 0) * 1

        dL_dWh = X.T @ dL_dg
        dL_dbh = dL_dg

        return dL_dWo, dL_dbo, dL_dWh, dL_dbh

trainTarget, validTarget, testTarget = convertOneHot(trainTarget, validTarget, testTarget)

F = 100     # F = features
model = NN(d, F, K)
X = trainData.reshape(N, d)
y = trainTarget

dL_dWo, dL_dbo, dL_dWh, dL_dbh = model.backprop(X, y)
print(dL_dWo, dL_dbo, dL_dWh, dL_dbh)
