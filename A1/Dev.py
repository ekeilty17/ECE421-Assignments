# ignore all future warnings
from warnings import simplefilter
simplefilter(action='ignore', category=FutureWarning)



# imports
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt



# given by the assignment
def loadData():
    with np.load('notMNIST.npz') as data :
        Data, Target = data['images'], data['labels']
        posClass = 2
        negClass = 9
        dataIndx = (Target==posClass) + (Target==negClass)
        Data = Data[dataIndx]/255.
        Target = Target[dataIndx].reshape(-1, 1)
        Target[Target==posClass] = 1
        Target[Target==negClass] = 0
        np.random.seed(421)
        randIndx = np.arange(len(Data))
        np.random.shuffle(randIndx)
        Data, Target = Data[randIndx], Target[randIndx]
        trainData, trainTarget = Data[:3500], Target[:3500]
        validData, validTarget = Data[3500:3600], Target[3500:3600]
        testData, testTarget = Data[3600:], Target[3600:]
    return trainData, validData, testData, trainTarget, validTarget, testTarget



trainData, validData, testData, trainTarget, validTarget, testTarget = loadData()
print(f"Training Data: {trainData.shape}\tTraining tagets: {trainTarget.shape}")
print(f"Validation Data: {validData.shape}\tValidation tagets: {validTarget.shape}")
print(f"Testing Data: {testData.shape}\tTesting tagets:{testTarget.shape}")



def plot(image, target):
    plt.imshow(image, cmap="hot")
    plt.title('J' if target == 0 else 'C')
    # targets are binary encoded 0 == 'J' and 1 == 'C'
    plt.show()



plot(trainData[0], trainTarget[0])
plot(trainData[1], trainTarget[1])


def augment(X, w, b):
    # flatten X
    if len(X.shape) == 3:
        X = X.reshape(X.shape[0], -1)

    # insert 1's at position 0 along the columns
    X = np.insert(X, 0, 1, axis=1)

    # insert b at the front of W
    w = np.insert(w, 0, b, axis=0)

    return X, w



def predict(w, b, X):
    X = X.reshape(X.shape[0], -1)
    return X.dot(w) + b



def accuracy(w, b, X, y):
    y = y.reshape(-1)
    y_pred = predict(w, b, X)
    y_pred = np.vectorize(lambda z: 1 if z > 0 else 0)(y_pred)
    return sum(y_pred == y) / y.shape[0]


# Mean Squared Error Loss
def MSE(w, b, X, y, reg):
    X = X.reshape(X.shape[0], -1)
    y = y.reshape(-1)
    return np.square(X.dot(w) + b - y).mean() + reg * np.square(w).sum()


def gradMSE(w, b, X, y, reg):
    X = X.reshape(X.shape[0], -1)
    y = y.reshape(-1)
    N = y.shape[0]

    w_grad = 2.0 / N * X.T.dot(X.dot(w) + b - y) + reg * w
    b_grad = 2.0 / N * np.sum(X.dot(w) + b - y)
    return w_grad, b_grad


# The below is a test for MSE Loss, which is correct (at least without the regulator)
"""
from sklearn.metrics import mean_squared_error

X = trainData
y = trainTarget
N = X.shape[0]
d = X.shape[1] * X.shape[2]

w = np.random.random_sample(d)
b = np.random.random_sample(1)

y_pred = predict(w_LS, b_LS, X)
print(mean_squared_error(y, y_pred))
print(MSE(w_LS, b_LS, X, y, 0))
"""

# gradMSE(w, b, X, y, 0.1)


def grad_descent_MSE(w, b, X, y, alpha, epochs, reg, error_tol, validData=None, validTarget=None, testData=None,
                     testTarget=None):
    train_loss, train_acc = [], []
    valid_loss, valid_acc = [], []
    test_loss, test_acc = [], []
    printing = True
    for i in range(epochs):
        grad_w, grad_b = gradMSE(w, b, X, y, reg)
        w -= alpha * grad_w
        b -= alpha * grad_b

        # Calculating Statistics
        train_loss.append(MSE(w, b, X, y, reg))
        train_acc.append(accuracy(w, b, X, y))

        if validData != None and validTarget != None:
            valid_loss.append(MSE(w, b, validData, validTarget, reg))
            valid_acc.append(accuracy(w, b, validData, validTarget))
        if testData != None and testTarget != None:
            test_loss.append(MSE(w, b, testData, testTarget, reg))
            valid_acc.append(accuracy(w, b, testData, testTarget))

        # Print Losses and Accurancies if printing is on
        if printing:
            print(f"Training loss: {train_loss[-1]:.4f}\tTraining acc: {train_acc[-1] * 100:.2f}%")
            if validData != None and validTarget != None:
                print(f"Validation loss: {valid_loss[-1]:.4f}\tValidation acc: {valid_acc[-1] * 100:.2f}%")
            if testData != None and testTarget != None:
                print(f"Testing loss: {test_loss[-1]:.4f}\tTesting acc: {test_acc[-1] * 100:.2f}%")

        # Check stopping condition
        if i > 1 and np.abs(train_loss[-2] - train_loss[-1]) <= error_tol:
            break

    statistics = (train_loss, train_acc)
    if validData != None and validTarget != None:
        statistics += (valid_loss, valid_acc,)
    if testData != None and testTarget != None:
        statistics += (test_loss, test_acc,)
    # Python 3.8 made this easier, but 3.7 you have to do this
    out = (w, b, *statistics)

    return out


X = trainData
N = X.shape[0]
d = X.shape[1] * X.shape[2]

w = np.random.random_sample(d)
b = np.random.random_sample(1)
w, b, *statistics = grad_descent_MSE(w, b, trainData, trainTarget, 0.005, 5000, 0, 0.01)
# train_loss, train_acc, valid_loss, valid_acc, test_loss, test_acc = statistics
train_loss, train_acc = statistics


# functions to plot loss and accuracy
def plot_loss(x, train_loss=None, valid_loss=None, test_loss=None, title=None):
    if train_loss != None:
        plt.plot(x, train_loss, label="Training Loss")
    if valid_loss != None:
        plt.plot(x, valid_loss, label="Validation Loss")
    if test_loss != None:
        plt.plot(x, test_loss, label="Testing Loss")

    if title == None:
        plt.title("Training Loss")
    else:
        plt.title(title)

    plt.xlabel("Epochs")
    plt.xlim(left=0)
    plt.ylabel("Loss")
    plt.legend(loc="upper right")
    plt.show()


def plot_accuracy(x, train_accuracy=None, valid_accuracy=None, test_accuracy=None, title=None):
    if train_accuracy != None:
        plt.plot(x, train_accuracy, label="Training Accuracy")
    if valid_accuracy != None:
        plt.plot(x, valid_accuracy, label="Validation Accuracy")
    if test_accuracy != None:
        plt.plot(x, test_accuracy, label="Testing Accuracy")

    if title == None:
        plt.title("Accuracy")
    else:
        plt.title(title)

    plt.xlabel("Epochs")
    plt.xlim(left=0)
    plt.ylabel("Accuracy")
    plt.yticks(np.arange(0, 1.1, step=0.1))
    plt.grid(linestyle='-', axis='y')
    plt.legend(loc="lower right")
    plt.show()


"""
plot_loss(np.arange(0, len(train_loss), 1), train_loss)#, valid_loss, test_loss)
plot_accuracy(np.arange(0, len(train_loss), 1), train_acc)#, valid_acc, test_acc)
"""



# Test your implementation of Gradient Descent with 5000 epochs and \lambda = 0. Investigate the
# impact of learning rate, \alpha = 0.005, 0.001, 0.0001 on the performance of your classifier.
# Plot the training, validation and test losses.

# Eric



# Investigate impact by modifying the regularization parameter, \lambda = {0.001, 0.1, 0.5}.
# Plot the training, validation and test loss for \alpha = 0:005 and report the final training,
# validation and test accuracy of your classifier.

# Sandra



def least_squares(X, y):
    N = X.shape[0]
    d = X.shape[1] * X.shape[2]
    X, _ = augment(X, np.zeros(X.shape[0]), 0)
    y = y.reshape(-1)
    if N < d:
        w_aug = X.T.dot(np.linalg.inv(np.dot(X, X.T))).dot(y)
    else:
        w_aug = np.linalg.inv(np.dot(X.T, X)).dot(X.T).dot(y)

    return w_aug[1:], w_aug[0]


# compare above to gradient descent solution
w_LS, b_LS = least_squares(trainData, trainTarget)

loss = MSE(w_LS, b_LS, trainData, trainTarget, 0)
acc = accuracy(w_LS, b_LS, trainData, trainTarget)
print(f"Least Squares Training loss: {loss:.4f}\tLeast Squares Training acc: {acc*100:.2f}%")
loss = MSE(w_LS, b_LS, validData, validTarget, 0)
acc = accuracy(w_LS, b_LS, validData, validTarget)
print(f"Least Squares Validation loss: {loss:.4f}\tLeast Squares Validation acc: {acc*100:.2f}%")
loss = MSE(w_LS, b_LS, testData, testTarget, 0)
acc = accuracy(w_LS, b_LS, testData, testTarget)
print(f"Least Squares Testing loss: {loss:.4f}\tLeast Squares Testing acc: {acc*100:.2f}%")



# this will work for both scalar and vector z
def sigmoid(z):
    return 1.0 / (1 + np.exp(-z))


# Cross Entropy Loss
def crossEntropyLoss(w, b, X, y, reg):
    X, w = augment(X, w, b)
    y = y.reshape(-1)
    N = y.shape[0]

    #
    # LOGARITHMS DONE IN BASE E for now
    #

    y_hat = sigmoid(X.dot(w))
    return (-y.dot(np.log(y_hat)) - (1 - y).dot(np.log(1 - y_hat))) / N + reg / 2.0 * np.square(w).sum()


def gradCE(w, b, X, y, reg):
    X, w = augment(X, w, b)
    y = y.reshape(-1)
    N = y.shape[0]

    y_hat = sigmoid(X.dot(w))
    sigmoid_derivative = np.multiply(y_hat, 1 - y_hat)
    w_grad = 1.0 / N * X.T.dot(np.multiply(-1 * np.multiply(y, 1 / y_hat) + np.multiply(1 - y, 1.0 / (1 - y_hat)),
                                           sigmoid_derivative)) + reg * w
    return w_grad[1:], w_grad[0] - reg * w[0]

# Dev


def grad_descent(w, b, X, y, alpha, epochs, reg, error_tol, lossType="MSE", validData=None, validTarget=None,
                 testData=None, testTarget=None):
    if lossType == "MSE":
        return grad_descent_MSE(w, b, X, y, alpha, epochs, reg, error_tol)
    elif lossType == "CE":
        train_loss, train_acc = [], []
        valid_loss, valid_acc = [], []
        test_loss, test_acc = [], []
        printing = True
        for i in range(epochs):
            grad_w, grad_b = gradCE(w, b, X, y, reg)
            w -= alpha * grad_w
            b -= alpha * grad_b

            # Calculating Statistics
            train_loss.append(crossEntropyLoss(w, b, X, y, reg))
            train_acc.append(accuracy(w, b, X, y))

            if validData is not None and validTarget is not None:
                valid_loss.append(crossEntropyLoss(w, b, validData, validTarget, reg))
                valid_acc.append(accuracy(w, b, validData, validTarget))
            if testData is not None and testTarget is not None:
                test_loss.append(crossEntropyLoss(w, b, testData, testTarget, reg))
                valid_acc.append(accuracy(w, b, testData, testTarget))

            # Print Losses and Accurancies if printing is on
            if printing:
                print(f"Training loss: {train_loss[-1]:.4f}\tTraining acc: {train_acc[-1] * 100:.2f}%")
                if validData is not None and validTarget is not None:
                    print(f"Validation loss: {valid_loss[-1]:.4f}\tValidation acc: {valid_acc[-1] * 100:.2f}%")
                if testData is not None and testTarget is not None:
                    print(f"Testing loss: {test_loss[-1]:.4f}\tTesting acc: {test_acc[-1] * 100:.2f}%")

            # Check stopping condition
            if i > 1 and np.abs(train_loss[-2] - train_loss[-1]) <= error_tol:
                break

        statistics = (train_loss, train_acc)
        if validData is not None and validTarget is not None:
            statistics += (valid_loss, valid_acc,)
        if testData is not None and testTarget is not None:
            statistics += (test_loss, test_acc,)
        out = (w, b, *statistics)

        return out

    else:
        raise ValueError("Variable 'lossType' must be either 'MSE' or 'CE'.")



# For zero weight decay, learning rate of 0.005 and 5000 epochs,
# plot the training cross entropy loss and MSE loss for
# logistic regression and linear regression respectively.
# Comment on the effect of cross-entropy loss convergence behaviour.

# Sandra


def buildGraph(beta1=None, beta2=None, epsilon=None, alpha=0.001, loss="MSE", d=784):
    # Initialize weight and bias tensors
    seed = 421
    tf.set_random_seed(seed)

    # Getting small random initial values for weights and bias
    w = tf.truncated_normal([d], stddev=0.5)
    b = tf.truncated_normal([1], stddev=0.5)

    # Converting into tensorflow Variable objects
    w = tf.Variable(w, name="weights")
    b = tf.Variable(b, name="bias")

    # tensorflow objects for data
    trainData, _, _, trainTarget, _, _ = loadData()
    trainData = tf.placeholder(tf.float32, (None, trainData.shape[1] * trainData.shape[2]))
    trainTarget = tf.placeholder(tf.int8, (None, trainTarget.shape[1]))
    reg = tf.Variable(tf.ones(1), name="reg")

    if loss == "MSE":
        y_pred = tf.matmul(trainData, w) + b
        loss_tensor = tf.losses.mean_squared_error(trainTarget, y_pred)
    elif loss == "CE":
        y_pred = tf.sigmoid(tf.matmul(trainData, w) + tf.convert_to_tensor(b))
        loss_tensor = tf.losses.sigmoid_cross_entropy(trainTarget, y_pred)
    else:
        raise ValueError("Variable 'lossType' must be either 'MSE' or 'CE'.")

    opt = tf.train.AdamOptimizer(learning_rate=alpha, beta1=beta1, beta2=beta2, epsilon=epsilon)
    opt_op = opt.minimize(loss_tensor)

    return w, b, testData, y_pred, testTarget, loss_tensor, opt_op, reg
