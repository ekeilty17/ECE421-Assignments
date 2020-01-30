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
"""
print(f"Training Data: {trainData.shape}\tTraining tagets: {trainTarget.shape}")
print(f"Validation Data: {validData.shape}\tValidation tagets: {validTarget.shape}")
print(f"Testing Data: {testData.shape}\tTesting tagets:{testTarget.shape}")
"""

"""
        Helpful Functions
"""
def plot(image, target):
    plt.imshow(image, cmap="hot")
    plt.title('J' if target == 0 else 'C')
    # targets are binary encoded 0 == 'J' and 1 == 'C'
    plt.show()

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

# functions to plot loss and accuracy
def plot_loss(x, train_loss=None, valid_loss=None, test_loss=None, title=None, ax=None):
    ax = plt if ax == None else ax
    if train_loss != None:
        ax.plot(x, train_loss, label="Training Loss")
    if valid_loss != None:
        ax.plot(x, valid_loss, label="Validation Loss")
    if test_loss != None:
        ax.plot(x, test_loss, label="Testing Loss")
    
    if title == None:
        ax.set_title("Loss")
    else:
        ax.set_title(title)
    
    ax.set_xlabel("Epochs")
    ax.set_xlim(left=0)
    ax.set_ylabel("Loss")
    ax.legend(loc="upper right")

def plot_accuracy(x, train_accuracy=None, valid_accuracy=None, test_accuracy=None, title=None, ax=None):
    ax = plt if ax == None else ax
    if train_accuracy != None:
        ax.plot(x, train_accuracy, label="Training Accuracy")
    if valid_accuracy != None:
        ax.plot(x, valid_accuracy, label="Validation Accuracy")
    if test_accuracy != None:
        ax.plot(x, test_accuracy, label="Testing Accuracy")
    
    if title == None:
        ax.set_title("Accuracy")
    else:
        ax.set_title(title)

    ax.set_xlabel("Epochs")
    ax.set_xlim(left=0)
    ax.set_ylabel("Accuracy")
    ax.set_yticks(np.arange(0, 1.1, step=0.1))
    ax.grid(linestyle='-', axis='y')
    ax.legend(loc="lower right")


"""
        MSE Loss
"""
def MSE(w, b, X, y, reg=0):
    X = X.reshape(X.shape[0], -1)
    y = y.reshape(-1)
    return np.square(X.dot(w) + b - y).mean() + reg * np.square(w).sum()

def gradMSE(w, b, X, y, reg=0):
    X = X.reshape(X.shape[0], -1)
    y = y.reshape(-1)
    N = y.shape[0]
    
    w_grad = 2.0/N * X.T.dot(X.dot(w) + b - y) + reg * w
    b_grad = 2.0/N * np.sum(X.dot(w) + b - y)
    return w_grad, b_grad


"""
        GD with MSE
"""
def grad_descent_MSE(w, b, X, y, alpha, epochs, reg, error_tol, validData=None, validTarget=None, testData=None, testTarget=None):
    train_loss, train_acc = [], []
    valid_loss, valid_acc = [], []
    test_loss, test_acc = [], []
    printing = True
    for e in range(epochs):
        grad_w, grad_b = gradMSE(w, b, X, y, reg)
        w -= alpha * grad_w
        b -= alpha * grad_b
        
        # Calculating Statistics
        train_loss.append( MSE(w, b, X, y, reg) )
        train_acc.append( accuracy(w, b, X, y) )

        if not validData is None and not validTarget is None:
            valid_loss.append( MSE(w, b, validData, validTarget, reg) )
            valid_acc.append( accuracy(w, b, validData, validTarget) )
        if not testData is None and not testTarget is None:
            test_loss.append( MSE(w, b, testData, testTarget, reg) )
            test_acc.append( accuracy(w, b, testData, testTarget) )
        
        # Print Losses and Accurancies if printing is on
        if printing:
            print(f"Training loss: {train_loss[-1]:.4f}\tTraining acc: {train_acc[-1]*100:.2f}%")
            if not validData is None and not validTarget is None:
                print(f"Validation loss: {valid_loss[-1]:.4f}\tValidation acc: {valid_acc[-1]*100:.2f}%")
            if not testData is None and not testTarget is None:
                print(f"Testing loss: {test_loss[-1]:.4f}\tTesting acc: {test_acc[-1]*100:.2f}%")

        # Check stopping condition
        if e > 1 and np.abs(train_loss[-2] - train_loss[-1]) <= error_tol:
            break

    statistics = (train_loss, train_acc)
    if not validData is None and not validTarget is None:
        statistics += (valid_loss, valid_acc, )
    if not testData is None and not testTarget is None:
        statistics += (test_loss, test_acc,)
    # Python 3.8 made this easier, but 3.7 you have to do this
    out = (w, b, *statistics)
    
    return out


# 3. Tuning the Learning Rate
N = trainData.shape[0]
d = trainData.shape[1] * trainData.shape[2]
for alpha in [0.005, 0.001, 0.0001]:
    
    print("alpha =", alpha)
    
    w = np.random.random_sample(d)
    b = np.random.random_sample(1)
    w, b, *statistics = grad_descent_MSE(w, b, trainData, trainTarget, alpha, 50, 0, 0.01, validData, validTarget, testData, testTarget)
    train_loss, train_acc, valid_loss, valid_acc, test_loss, test_acc = statistics
    
    print(f"Training loss: {train_loss[-1]:.4f}{'':.20s}Training acc: {train_acc[-1]*100:.2f}%")
    print(f"Validation loss: {valid_loss[-1]:.4f}\tValidation acc: {valid_acc[-1]*100:.2f}%")
    print(f"Testing loss: {test_loss[-1]:.4f}\tTesting acc: {test_acc[-1]*100:.2f}%")
    
    fig, ax = plt.subplots(1, 2, figsize=(18, 6))
    plot_loss(np.arange(0, len(train_loss), 1), train_loss, valid_loss, test_loss, ax=ax[0])
    plot_accuracy(np.arange(0, len(train_loss), 1), train_acc, valid_acc, test_acc, ax=ax[1])
    plt.show()
    plt.close()

"""
        Least Squares Method
"""
def least_squares(X, y):
    N = X.shape[0]
    d = X.shape[1] * X.shape[2]
    X, _ = augment(X, np.zeros(X.shape[0]), 0)
    y = y.reshape(-1)
    
    # overparameterized (deep learning)
    if N < d:
        w_aug = X.T.dot(np.linalg.inv( np.dot(X, X.T) )).dot(y)
    # underparameterized (typical case)
    else:
        w_aug = np.linalg.inv( X.T @ X ) @ X.T @ y
    
    return w_aug[1:], w_aug[0]
        
# compare above to gradient descent solution
w_LS, b_LS = least_squares(trainData, trainTarget)


"""
        CE Loss
"""
# this will work for both scalar and vector z
def sigmoid(z):
    return 1.0 / (1 + np.exp(-z))
    

# Cross Entropy Loss
def crossEntropyLoss(w, b, X, y, reg):    
    X, w = augment(X, w, b)
    y = y.reshape(-1)
    N = y.shape[0]
    
    y_hat = sigmoid(X.dot(w))
    
    return 1.0/N * (-y.dot(np.log(y_hat)) - (1 - y).dot(np.log(1 - y_hat))) + reg/2.0 * np.square(w[1:]).sum()
    

def gradCE(w, b, X, y, reg):
    X, w = augment(X, w, b)
    y = y.reshape(-1)
    N = y.shape[0]
    
    y_hat = sigmoid(X.dot(w))
    
    w_grad = 1.0 /N * X.T.dot(y_hat - y) + reg * w
    
    return w_grad[1:], w_grad[0] - reg * w[0]



"""
X = trainData
y = trainTarget
N = X.shape[0]
d = X.shape[1] * X.shape[2]

w = np.random.random_sample(d)
b = np.random.random_sample(1)

print(gradCE(w_LS, b_LS, X, y, 0))
print(gradCE_test(w_LS, b_LS, X, y, 0))
"""

"""
        GD with either MSE or CE
"""
def grad_descent(W, b, x, y, alpha, epochs, reg, error_tol, lossType="MSE"):
    loss_func, grad_func = None, None
    if lossType == "MSE":
        loss_func, grad_func = MSE, gradMSE
    elif lossType == "CE":
        loss_func, grad_func = crossEntropyLoss, gradCE
    else:
        raise ValueError("Variable 'lossType' must be either 'MSE' or 'CE'.")

    train_loss, train_acc = [], []
    valid_loss, valid_acc = [], []
    test_loss, test_acc = [], []
    printing = True
    for e in range(epochs):
        grad_w, grad_b = grad_func(w, b, X, y, reg)
        w -= alpha * grad_w
        b -= alpha * grad_b
        
        # Calculating Statistics
        train_loss.append( loss_func(w, b, X, y, reg) )
        train_acc.append( accuracy(w, b, X, y) )

        if validData != None and validTarget != None:
            valid_loss.append( loss_func(w, b, validData, validTarget, reg) )
            valid_acc.append( accuracy(w, b, validData, validTarget) )
        if testData != None and testTarget != None:
            test_loss.append( loss_func(w, b, testData, testTarget, reg) )
            valid_acc.append( accuracy(w, b, testData, testTarget) )
        
        # Print Losses and Accurancies if printing is on
        if printing:
            print(f"Training loss: {train_loss[-1]:.4f}\tTraining acc: {train_acc[-1]*100:.2f}%")
            if validData != None and validTarget != None:
                print(f"Validation loss: {valid_loss[-1]:.4f}\tValidation acc: {valid_acc[-1]*100:.2f}%")
            if testData != None and testTarget != None:
                print(f"Testing loss: {test_loss[-1]:.4f}\tTesting acc: {test_acc[-1]*100:.2f}%")

        # Check stopping condition
        if e > 1 and np.abs(train_loss[-2] - train_loss[-1]) <= error_tol:
            break

    statistics = (train_loss, train_acc)
    if validData != None and validTarget != None:
        statistics += (valid_loss, valid_acc, )
    if testData != None and testTarget != None:
        statistics += (test_loss, test_acc,)
    # Python 3.8 made this easier, but 3.7 you have to do this
    out = (w, b, *statistics)
    
    return out

"""
        Stochastic Gradient Descent
"""
# Implement the SGD algorithm for a minibatch size of 500 
# optimizing over 700 epochs 2, minimizing the MSE (you will repeat this for the CE later).
# Calculate the total number of batches required by dividing the number
# of training instances by the minibatch size. After each epoch you will need to reshuffle the
# training data and start sampling from the beginning again. Initially, set \lambda = 0 and continue
# to use the same \alpha value (i.e. 0.001). After each epoch, store the training, validation and test
# losses and accuracies. Use these to plot the loss and accuracy curves.

# Implement the SGD algorithm for a minibatch size of 500 

class BatchLoader(object):

    def __init__(self, data, batch_size=None, randomize=True, drop_last=False, seed=None):
    
        # error checking
        if len(data) > 1:
            for i in range(len(data)-1):
                if data[i].shape[0] != data[i+1].shape[0]:
                    raise ValueError("All inputs must have the same number of elements")
    
        self.data = data if type(data) == tuple else (data, )
        self.N = data[0].shape[0]
        self.batch_size = batch_size if batch_size != None else self.N
        self.drop_last = drop_last

        # shuffling data
        if randomize:
            indices = np.arange(self.N)
            np.random.seed(seed)
            np.random.shuffle(indices)
            self.data = tuple([d[indices] for d in self.data])
    
        self.index = 0 

    def __iter__(self):
        return self
    
    def __next__(self):
    
        # stop condition
        if self.index >= self.N:
            self.index = 0          # resetting index for next iteration
            raise StopIteration

        # iterating
        self.index += self.batch_size
    
        if self.index > self.N:
            if self.drop_last:
                self.index = 0      # resetting index for next iteration
                raise StopIteration
            else:
                #return self.index - self.batch_size, "end"
                return tuple([ d[self.index - self.batch_size: ] for d in self.data ])
        else:
            #return self.index - self.batch_size, self.index
            return tuple([ d[self.index - self.batch_size: self.index] for d in self.data ])

def SGD(w, b, X, y, alpha, epochs, reg, error_tol, batch_size, lossType="MSE", 
                 validData=None, validTarget=None, testData=None, testTarget=None, randomize=False):
    loss_func, grad_func = None, None
    if lossType == "MSE":
        loss_func, grad_func = MSE, gradMSE
    elif lossType == "CE":
        loss_func, grad_func = crossEntropyLoss, gradCE
    else:
        raise ValueError("Variable 'lossType' must be either 'MSE' or 'CE'.")
    
    train_loss, train_acc = [], []
    valid_loss, valid_acc = [], []
    test_loss, test_acc = [], []
    printing = False
    
    batch_iter = BatchLoader((X, y), batch_size=batch_size)
    
    running_loss = 0.0
    running_acc = 0.0
    
    for i in range(epochs):
        for batch, targets in batch_iter:
            grad_w, grad_b = grad_func(w, b, batch, targets, reg)
            w -= alpha * grad_w
            b -= alpha * grad_b
            
            # Calculating Statistics
            running_loss += loss_func(w, b, batch, targets, reg) * batch.shape[0]
            running_acc += accuracy(w, b, batch, targets) * batch.shape[0]
            
            # Check stopping condition
            if i > 1 and np.abs(train_loss[-2] - train_loss[-1]) <= error_tol:
                break
        else:
            # Calculating Statistics
            train_loss.append(running_loss / X.shape[0])
            train_acc.append(running_acc / X.shape[0])
            running_loss = 0.0
            running_acc = 0.0
            
            if validData is not None and validTarget is not None:
                valid_loss.append(loss_func(w, b, validData, validTarget, reg))
                valid_acc.append(accuracy(w, b, validData, validTarget))
            if testData is not None and testTarget is not None:
                test_loss.append(loss_func(w, b, testData, testTarget, reg))
                test_acc.append(accuracy(w, b, testData, testTarget))

            # Print Losses and Accurancies if printing is on
            if printing:
                print(f"Training loss: {train_loss[-1]:.4f}\tTraining acc: {train_acc[-1] * 100:.2f}%")
                if validData is not None and validTarget is not None:
                    print(f"Validation loss: {valid_loss[-1]:.4f}\tValidation acc: {valid_acc[-1] * 100:.2f}%")
                if testData is not None and testTarget is not None:
                    print(f"Testing loss: {test_loss[-1]:.4f}\tTesting acc: {test_acc[-1] * 100:.2f}%")
            
            continue
        break

    statistics = (train_loss, train_acc)
    if validData is not None and validTarget is not None:
        statistics += (valid_loss, valid_acc,)
    if testData is not None and testTarget is not None:
        statistics += (test_loss, test_acc,)
    out = (w, b, *statistics)

    return out

"""
X = trainData
N = X.shape[0]
d = X.shape[1] * X.shape[2]

w = np.random.random_sample(d)
w = w - w.mean()
b = np.random.random_sample(1)
w, b, *statistics = SGD(w, b, trainData, trainTarget, 0.005, 100, 0.1, 0.0001, 100, "CE", validData, validTarget, testData, testTarget)
train_loss, train_acc, valid_loss, valid_acc, test_loss, test_acc = statistics
#train_loss, train_acc = statistics

fig, ax = plt.subplots(1, 2, figsize=(18, 6))
plot_loss(np.arange(0, len(train_loss), 1), train_loss, valid_loss, test_loss, ax=ax[0])
plot_accuracy(np.arange(0, len(train_loss), 1), train_acc, valid_acc, test_acc, ax=ax[1])
plt.show()
plt.clf()
"""

def buildGraph(loss="MSE"):
    #Initialize weight and bias tensors
    tf.set_random_seed(421)

    loss_func, grad_func = None, None
    if loss == "MSE":
        loss_func, grad_func = MSE, gradMSE
    elif loss == "CE":
        loss_func, grad_func = crossEntropyLoss, gradCE
    else:
        raise ValueError("Variable 'loss' must be either 'MSE' or 'CE'.")

    
"""
Some Latex Stuff
$$
\mathcal{L} = \frac{1}{N}\sum_{n=1}^{N} \left [ -y^{(n)} \log( \sigma (W^T\textbf{x}^{(n)} + b)) -(1- y^{(n)}) - \log (1 - \sigma (W^T\textbf{x}^{(n)} + b) ) \right ] + \frac{\lambda}{2} \Vert W \Vert^2_2
$$

$$
\frac{\partial \mathcal{L}}{\partial \boldsymbol{b}} = \frac{1}{N}\sum_{n=1}^{N} \left [ -\frac{y^{(n)}}{\sigma (W^T\textbf{x}^{(n)} + b)} + \frac{1- y^{(n)}}{1 - \sigma (W^T\textbf{x}^{(n)} + b)} \right ] \cdot \sigma' (W^T\textbf{x}^{(n)} + b) \cdot 1
$$

$$
\frac{\partial \mathcal{L}}{\partial W} = \frac{1}{N} \sum_{n=1}^{N} \left [ -\frac{y^{(n)}}{\sigma (W^T\textbf{x}^{(n)} + b)} + \frac{1- y^{(n)}}{1 - \sigma (W^T\textbf{x}^{(n)} + b)} \right ] \cdot \sigma' (W^T\textbf{x}^{(n)} + b) \cdot \textbf{x}^{(n)} + \lambda W
$$
"""