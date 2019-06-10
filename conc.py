import pandas as pd
import numpy as np
from sklearn import preprocessing, model_selection as ms      #for feature normalization and train_test_split
import scipy.optimize as opt                                  #for using minimize function

import warnings        #for ignoring DataConversionWarning errors
from sklearn.exceptions import DataConversionWarning
warnings.filterwarnings(action='ignore', category=DataConversionWarning)

def sigmoid(z):                         #sigmoid function
    return 1 / (1 + np.exp(-z))    
    
def sigmoid_grad(z):                    #calculates gradient of sigmoid function
    g = sigmoid(z)
    return np.multiply(g, 1 - g)

def rand_init(features, hidden, output):                    #randomly initializes NN parameters
    theta1 = (2 * np.random.rand(hidden, features)) - 1
    theta2 = (2 * np.random.rand(output, hidden+1)) - 1
    return (theta1, theta2)

def unroll(mat1, mat2):         #unrolls all Parameter matrices into a single vector for minimize function
    m1, n1 = mat1.shape
    m2, n2 = mat2.shape
    
    vec1 = mat1.ravel()
    vec2 = mat2.ravel()
    
    return np.concatenate([vec1, vec2])

def cost_func(Theta, X, y, reg_const, hid, op):     #calculates cost 
    m, n= X.shape
    pos = n * hid
    
    theta1 = Theta[0: pos]
    theta2 = Theta[pos:]

    theta1 = theta1.reshape((hid, n))
    theta2 = theta2.reshape((op, hid+1))
    
    Y = np.reshape((y.values).T, (m, 1))
    A1 = X
    Z2 = np.dot(A1, theta1.T)
    A2 = sigmoid(Z2)
    A2 = np.column_stack(( np.ones(m), A2))
    
    H = np.dot(A2, theta2.T)
    
    Diff = H - Y
    cost = Diff * Diff 
    
    J = (1/(m*2)) * np.sum(cost)
    
    reg1 = theta1[:, 1:]
    reg2 = theta2[:, 1:]
    
    reg1 = reg1 * reg1
    reg2 = reg2 * reg2
    
    J = J + (( reg_const/(2*m) ) * ( np.sum(reg1) + np.sum(reg2) )) 
    #print(J)
    return J
    
def gradient(Theta, X, y, reg_const, hid, op):              #calculates gradient using backward propagation
    m, n= X.shape
    pos = n * hid
    
    theta1 = Theta[0: pos]
    theta2 = Theta[pos:]

    theta1 = theta1.reshape((hid, n))
    theta2 = theta2.reshape((op, hid+1))
    
    Y = np.reshape((y.values).T, (m, 1))
    
    grad1 = np.zeros(theta1.shape)
    grad2 = np.zeros(theta2.shape)
    
    for t in range(m):
        x = X[t, :]
        a1 = x.reshape((1,n))
        z2 = np.dot(a1, theta1.T)
        a2 = sigmoid(z2)
        a2 = np.column_stack( (np.ones(a2.shape[0]), a2) )
        z2 = np.column_stack( (np.ones(z2.shape[0]), z2) )
        h = np.dot(a2, theta2.T)
        
        diff = h - Y[t, :]
        
        del1 = np.dot(diff, theta2) * sigmoid_grad(z2)
        Del1 = np.dot(del1.T, a1)
        Del1 = Del1[1:, :]
        
        grad1 = grad1 + Del1
        grad2 = grad2 + (np.dot(diff, a2))
    
    grad1 = grad1/m;
    grad2 = grad2/m;
    
    sum1 = theta1
    sum2 = theta2
    
    sum1[:, 0] = np.zeros((sum1[:, 0]).shape)
    sum2[:, 0] = np.zeros((sum2[:, 0]).shape)
    
    grad1 = grad1 + (reg_const/m) * sum1
    grad2 = grad2 + (reg_const/m) * sum2
    
    vec = unroll(grad1, grad2)
    
    return vec

def accuracy(Theta, X, y, reg_const, hid, op):              #calculates accuracy of the trained NN
    m, n= X.shape
    pos = n * hid
    
    theta1 = Theta[0: pos]
    theta2 = Theta[pos:]

    theta1 = theta1.reshape((hid, n))
    theta2 = theta2.reshape((op, hid+1))
    
    Y = np.reshape((y.values).T, (m, 1))
    A1 = X
    Z2 = np.dot(A1, theta1.T)
    A2 = sigmoid(Z2)
    A2 = np.column_stack(( np.ones(m), A2))
    
    H = np.dot(A2, theta2.T)
    
    Diff = H - Y
   
    for i in range(m):
       print("Real strength: ",Y[i])
       print("Predicted strength: ", H[i],"\n")
        
    val = cost_func(Theta, X, y, reg_const, hid, op)
    val = val * m;
    
    val = np.sqrt(val)
    sum_val = np.sum(Y)
    
    return 100 - (val/sum_val)
    
data = pd.read_csv("concrete.csv")
labels = list(data.columns)

# 3 Layers with 1 input, 1 hidden, and 1 output layer

m, fnum = data.shape    #Number of input neurons   
hid = 10              #Number of hidden neurons
out = 1                 #Number of output neurons

lamb = 0

X = data.iloc[:, 0 : fnum - 1]
y = data.strength

X_scaled = preprocessing.scale(X)                       #feature scaling
X_final = np.column_stack(( np.ones(m), X_scaled) ) 

X_train, X_test, y_train, y_test = ms.train_test_split(X_final, y, test_size = 0.20)      #splitting data in 7:3 ratio for training and testing

theta1, theta2 = rand_init(fnum, hid, out)      #initialising parameters
theta = unroll(theta1, theta2)                  #unrolling parameter matrices into single vector

print("Training Neural Network...\n\n")

Result = opt.minimize(fun = cost_func, x0 = theta, args = (X_train, y_train, lamb, hid, out), method = 'BFGS', jac = gradient) #Minimizing Cost Function 
opt_theta = Result.x

print("\nTraining complete :)\n")
print("Optimum Parameters:", opt_theta, "\n\n")
print("\nAccuracy obtained: ", accuracy(opt_theta, X_test, y_test, 0, hid, out ),"%\n\n")   

