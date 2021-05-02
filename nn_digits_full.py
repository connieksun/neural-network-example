# step 1: get the dataset
# we will be using a classic "handwritten digit" dataset for a classification task
from sklearn.datasets import load_digits
digits = load_digits()

# step 2: understand the dataset
print(digits.keys()) # this is how we access our dataset
print(digits['DESCR']) # description of the dataset

'''
# this is just so we can visualize the dataset
import matplotlib.pyplot as plt
import numpy as np
fig, axs = plt.subplots(2, 5) #, figsize=(6, 15))
for i in range(10):
    axs[i//5, i%5].imshow(digits.images[i], cmap=plt.cm.gray_r, interpolation='nearest')
    axs[i//5, i%5].set_title(digits.target[i])
    axs[i//5, i%5].set_xticks(np.arange(0, 8, 2))
    axs[i//5, i%5].set_yticks(np.arange(0, 8, 2))
plt.show()
'''

# step 3: get the X (input/features) and y (target values)
X = digits['data']  # get features
y = digits['target']# get target
# now we'll print out an example so we know the input data shape
print("Example input data: " + str(X[0])) # already flattened
print("Example target data: " + str(y[0]))

# step 4: split the data into teting and training sets
# sklearn has a really handy method for this already
# we'll use a random state so results are reproducible
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)

# step 5: train our neural network
from sklearn.neural_network import MLPClassifier
nn = MLPClassifier() # all defaults
nn = nn.fit(X_train, y_train) # fit to the training data

# step 6: test our neural network
y_pred = nn.predict(X_test) # test on our testing data
from sklearn.metrics import r2_score
print(r2_score(y_test, y_pred)) # performance measure; compare to the true test vals

'''
# this is just for visualizing our predictions/performance
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, plot_confusion_matrix
disp = plot_confusion_matrix(nn, X_test, y_test)
disp.figure_.suptitle("Confusion Matrix")
plt.show()
'''

# step 7: tuning hyperparameters
# let's try tuning the hyperparameters; does this improve?
nn = MLPClassifier(hidden_layer_sizes=(300), alpha=0.001)
nn = nn.fit(X_train, y_train)
y_pred = nn.predict(X_test)
print(r2_score(y_test, y_pred))


