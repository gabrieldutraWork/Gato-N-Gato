import numpy as np
import h5py
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

def load_data():
    train_dataset = h5py.File('../data/train_catvnoncat.h5', "r")
    train_set_x_orig = np.array(train_dataset["train_set_x"][:10]) 
    train_set_y_orig = np.array(train_dataset["train_set_y"][:10]) 

    test_dataset = h5py.File('../data/test_catvnoncat.h5', "r")
    test_set_x_orig = np.array(test_dataset["test_set_x"][:]) 
    test_set_y_orig = np.array(test_dataset["test_set_y"][:]) 

    train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))
    test_set_y_orig = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))

    return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig



train_x_orig, train_y, test_x_orig, test_y = load_data()

train_x_flatten = train_x_orig.reshape(train_x_orig.shape[0], -1) / 255
test_x_flatten = test_x_orig.reshape(test_x_orig.shape[0], -1) / 255

clf = LogisticRegression(random_state=0).fit(train_x_flatten, train_y.ravel())

test_y_pred = clf.predict(test_x_flatten)
accuracy = accuracy_score(test_y_pred, test_y.T)
print('accuracy: ' + str(accuracy))
cm = confusion_matrix(test_y.T, test_y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['noncat', 'cat'], yticklabels=['noncat', 'cat'])
plt.xlabel('Predito')
plt.ylabel('Real')
plt.title('Matriz de Confusão')
plt.show()
print(classification_report(test_y.ravel(), test_y_pred, target_names=['cat', 'noncat']))