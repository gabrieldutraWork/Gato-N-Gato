import numpy as np
import h5py
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten

# Função para carregar os dados
def load_data():
    train_dataset = h5py.File('../data/train_catvnoncat.h5', "r")
    train_set_x_orig = np.array(train_dataset["train_set_x"][:]) 
    train_set_y_orig = np.array(train_dataset["train_set_y"][:]) 

    test_dataset = h5py.File('../data/test_catvnoncat.h5', "r")
    test_set_x_orig = np.array(test_dataset["test_set_x"][:]) 
    test_set_y_orig = np.array(test_dataset["test_set_y"][:]) 

    train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))
    test_set_y_orig = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))

    return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig


train_x_orig, train_y, test_x_orig, test_y = load_data()


train_x = train_x_orig / 255.
test_x = test_x_orig / 255.


train_y = train_y.T
test_y = test_y.T


model = Sequential([
    Flatten(input_shape=(64, 64, 3)),  
    Dense(200, activation='relu'), 
    Dense(1, activation='sigmoid') 
])


model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])


history = model.fit(train_x, train_y, epochs=50, batch_size=32, validation_split=0.2)


test_loss, test_accuracy = model.evaluate(test_x, test_y)
print('accuracy: ' + str(test_accuracy))

test_y_pred = (model.predict(test_x) > 0.5).astype("int32")


cm = confusion_matrix(test_y, test_y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Não Gato', 'Gato'], yticklabels=['Não Gato', 'Gato'])
plt.xlabel('Predito')
plt.ylabel('Real')
plt.title('Matriz de Confusão')
plt.show()

print(classification_report(test_y, test_y_pred, target_names=['Não Gato', 'Gato']))
