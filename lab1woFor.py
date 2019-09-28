import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import mnist

X_size = 5
Y_size = 3

(X_train, y_train), (X_test, y_test) = mnist.load_data()

nummas = np.random.randint(0, 60000, X_size*Y_size)

vvv = X_train[nummas]
vvv = vvv.reshape(Y_size,X_size,28,28)
vvv = vvv.transpose(0,2,1,3).reshape(Y_size*28,X_size*28)

plt.imshow(vvv , cmap='Greys')
plt.show()
