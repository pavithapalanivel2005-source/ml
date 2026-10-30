import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import mnist
import matplotlib.pyplot as plt

(x_train,_),(x_test,_) = mnist.load_data()

x_train = x_train.reshape(-1,784)/255.0
x_test = x_test.reshape(-1,784)/255.0

inp = tf.keras.Input(shape=(784,))
encoded = layers.Dense(32, activation='relu')(inp)
decoded = layers.Dense(784, activation='sigmoid')(encoded)

model = models.Model(inp, decoded)
model.compile(optimizer='adam', loss='binary_crossentropy')

model.fit(x_train, x_train, epochs=5, batch_size=256,
          validation_data=(x_test,x_test))

output = model.predict(x_test[:5])

plt.figure(figsize=(10,4))

for i in range(5):
    plt.subplot(2,5,i+1)
    plt.imshow(x_test[i].reshape(28,28), cmap='gray')
    plt.title("Original")
    plt.axis('off')

    plt.subplot(2,5,i+6)
    plt.imshow(output[i].reshape(28,28), cmap='gray')
    plt.title("Reconstructed")
    plt.axis('off')

plt.show()