import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.losses import MSE

model = load_model('final_model.h5')
epsilon = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.3, 0.35]


def load_dataset():
    (trainX, trainY), (testX, testY) = mnist.load_data()
    testX = testX.reshape((testX.shape[0], 28, 28, 1))
    testY = to_categorical(testY)
    testX = testX.astype('float32')
    testX = testX / 255.0
    return testX, testY


def fgsm(image, label, eps):
    image = tf.cast(image, tf.float32)
    with tf.GradientTape() as tape:
        tape.watch(image)
        prediction = model(image)
        loss = MSE(label, prediction)

    grad = tape.gradient(loss, image)
    sign = tf.sign(grad)
    sign = sign.numpy()
    perturb = eps * sign

    adversarial = image + perturb

    return adversarial


def test(testX, testY, eps):
    _, acc = model.evaluate(fgsm(testX, testY, eps), testY, verbose=0)
    print('for epsilon :', eps, '> %.3f' % (acc * 100.0))


for eps in range(len(epsilon)):
    testX, testY = load_dataset()
    test(testX, testY, epsilon[eps])





