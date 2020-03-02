import cv2
import numpy as np
from keras.models import load_model
import sys

model = load_model('model.h5')

image_path = sys.argv[1]
img = cv2.imread(image_path, 1)

img = cv2.resize(img, (28, 28))
print(np.shape(img))
img = np.reshape(img, (1, 28, 28, 3))

img = img.astype('float32')
img = img / 255

probs = model.predict(img)
floatProbs = []
for prob in probs[0]:
    floatProbs.append("%.2f" % (float(prob)))
print(floatProbs)
y_prob = model.predict_classes(img)
print(y_prob)
