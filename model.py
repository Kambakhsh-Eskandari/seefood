### import libraries
import os
from statistics import mode
import tensorflow as tf
import numpy as np

## load test
datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)

test_path = 'test/'
test_generator = datagen.flow_from_directory(test_path, shuffle=True, target_size=(224,224))



# load model 
model = tf.keras.models.load_model('./models/model_1')
pred=model.predict_generator(test_generator,verbose=1)

pred = np.argmax(pred, axis=1)

labels = test_generator.class_indices
labels = dict((v,k) for (k,v) in labels.items())

predictions = [labels[k] for k in pred]
