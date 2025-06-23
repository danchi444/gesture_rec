import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout
from tcn import TCN
import tensorflow as tf
import os
from keras.callbacks import EarlyStopping

NUM_FILTERS = 6
DENSE_UNITS = 12

data = np.load('train_dataset.npz')
X = data['X']
y = data['y']

model = Sequential([
    TCN(nb_filters=NUM_FILTERS, input_shape=(X.shape[1], X.shape[2])),
    Dropout(0.3),
    Dense(DENSE_UNITS, activation='relu'),
    Dense(len(np.unique(y)), activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.summary()

early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=5, 
    restore_best_weights=True
)

model.fit(X, y, epochs=20, batch_size=8, validation_split=0.2, callbacks=[early_stopping])

model.save('gesture_model.h5')

converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

os.makedirs('final', exist_ok=True)
with open('final/model.tflite', 'wb') as f:
    f.write(tflite_model)