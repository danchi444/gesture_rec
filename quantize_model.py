import numpy as np
import tensorflow as tf
from keras.models import load_model
from tcn import TCN

model = load_model("gesture_model.h5", custom_objects={"TCN": TCN})

def representative_data_gen():
    data = np.load("train_dataset.npz")
    X = data['X']
    for i in range(100):
        yield [X[i:i+1].astype(np.float32)]

converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_data_gen

converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]

converter.inference_input_type = tf.int8
converter.inference_output_type = tf.int8

try:
    tflite_model = converter.convert()
except Exception as e:
    print("Error during TFLite conversion:", e)
    raise

with open("model_quantized.tflite", "wb") as f:
    f.write(tflite_model)

with open("final/model_quantized.h", "w") as f:
    f.write('alignas(16) const unsigned char model_tflite[] = {\n')
    hex_array = ',\n'.join(
        [', '.join(f'0x{b:02x}' for b in tflite_model[i:i+12]) for i in range(0, len(tflite_model), 12)]
    )
    f.write(hex_array)
    f.write('\n};\n')
    f.write(f'const int model_tflite_len = {len(tflite_model)};\n')