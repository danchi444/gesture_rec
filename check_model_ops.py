import tensorflow as tf

interpreter = tf.lite.Interpreter(model_path="model_quantized.tflite")
interpreter.allocate_tensors()

ops = set()
for detail in interpreter._get_ops_details():
    ops.add(detail['op_name'])

print("Ops used:", sorted(ops))