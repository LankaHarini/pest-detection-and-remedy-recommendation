import tensorflow as tf

print("Converting binary_model.h5 ...")
m1 = tf.keras.models.load_model("binary_model.h5")
c1 = tf.lite.TFLiteConverter.from_keras_model(m1)
with open("binary_model.tflite", "wb") as f:
    f.write(c1.convert())
print("binary_model.tflite created ✓")

print("Converting multi_model.h5 ...")
m2 = tf.keras.models.load_model("multi_model.h5")
c2 = tf.lite.TFLiteConverter.from_keras_model(m2)
with open("multi_model.tflite", "wb") as f:
    f.write(c2.convert())
print("multi_model.tflite created ✓")

print("\nDone! Copy files to Raspberry Pi.")