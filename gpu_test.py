import tensorflow as tf

print("=" * 50)
print("TensorFlow Version :", tf.__version__)

print("\nAvailable GPUs:")
print(tf.config.list_physical_devices('GPU'))

print("\nAvailable CPUs:")
print(tf.config.list_physical_devices('CPU'))
print("=" * 50)