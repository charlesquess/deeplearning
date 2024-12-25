import tensorflow as tf

print(tf.__version__)

physical_devices = tf.config.list_physical_devices()
print("Physical devices: ", physical_devices)

# 检查是否有 GPU
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print("GPUs detected: ", gpus)
else:
    print("No GPUs detected, using CPU.")
