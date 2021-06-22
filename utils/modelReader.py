import tensorflow as tf

model = tf.keras.models.load_model("model.h5")
for i in range(len(model.layers)):
    weights, biases = model.layers[i].get_weights()
    print("layer", i)
    print("weights", weights)
    print("biases", biases)
