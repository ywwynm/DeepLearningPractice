import model

model.train_and_evaluate("vgg_16", epochs=200, train_batch_size=32, learning_rate=1e-5, optimizer="rmsprop")