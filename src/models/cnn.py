class CNNModel:
    def __init__(self, conv_layers=2, filters=32, kernel_size=3, pooling='max'):
        self.conv_layers = conv_layers
        self.filters = filters
        self.kernel_size = kernel_size
        self.pooling = pooling
        self.model = self._build_model()

    def _build_model(self):
        model = models.Sequential()
        model.add(layers.Input(shape=(32, 32, 3)))

        for _ in range(self.conv_layers):
            model.add(layers.Conv2D(self.filters, self.kernel_size, activation='relu', padding='same'))
            if self.pooling == 'max':
                model.add(layers.MaxPooling2D())
            else:
                model.add(layers.AveragePooling2D())

        model.add(layers.Flatten())
        model.add(layers.Dense(128, activation='relu'))
        model.add(layers.Dense(10))
        return model

    def compile(self):
        self.model.compile(
            loss=losses.SparseCategoricalCrossentropy(from_logits=True),
            optimizer=optimizers.Adam(),
            metrics=['accuracy']
        )

    def train(self, x_train, y_train, x_val, y_val, epochs=2, batch_size=64):
        return self.model.fit(
            x_train, y_train,
            validation_data=(x_val, y_val),
            epochs=epochs,
            batch_size=batch_size
        )

    def evaluate(self, x_test, y_test):
        y_pred_logits = self.model.predict(x_test)
        y_pred = np.argmax(y_pred_logits, axis=1)
        f1 = f1_score(y_test, y_pred, average='macro')
        print("Test Macro F1 Score:", f1)
        return f1

    def save_weights(self, path="cnn.weights.h5"):
        self.model.save_weights(path)

    def load_weights(self, path="cnn.weights.h5"):
        self.model.load_weights(path)


class TrainCNN:
    def __init__(self, x_train, y_train, x_val, y_val, x_test, y_test):
        self.x_train = x_train
        self.y_train = y_train
        self.x_val = x_val
        self.y_val = y_val
        self.x_test = x_test
        self.y_test = y_test

    def run(self):
        model = CNNModel()
        model.compile()
        model.train(x_train, y_train, x_val, y_val)
        model.save_weights()
        model.evaluate(x_test, y_test)