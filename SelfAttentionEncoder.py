from __future__ import print_function
from parser_1 import parameter_parser
import tensorflow as tf
from sklearn.utils import compute_class_weight
from sklearn.metrics import confusion_matrix
import numpy as np

tf.compat.v1.set_random_seed(6603)

print(tf.__version__)
args = parameter_parser()

class SelfAttentionEncoder:
    def __init__(self, graph_train, graph_test, pattern1train, pattern2train, pattern3train, pattern1test, pattern2test,
                 pattern3test, y_train, y_test, batch_size=args.batch_size, lr=args.lr, epochs=args.epochs):
        
        graph_input = tf.keras.Input(shape=(250,), name='graph_input')  
        pattern1_input = tf.keras.Input(shape=(250,), name='pattern1_input')  
        pattern2_input = tf.keras.Input(shape=(250,), name='pattern2_input')  
        pattern3_input = tf.keras.Input(shape=(250,), name='pattern3_input')  

        self.graph_train = graph_train if graph_train.shape[1] != 1 else np.squeeze(graph_train, axis=1)
        self.graph_test = graph_test if graph_test.shape[1] != 1 else np.squeeze(graph_test, axis=1)
        self.pattern1train = pattern1train if pattern1train.shape[1] != 1 else np.squeeze(pattern1train, axis=1)
        self.pattern2train = pattern2train if pattern2train.shape[1] != 1 else np.squeeze(pattern2train, axis=1)
        self.pattern3train = pattern3train if pattern3train.shape[1] != 1 else np.squeeze(pattern3train, axis=1)
        self.pattern1test = pattern1test if pattern1test.shape[1] != 1 else np.squeeze(pattern1test, axis=1)
        self.pattern2test = pattern2test if pattern2test.shape[1] != 1 else np.squeeze(pattern2test, axis=1)
        self.pattern3test = pattern3test if pattern3test.shape[1] != 1 else np.squeeze(pattern3test, axis=1)
        
        self.y_train = y_train
        self.y_test = y_test
        self.batch_size = batch_size
        self.epochs = epochs
        self.class_weight = compute_class_weight(class_weight='balanced', classes=[0, 1], y=y_train)

        graph2vec = tf.keras.layers.Dense(250, activation='relu', name='graph2vec')(graph_input)
        pattern1vec = tf.keras.layers.Dense(250, activation='relu', name='pattern1vec')(pattern1_input)
        pattern2vec = tf.keras.layers.Dense(250, activation='relu', name='pattern2vec')(pattern2_input)
        pattern3vec = tf.keras.layers.Dense(250, activation='relu', name='pattern3vec')(pattern3_input)


        # Self-attention coefficients computation
        graph_coef = tf.keras.layers.Dense(250, activation='sigmoid', name='graph_coef')(graph2vec)
        pattern1_coef = tf.keras.layers.Dense(250, activation='sigmoid', name='pattern1_coef')(pattern1vec)
        pattern2_coef = tf.keras.layers.Dense(250, activation='sigmoid', name='pattern2_coef')(pattern2vec)
        pattern3_coef = tf.keras.layers.Dense(250, activation='sigmoid', name='pattern3_coef')(pattern3vec)

        # Updating the features with self-attention mechanism
        graphatt = tf.keras.layers.Multiply(name='graphatt')([graph2vec, graph_coef])
        pattern1att = tf.keras.layers.Multiply(name='pattern1att')([pattern1vec, pattern1_coef])
        pattern2att = tf.keras.layers.Multiply(name='pattern2att')([pattern2vec, pattern2_coef])
        pattern3att = tf.keras.layers.Multiply(name='pattern3att')([pattern3vec, pattern3_coef])

        mergeattvec = tf.keras.layers.Concatenate(name='concatvec2')([graphatt, pattern1att, pattern2att, pattern3att])
        mergeattvec = tf.keras.layers.Dense(100, activation='relu', name='mergeattvec')(mergeattvec)

        prediction = tf.keras.layers.Flatten()(tf.keras.layers.Dense(1, activation='sigmoid', name='output')(mergeattvec))

        model = tf.keras.Model(inputs=[graph_input, pattern1_input, pattern2_input, pattern3_input], outputs=[prediction])

        adama = tf.keras.optimizers.Adam(lr)
        model.compile(optimizer=adama, loss='binary_crossentropy', metrics=['accuracy'])
        model.summary()
        self.model = model
        
    def get_encoded_features(self):
        # This method should return the encoded features for train and test datasets
        encoded_features_train = self.model.predict([self.graph_train, self.pattern1train, self.pattern2train, self.pattern3train])
        encoded_features_test = self.model.predict([self.graph_test, self.pattern1test, self.pattern2test, self.pattern3test])
        
        return encoded_features_train, encoded_features_test

    def train(self):
        class_weights = dict(enumerate(self.class_weight))  # Convert class_weight array to dictionary
        self.model.fit([self.graph_train, self.pattern1train, self.pattern2train, self.pattern3train], self.y_train,
                       batch_size=self.batch_size, epochs=self.epochs, class_weight=class_weights,  # Use the dictionary here
                       validation_split=0.2, verbose=2)

    def test(self):
        values = self.model.evaluate([self.graph_test, self.pattern1test, self.pattern2test, self.pattern3test],
                                     self.y_test, batch_size=self.batch_size, verbose=1)
        print("Loss: ", values[0], "Accuracy: ", values[1])

        predictions = (self.model.predict([self.graph_test, self.pattern1test, self.pattern2test, self.pattern3test],
                                          batch_size=self.batch_size).round())

        predictions = predictions.flatten()
        tn, fp, fn, tp = confusion_matrix(self.y_test, predictions).ravel()
        print("Accuracy: ", (tp + tn) / (tp + tn + fp + fn))
        print('False positive rate(FPR): ', fp / (fp + tn))
        print('False negative rate(FN): ', fn / (fn + tp))
        recall = tp / (tp + fn)
        print('Recall(TPR): ', recall)
        precision = tp / (tp + fp)
        print('Precision: ', precision)
        print('F1 score: ', (2 * precision * recall) / (precision + recall))
