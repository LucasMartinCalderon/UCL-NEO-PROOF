from __future__ import print_function
from parser_1 import parameter_parser
import tensorflow as tf
import numpy as np
from sklearn.utils import compute_class_weight
from sklearn.metrics import confusion_matrix

# Eager execution is enabled by default in TF 2.0 or above
tf.random.set_seed(6603)

print(tf.__version__)
args = parameter_parser()

"""
The graph feature and the pattern feature are fed into the AME network for giving the final detection result and the 
interpretable weights.
"""


class CrossAttentionEncoder:
    def __init__(self, graph_train, graph_test, pattern1train, pattern2train, pattern3train, pattern1test, pattern2test,
                 pattern3test, y_train, y_test, batch_size=args.batch_size, lr=args.lr, epochs=args.epochs):
                
        graph_input = tf.keras.Input(shape=(1,), name='graph_input')  # Updated shape to (1,)

        pattern1_input = tf.keras.Input(shape=(250,), name='pattern1_input')  # Same here
        pattern2_input = tf.keras.Input(shape=(250,), name='pattern2_input')  # And here
        pattern3_input = tf.keras.Input(shape=(250,), name='pattern3_input')  # And here too

        self.graph_train = graph_train
        self.graph_test = graph_test
        self.pattern1train = pattern1train
        self.pattern2train = pattern2train
        self.pattern3train = pattern3train
        self.pattern1test = pattern1test
        self.pattern2test = pattern2test
        self.pattern3test = pattern3test
        self.y_train = y_train
        self.y_test = y_test
        self.batch_size = batch_size
        self.epochs = epochs
        self.class_weight = dict(enumerate(compute_class_weight(class_weight='balanced', classes=[0, 1], y=y_train)))
        
        print("NEO sizes:")
        print(self.graph_train.shape)
        print(self.pattern1train.shape)
        print(self.pattern2train.shape)
        print(self.pattern3train.shape)

        
        graph2vec = tf.keras.layers.Dense(250, activation='relu', name='outputgraphvec')(graph_input)

        graphweight = tf.keras.layers.Dense(1, activation='sigmoid', name='outputgraphweight')(graph2vec)
        newgraphvec = tf.keras.layers.Multiply(name='outputnewgraphvec')([graph2vec, graphweight])

        pattern1vec = tf.keras.layers.Dense(250, activation='relu', name='outputpattern1vec')(pattern1_input)
        pattern1weight = tf.keras.layers.Dense(1, activation='sigmoid', name='outputpattern1weight')(pattern1vec)
        newpattern1vec = tf.keras.layers.Multiply(name='newpattern1vec')([pattern1vec, pattern1weight])

        pattern2vec = tf.keras.layers.Dense(250, activation='relu', name='outputpattern2vec')(pattern2_input)
        pattern2weight = tf.keras.layers.Dense(1, activation='sigmoid', name='outputpattern2weight')(pattern2vec)
        newpattern2vec = tf.keras.layers.Multiply(name='newpattern2vec')([pattern2vec, pattern2weight])

        pattern3vec = tf.keras.layers.Dense(250, activation='relu', name='outputpattern3vec')(pattern3_input)
        pattern3weight = tf.keras.layers.Dense(1, activation='sigmoid', name='outputpattern3weight')(pattern3vec)
        newpattern3vec = tf.keras.layers.Multiply(name='newpattern3vec')([pattern3vec, pattern3weight])
      
        mergevec = tf.keras.layers.Concatenate(axis=1, name='mergevec')([newgraphvec, newpattern1vec, newpattern2vec, newpattern3vec])
        semantic_vector = tf.keras.layers.Dense(250, activation='relu', name='semantic_vector')(mergevec)  

        graph_weight = tf.keras.layers.Dot(axes=1, normalize=True, name='graph_weight')([newgraphvec, semantic_vector])  
        pattern1_weight = tf.keras.layers.Dot(axes=1, normalize=True, name='pattern1_weight')([newpattern1vec, semantic_vector])  
        pattern2_weight = tf.keras.layers.Dot(axes=1, normalize=True, name='pattern2_weight')([newpattern2vec, semantic_vector])  
        pattern3_weight = tf.keras.layers.Dot(axes=1, normalize=True, name='pattern3_weight')([newpattern3vec, semantic_vector])  

        weighted_graph = tf.keras.layers.Multiply(name='weighted_graph')([newgraphvec, graph_weight])
        weighted_pattern1 = tf.keras.layers.Multiply(name='weighted_pattern1')([newpattern1vec, pattern1_weight])
        weighted_pattern2 = tf.keras.layers.Multiply(name='weighted_pattern2')([newpattern2vec, pattern2_weight])
        weighted_pattern3 = tf.keras.layers.Multiply(name='weighted_pattern3')([newpattern3vec, pattern3_weight])

        combined_features = tf.keras.layers.Concatenate(axis=1, name='combined_features')(
            [weighted_graph, weighted_pattern1, weighted_pattern2, weighted_pattern3])

        final_features = tf.keras.layers.Dense(250, activation='relu', name='final_features')(combined_features)  
        prediction = tf.keras.layers.Dense(1, activation='sigmoid', name='output')(final_features)

        model = tf.keras.Model(inputs=[graph_input, pattern1_input, pattern2_input, pattern3_input], outputs=[prediction])

        adama = tf.keras.optimizers.Adam(lr)
        model.compile(optimizer=adama, loss='binary_crossentropy', metrics=['accuracy'])
        model.summary()

        self.model = model
        self.finalmergevec = final_features

        # Training and testing methods remain the same 
        
    def get_encoded_features(self):
        # Assuming that you want to return the features after training
        # and testing. Adjust this part based on your actual needs.

        # Make sure that your model and data are ready before calling this method.
        if self.model is None:
            raise Exception("The model is not trained yet.")

        # Get the encoded features from the model
        encoder_model = tf.keras.Model(inputs=self.model.input, outputs=self.finalmergevec)
        encoded_features_train = encoder_model.predict(
            [self.graph_train, self.pattern1train, self.pattern2train, self.pattern3train])
        encoded_features_test = encoder_model.predict(
            [self.graph_test, self.pattern1test, self.pattern2test, self.pattern3test])

        return encoded_features_train, encoded_features_test
        
    """
    Training model
    """

    def train(self):
        train_history = self.model.fit([self.graph_train, self.pattern1train, self.pattern2train, self.pattern3train],
                                       self.y_train, batch_size=self.batch_size, epochs=self.epochs,
                                       class_weight=self.class_weight, validation_split=0.1, verbose=2)

        finalvec = tf.keras.Model(inputs=self.model.input, outputs=self.model.get_layer('combined_features').output)
        finalvec_output = finalvec.predict(
            [self.graph_train, self.pattern1train, self.pattern2train, self.pattern3train])
        
        # No need to create another dense layer here, you can directly use the outputs
        print(finalvec_output.shape)

    """
    Testing model
    """

    def test(self):
        # self.model.load_weights("_model.pkl")
        values = self.model.evaluate([self.graph_test, self.pattern1test, self.pattern2test, self.pattern3test],
                                     self.y_test, batch_size=self.batch_size, verbose=1)
        print("Loss: ", values[0], "Accuracy: ", values[1])

        # graphweight
        graphweight = tf.keras.Model(inputs=self.model.input, outputs=self.model.get_layer('outputgraphweight').output)

        graphweight_output = graphweight.predict(
            [self.graph_test, self.pattern1test, self.pattern2test, self.pattern3test])

        # pattern1weight
        pattern1weight = tf.keras.Model(inputs=self.model.input,
                                        outputs=self.model.get_layer('outputpattern1weight').output)
        pattern1weight_output = pattern1weight.predict(
            [self.graph_test, self.pattern1test, self.pattern2test, self.pattern3test])

        # pattern2weight
        pattern2weight = tf.keras.Model(inputs=self.model.input,
                                        outputs=self.model.get_layer('outputpattern2weight').output)
        pattern2weight_output = pattern2weight.predict(
            [self.graph_test, self.pattern1test, self.pattern2test, self.pattern3test])

        # pattern3weight
        pattern3weight = tf.keras.Model(inputs=self.model.input,
                                        outputs=self.model.get_layer('outputpattern3weight').output)
        pattern3weight_output = pattern3weight.predict(
            [self.graph_test, self.pattern1test, self.pattern2test, self.pattern3test])

        # output the weights
        print("start")
        gw = graphweight_output.flatten()
        np.savetxt("results/re_gw.txt", gw)
        g_av = gw.mean()
        print("gw_mean:", g_av, "gw_all:", gw.var())
        pw1 = pattern1weight_output.flatten()
        np.savetxt("results/re_pw1.txt", pw1)
        pw1_av = pw1.mean()
        print("pw1_mean:", pw1_av, "pw1_all:", pw1.var())
        pw2 = pattern2weight_output.flatten()
        np.savetxt("results/re_pw2.txt", pw2)
        pw2_av = pw2.mean()
        print("pw2_mean:", pw2_av, "pw2_all:", pw2.var())
        pw3 = pattern3weight_output.flatten()
        np.savetxt("results/re_pw3.txt", pw3)
        pw3_av = pw3.mean()
        print("pw3_mean:", pw3_av, "pw3_all:", pw3.var())
        f = open("results/re_weights.txt", 'a')
        f.write(
            "g_av: " + str(g_av) + ", gw_all :" + str(gw.var()) + "\n pw1_mean:" + str(pw1_av) + ", pw1_all:" + str(
                pw1.var()) + "\n pw2_mean: " + str(pw2_av) + ", pw2_all: " + str(pw2.var()) + "\n pw3_mean: " + str(
                pw3_av) + ", pw3_all: " + str(pw3.var()) + "\n")

        print("end")

        # decoder the testing vectors
        finalvec = tf.keras.Model(inputs=self.model.input, outputs=self.model.get_layer('combined_features').output)
        finalvec_output = finalvec.predict(
            [self.graph_test, self.pattern1test, self.pattern2test, self.pattern3test])
        finalveclayer = tf.keras.layers.Dense(1000, activation='relu')
        finalvec = finalveclayer(finalvec_output)
        finalvecvalue = finalvec.numpy()
        value = np.hsplit(finalvecvalue, 4)
        # print(value)
        # print(value[0].shape, value[1].shape, value[2].shape, value[3].shape)
        print(finalvec_output.shape)

        # predictions
        predictions = self.model.predict([self.graph_test, self.pattern1test, self.pattern2test, self.pattern3test],
                                         batch_size=self.batch_size).round()
        print('predict:')
        predictions = predictions.flatten()
        print(predictions)
        tn, fp, fn, tp = confusion_matrix(self.y_test, predictions).ravel()
        print("Accuracy: ", (tp + tn) / (tp + tn + fp + fn))
        print('False positive rate(FPR): ', fp / (fp + tn))
        print('False negative rate(FN): ', fn / (fn + tp))
        recall = tp / (tp + fn)
        print('Recall(TPR): ', recall)
        precision = tp / (tp + fp)
        print('Precision: ', precision)
        print('F1 score: ', (2 * precision * recall) / (precision + recall))


