import numpy as np
from parser_1 import parameter_parser
from models.CrossAttentionEncoder import CrossAttentionEncoder as CrossAttentionEncoder
from models.SelfAttentionEncoder import SelfAttentionEncoder as SelfAttentionEncoder
from models.MLP import MLP
from preprocessing import get_graph_feature, get_pattern_feature
import tensorflow as tf
from sklearn.utils import compute_class_weight
from sklearn.metrics import confusion_matrix

args = parameter_parser()


def main():
    graph_train, graph_test, graph_experts_train, graph_experts_test = get_graph_feature(vulnerability_type="reentrancy")
    pattern_train, pattern_test, label_by_extractor_train, label_by_extractor_valid = get_pattern_feature(vulnerability_type="reentrancy")

    graph_train = np.array(graph_train)
    graph_test = np.array(graph_test)

    pattern1train = np.array([x[0] for x in pattern_train])
    pattern2train = np.array([x[1] for x in pattern_train])
    pattern3train = np.array([x[2] for x in pattern_train])

    pattern1test = np.array([x[0] for x in pattern_test])
    pattern2test = np.array([x[1] for x in pattern_test])
    pattern3test = np.array([x[2] for x in pattern_test])


    y_train = np.array(graph_experts_train).astype(int)
    y_test = np.array(graph_experts_test).astype(int)

    y_train_pattern = np.array(label_by_extractor_train).astype(int)
    y_test_pattern = np.array(label_by_extractor_valid).astype(int)

    # Step 1: Self-Attention Encoder
    selfAttentionModel = SelfAttentionEncoder(
        graph_train, graph_test, pattern1train, pattern2train, pattern3train,
        pattern1test, pattern2test, pattern3test, y_train, y_test
    )
    selfAttentionModel.train()

    # Assuming we have a method to get the self-attention encoded features
    encoded_features_train, encoded_features_test = selfAttentionModel.get_encoded_features()

    # Step 2: Cross-Attention Encoder
    crossAttentionModel = CrossAttentionEncoder(
        encoded_features_train, encoded_features_test, pattern1train, pattern2train, pattern3train,
        pattern1test, pattern2test, pattern3test, y_train, y_test
    )
    crossAttentionModel.train()

    # Assuming we have a method to get the cross-attention encoded features
    # Modify this part according to the actual implementation and outputs of your CrossAttentionEncoder
    cross_encoded_features_train, cross_encoded_features_test = crossAttentionModel.get_encoded_features()

    # Step 3: MLP
    mlpModel = MLP(pattern1train, pattern2train, pattern3train, pattern1test, pattern2test,
               pattern3test, y_train, y_test)
    mlpModel.train()
    mlpModel.test()

if __name__ == "__main__":
    main()