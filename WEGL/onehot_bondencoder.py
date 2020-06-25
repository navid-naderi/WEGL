import torch

def BondEncoderOneHot(edge_attr):
    """
    Encode each of the three bond features using one-hot encoding, and concatenate them together
    Each bond has three features:
        Feature 0 (possible_bond_type_list) has 5 categories.
        Feature 1 (possible_bond_stereo_list) has 6 categories.
        Feature 2 (possible_is_conjugated_list) has 2 categories.
    
    Source: https://github.com/snap-stanford/ogb/blob/master/ogb/utils/features.py
    
    Input: |E| x 3 edge feature matrix
    Output: |E| x (5 + 6 + 2) = |E| x 13 one-hot edge feature matrix
    """
    
    num_feature_categories = [5, 6, 2]
    all_one_hot_features = []
    for col in range(3):
        one_hot_features = torch.nn.functional.one_hot(edge_attr[:, col], num_feature_categories[col])
        all_one_hot_features.append(one_hot_features)
    all_one_hot_features = torch.cat(all_one_hot_features, dim=1)
    
    return all_one_hot_features