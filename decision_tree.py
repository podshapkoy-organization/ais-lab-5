import numpy as np


def build_tree(data, target, depth, max_depth, n_features):
    unique_classes = np.unique(target)
    if len(unique_classes) == 1 or depth >= max_depth:
        return unique_classes[0]
    selected_features = data.sample(n=n_features, axis=1)
    best_feature = None
    best_criterion = -np.inf
    for feature in selected_features.columns:
        values, counts = np.unique(data[feature], return_counts=True)
        feature_entropy = 0
        for value, count in zip(values, counts):
            subset = target[data[feature] == value]
            prob = count / len(data)
            if len(subset) == 0:
                continue
            class_prob = np.bincount(subset) / len(subset)
            entropy = -np.sum(class_prob * np.log2(class_prob + 1e-10))
            feature_entropy += prob * entropy
        criterion = 1 - feature_entropy
        if criterion > best_criterion:
            best_criterion = criterion
            best_feature = feature
    tree = {best_feature: {}}
    best_feature_values = data[best_feature].unique()
    for value in best_feature_values:
        subset_data = data[data[best_feature] == value]
        subset_target = target[subset_data.index]
        if len(subset_target) == 0:
            tree[best_feature][value] = np.bincount(target).argmax()
        else:
            tree[best_feature][value] = build_tree(
                subset_data.drop(columns=[best_feature]),
                subset_target,
                depth + 1,
                max_depth,
                n_features
            )
    return tree


def predict(sample, node, train_target):
    if not isinstance(node, dict):
        return node
    feature = list(node.keys())[0]
    value = sample[feature]
    if value in node[feature]:
        return predict(sample, node[feature][value], train_target)
    else:
        return np.bincount(
            train_target).argmax()
