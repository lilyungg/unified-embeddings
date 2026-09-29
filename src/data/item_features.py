def gather_item_features(item_idx, catalog_features):
    features = {"item_idx": item_idx}
    for name, values in catalog_features.items():
        features[name] = values[item_idx]
    return features
