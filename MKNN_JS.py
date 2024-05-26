import numpy as np
from scipy.spatial import distance
from collections import defaultdict

class KNN:
    def __init__(self, k):
        if k % 2 == 0:
            raise ValueError("k must be odd")
        self.k = k
        self.features = None
        self.labels = None
        self.distance_between_data = None
        self.validities = None
        self.weights = None
        self.weight_tests = None
        self.sorted_weight_test = None

    def train(self, data, labels):
        self.features = data
        self.labels = labels

        # 1
        self.distance_between_data = [[{
            'id': f"{i+1}-{j+1}",
            'item': data[i],
            'label': labels[i],
            'item2': data[j],
            'label2': labels[j],
            'distance': self.distance(data[i], data[j])
        } for j in range(len(data))] for i in range(len(data))]

        # 2
        self.validities = [{
            'data': data,
            'validity': np.mean([item['label'] == item['label2'] for item in sorted_distance[:self.k]]),
            'data_sorted_by_distance': sorted_distance[:self.k]
        } for sorted_distance in [sorted(item, key=lambda x: x['distance']) for item in self.distance_between_data]]

        # 3
        self.weights = [[{
            'weight': validity['validity'] / (item['distance'] + 0.5),
            **item
        } for item in item2] for item2, validity in zip(self.distance_between_data, self.validities)]

    def predict(self, data):
        if not self.weights:
            raise ValueError("Please call `train` before `predict`.")

        self.weight_tests = [[{
            'id': f"{i+1}-{j+1}",
            'item': data[i],
            'item2': self.features[j],
            'label2': self.labels[j],
            'weight': self.validities[j]['validity'] / (self.distance(data[i], self.features[j]) + 0.5)
        } for j in range(len(self.features))] for i in range(len(data))]

        self.sorted_weight_test = [sorted(item, key=lambda x: x['weight'], reverse=True) for item in self.weight_tests]

        labels = [{'label': self.majority_vote([item['label2'] for item in sorted_item[:self.k]])} for sorted_item in self.sorted_weight_test]

        return labels

    def get_distance_between_data(self):
        return self.distance_between_data

    def get_validities(self):
        return self.validities

    def get_weights(self):
        return self.weights

    def get_weight_tests(self):
        return self.weight_tests

    def get_k(self):
        return self.k

    def set_k(self, k):
        self.k = k

    def distance(self, a, b):
        return distance.euclidean(a, b)

    def majority_vote(self, labels):
        votes = defaultdict(int)
        for label in labels:
            votes[label] += 1
        sorted_votes = sorted(votes.items(), key=lambda x: x[1], reverse=True)
        return sorted_votes[0][0]

    def confusion_matrix(self, predictions, labels):
        if len(predictions) != len(labels):
            raise ValueError("predictions and labels must have the same length")

        tp = sum(1 for prediction, label in zip(predictions, labels) if prediction == label)
        tn = sum(1 for prediction, label in zip(predictions, labels) if prediction != label)
        fp = sum(1 for prediction, label in zip(predictions, labels) if prediction != label)
        fn = sum(1 for prediction, label in zip(predictions, labels) if prediction != label)

        accuracy = (tp + tn) / (tp + tn + fp + fn)
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        f1_score = 2 * (precision * recall) / (precision + recall)

        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score,
            'TP': tp,
            'TN': tn,
            'FP': fp,
            'FN': fn,
        }