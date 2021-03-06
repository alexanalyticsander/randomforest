import numpy as np 
import ast
from util import entropy, information_gain, partition_classes, best_split

class DecisionTree(object):
    def __init__(self, max_depth):
        # Initializing the tree as an empty dictionary or list, as preferred
        self.tree = {}
        self.max_depth = max_depth
        pass
    	
    def learn(self, X, y, par_node = {}, depth=0):
        # Train the decision tree (self.tree) using the the sample X and labels y

        # Use the function best_split in util.py to get the best split and 
        # data corresponding to left and right child nodes

        # number of zeros
        n0 = 0
        for i in range(len(y)):
            if y[i] == 0: 
                n0 += 1
        # number of ones
        n1 = len(y)-n0
        
        if depth >= self.max_depth:
            if n0 >= n1: return 0
            else: return 1
        
        if n0 == 0:  
            return 1
        
        if n1 == 0: 
            return 0
        
        if len(X[0]) <= 1:
            if n0 >= n1: return 0
            else: return 1
        
        split_attribute, split_point, X_left, X_right, y_left, y_right = best_split(X, y)
        
        if len(y_left) == 0 or len(y_right) == 0:
            if n0 >= n1: 
                return 0
            else: 
                return 1
        else:
            tree = {}
            tree[split_attribute] = [split_point, self.learn(X_left, y_left, depth = depth + 1), self.learn(X_right, y_right, depth = depth + 1)]
            self.tree = tree
            return tree


    def classify(self, record):
        # classify the record using self.tree and return the predicted label

        classifier = self.tree
        while isinstance(classifier, dict):
            split_attribute = list(classifier.keys())[0]
            if record[split_attribute] <= classifier[split_attribute][0]: 
                classifier = classifier[split_attribute][1]
            else: classifier = classifier[split_attribute][2]
        return classifier

