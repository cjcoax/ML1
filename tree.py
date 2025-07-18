# decision tree from scratch for banknote dataset
import csv
from math import log2

# load csv
def load_csv(filename):
    with open(filename, 'r') as f:
        return [row for row in csv.reader(f)]

# convert column to float
def str_column_to_float(dataset, column):
    for row in dataset:
        row[column] = float(row[column].strip())

# gini index
def gini_index(groups, classes):
    # count all samples at split point
    n_instances = sum([len(group) for group in groups])
    gini = 0.0
    for group in groups:
        size = len(group)
        if size == 0:
            continue
        score = 0.0
        for class_val in classes:
            p = [row[-1] for row in group].count(class_val) / size
            score += p * p
        gini += (1.0 - score) * (size / n_instances)
    return gini

# entropy
def entropy_index(groups, classes):
    n_instances = sum([len(group) for group in groups])
    ent = 0.0
    for group in groups:
        size = len(group)
        if size == 0:
            continue
        score = 0.0
        for class_val in classes:
            p = [row[-1] for row in group].count(class_val) / size
            if p > 0:
                score += -p * log2(p)
        ent += score * (size / n_instances)
    return ent

# split based on attribute and value

def test_split(index, value, dataset):
    left, right = list(), list()
    for row in dataset:
        if row[index] < value:
            left.append(row)
        else:
            right.append(row)
    return left, right

# get best split using gini

def get_split(dataset):
    class_values = list(set(row[-1] for row in dataset))
    b_index, b_value, b_score, b_groups = 999, 999, 999, None
    for index in range(len(dataset[0])-1):
        for row in dataset:
            groups = test_split(index, row[index], dataset)
            gini = gini_index(groups, class_values)
            if gini < b_score:
                b_index, b_value, b_score, b_groups = index, row[index], gini, groups
    return {'index': b_index, 'value': b_value, 'groups': b_groups}

# get best split using entropy

def get_split_entropy(dataset):
    class_values = list(set(row[-1] for row in dataset))
    b_index, b_value, b_score, b_groups = 999, 999, 999, None
    for index in range(len(dataset[0])-1):
        for row in dataset:
            groups = test_split(index, row[index], dataset)
            ent = entropy_index(groups, class_values)
            if ent < b_score:
                b_index, b_value, b_score, b_groups = index, row[index], ent, groups
    return {'index': b_index, 'value': b_value, 'groups': b_groups}

# to terminal node

def to_terminal(group):
    outcomes = [row[-1] for row in group]
    return max(set(outcomes), key=outcomes.count)

# create child splits

def split(node, max_depth, min_size, depth):
    left, right = node['groups']
    del(node['groups'])
    if not left or not right:
        node['left'] = node['right'] = to_terminal(left + right)
        return
    if depth >= max_depth:
        node['left'], node['right'] = to_terminal(left), to_terminal(right)
        return
    if len(left) <= min_size:
        node['left'] = to_terminal(left)
    else:
        node['left'] = get_split(left)
        split(node['left'], max_depth, min_size, depth+1)
    if len(right) <= min_size:
        node['right'] = to_terminal(right)
    else:
        node['right'] = get_split(right)
        split(node['right'], max_depth, min_size, depth+1)

# build a decision tree

def build_tree(train, max_depth, min_size):
    root = get_split(train)
    split(root, max_depth, min_size, 1)
    return root

# prediction

def predict(node, row):
    if row[node['index']] < node['value']:
        if isinstance(node['left'], dict):
            return predict(node['left'], row)
        else:
            return node['left']
    else:
        if isinstance(node['right'], dict):
            return predict(node['right'], row)
        else:
            return node['right']


def main():
    dataset = load_csv('data_banknote_authentication.csv')
    for i in range(len(dataset[0])):
        str_column_to_float(dataset, i)
    for row in dataset:
        row[-1] = int(row[-1])

    train = dataset[:int(len(dataset)*2/3)]
    test = dataset[int(len(dataset)*2/3):]

    tree = build_tree(train, 5, 10)
    predictions = [predict(tree, row) for row in test]
    actual = [row[-1] for row in test]
    accuracy = sum(1 for i in range(len(actual)) if actual[i]==predictions[i]) / len(actual)
    print('Accuracy: %.3f' % accuracy)

if __name__ == '__main__':
    main()
