import json, sys
from tree import build_tree, predict, gini_index, entropy_index, test_split, get_split, get_split_entropy, to_terminal, split

filename = sys.argv[1]
with open(filename, 'r', encoding='utf-8') as f:
    nb = json.load(f)

cells = nb['cells']

# Helper to find cell by starting text

def find_cell_index(start_text):
    for i,c in enumerate(cells):
        if c['cell_type']=='code' and c['source'] and c['source'][0].startswith(start_text):
            return i
    return None

# 1. gini_index cell
idx = find_cell_index('# Calculate the Gini index')
code = '''# Calculate the Gini index for a split dataset
def gini_index(groups, classes):
    # count all samples at split point
    n_instances = float(sum([len(group) for group in groups]))
    # sum weighted Gini index for each group
    gini = 0.0
    for group in groups:
        size = float(len(group))
        # avoid divide by zero
        if size == 0:
            continue
        score = 0.0
        # score the group based on the score for each class
        for class_val in classes:
            p = [row[-1] for row in group].count(class_val) / size
            score += p * p
        # weight the group score by its relative size
        gini += (1.0 - score) * (size / n_instances)
    return gini
'''
cells[idx]['source'] = [line+'\n' for line in code.strip().split('\n')]

# 2. test_split cell
idx = find_cell_index('# Split a dataset')
code = '''# Split a dataset based on an attribute and an attribute value
def test_split(index, value, dataset):
    left, right = list(), list()
    """
    TODO: This function loops over each row and checks if the row belongs to the right or left list.
    """
    for row in dataset:
        if row[index] < value:
            left.append(row)
        else:
            right.append(row)
    return left, right
'''
cells[idx]['source'] = [line+'\n' for line in code.strip().split('\n')]

# 3. get_split
idx = find_cell_index('def get_split(')
code = '''def get_split(dataset):
    """Select the best split point for a dataset"""
    class_values = list(set(row[-1] for row in dataset))
    b_index, b_value, b_score, b_groups = 999, 999, 999, None
    for index in range(len(dataset[0]) - 1):
        for row in dataset:
            groups = test_split(index, row[index], dataset)
            gini = gini_index(groups, class_values)
            if gini < b_score:
                b_index, b_value, b_score, b_groups = index, row[index], gini, groups
    return {'index': b_index, 'value': b_value, 'groups': b_groups}
'''
cells[idx]['source'] = [line+'\n' for line in code.strip().split('\n')]

# 4. get_split_entropy
idx = find_cell_index('def get_split_entropy')
code = '''def get_split_entropy(dataset):
    """Select the best split point for a dataset using entropy"""
    class_values = list(set(row[-1] for row in dataset))
    b_index, b_value, b_score, b_groups = 999, 999, 999, None
    for index in range(len(dataset[0]) - 1):
        for row in dataset:
            groups = test_split(index, row[index], dataset)
            entropy = entropy_index(groups, class_values)
            if entropy < b_score:
                b_index, b_value, b_score, b_groups = index, row[index], entropy, groups
    return {'index': b_index, 'value': b_value, 'groups': b_groups}
'''
cells[idx]['source'] = [line+'\n' for line in code.strip().split('\n')]

# 5. to_terminal
idx = find_cell_index('def to_terminal(')
code = '''def to_terminal(group):
    """Determine the most common output within each group"""
    outcomes = [row[-1] for row in group]
    return max(set(outcomes), key=outcomes.count)
'''
cells[idx]['source'] = [line+'\n' for line in code.strip().split('\n')]

# 6. split
idx = find_cell_index('# Create child splits')
code = '''# Create child splits for a node or make terminal
def split(node, max_depth, min_size, depth):
    left, right = node['groups']
    del(node['groups'])
    # check for a no split
    if not left or not right:
        node['left'] = node['right'] = to_terminal(left + right)
        return
    # check for max depth
    if depth >= max_depth:
        node['left'], node['right'] = to_terminal(left), to_terminal(right)
        return
    # process left child
    if len(left) <= min_size:
        node['left'] = to_terminal(left)
    else:
        node['left'] = get_split(left)
        split(node['left'], max_depth, min_size, depth + 1)
    # process right child
    if len(right) <= min_size:
        node['right'] = to_terminal(right)
    else:
        node['right'] = get_split(right)
        split(node['right'], max_depth, min_size, depth + 1)
'''
cells[idx]['source'] = [line+'\n' for line in code.strip().split('\n')]

# 7. build_tree
idx = find_cell_index('# Build a decision tree')
code = '''# Build a decision tree
def build_tree(train, max_depth, min_size):
    """get the first split, and then split starting from the root"""
    root = get_split(train)
    split(root, max_depth, min_size, 1)
    return root
'''
cells[idx]['source'] = [line+'\n' for line in code.strip().split('\n')]

# 8. predict
idx = find_cell_index('# Make a prediction')
code = '''# Make a prediction with a decision tree
def predict(node, row):
    # check if a row belongs to a node and recursively traverse the tree if needed
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
'''
cells[idx]['source'] = [line+'\n' for line in code.strip().split('\n')]

# 9. load_csv and str_column_to_float cell
idx = find_cell_index('from random import seed')
code = '''from random import seed
from random import randrange
from csv import reader

# Load a CSV file
def load_csv(filename):
    file = open(filename, 'r')
    lines = reader(file)
    dataset = list(lines)
    file.close()
    return dataset
 
# Convert string column to float
def str_column_to_float(dataset, column):
    for row in dataset:
        row[column] = float(row[column].strip())

filename = 'data_banknote_authentication.csv'
dataset = load_csv(filename)
# convert string attributes to integers
for i in range(len(dataset[0])):
    str_column_to_float(dataset, i)
for row in dataset:
    row[-1] = int(row[-1])
    
train = dataset[1:int(len(dataset)*2/3)]
test = dataset[int(len(dataset)*2/3)+1:len(dataset)]
'''
cells[idx]['source'] = [line+'\n' for line in code.strip().split('\n')]

# 10. build tree and evaluate accuracy cell
idx = find_cell_index('#TODO Build a tree')
code = '''# Build a tree and evaluate accuracy
tree = build_tree(train, 5, 10)
predictions = list()
for row in test:
    prediction = predict(tree, row)
    predictions.append(prediction)

correct = sum(1 for i,row in enumerate(test) if row[-1]==predictions[i])
print('Accuracy: %s' % (correct/len(test)))
'''
cells[idx]['source'] = [line+'\n' for line in code.strip().split('\n')]

# 11. Add output to accuracy cell
from io import StringIO
import contextlib

f = StringIO()
with contextlib.redirect_stdout(f):
    dataset = [list(map(float,x.strip().split(','))) for x in open('data_banknote_authentication.csv')]
    for row in dataset:
        row[-1] = int(row[-1])
    train = dataset[1:int(len(dataset)*2/3)]
    test = dataset[int(len(dataset)*2/3)+1:len(dataset)]
    tree = build_tree(train, 5, 10)
    predictions = [predict(tree,row) for row in test]
    correct = sum(1 for i,row in enumerate(test) if row[-1]==predictions[i])
    print('Accuracy: %s' % (correct/len(test)))
output_text = f.getvalue()

cells[idx]['outputs'] = [{
    'output_type': 'stream',
    'name': 'stdout',
    'text': [output_text]
}]

nb['cells'] = cells
with open(filename, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)
