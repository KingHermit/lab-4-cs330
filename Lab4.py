"""
 Name: Seven Son and Reina Buen
 Assignment: Lab 4 - Decision Tree
 Course: CS 330
 Semester: Spring 2025
 Instructor: Dr. Cao
 Date: 3/20/2025
 Sources consulted: any books, individuals, etc consulted

 Known Bugs: description of known bugs and other program imperfections

 Creativity: anything extra that you added to the lab

 Instructions: After a lot of practice in Python, in this lab, you are going to design the program for decision tree and implement it from scrath! Don't be panic, you still have some reference, actually you are going to translate the JAVA code to Python! The format should be similar to Lab 2!

"""
import sys
import argparse
import math
import pandas as pd
import os


# You may need to define the Tree node and add extra helper functions here

class TreeNode:
    """
    This is the class for Tree Node
    """

    def __init__(self):
        self.attribute = None
        self.children = {}
        self.label = None


def DTtrain(data, model):
    """
    Robust training function that generates complete decision trees
    """
    datamap = {}
    attvalues = {}
    atts = []
    
    with open(data, 'r') as file:
        first_line = file.readline().strip()
        if first_line.startswith('#'):
            first_line = first_line[1:]
        atts = first_line.split('|')
        numAtts = len(atts) - 1  
 
        for a in atts:
            attvalues[a] = set()

        for line in file:
            parts = line.strip().split()
            if not parts:
                continue
                
            dataclass = parts[0]
            datapoint = parts[1:]

            attvalues[atts[0]].add(dataclass)
            for i in range(numAtts):
                attvalues[atts[i+1]].add(datapoint[i])

            if dataclass not in datamap:
                datamap[dataclass] = []
            datamap[dataclass].append(datapoint)

    for a in attvalues:
        attvalues[a] = sorted(attvalues[a])
    
    numClasses = len(datamap)

    def build_tree(node_data, free_attrs):
        class_counts = {cls: len(dpoints) for cls, dpoints in node_data.items()}
        total = sum(class_counts.values())

        # Stopping conditions
        if total == 0:
            node = TreeNode()
            node.label = "undefined"
            return node
            
        if len(class_counts) == 1:
            node = TreeNode()
            node.label = next(iter(class_counts.keys()))
            return node

        if not any(free_attrs):
            node = TreeNode()
            node.label = max(class_counts.items(), key=lambda x: x[1])[0]
            return node

        best_att = None
        best_gain = -1
        best_att_idx = -1
        
        for i, att in enumerate(free_attrs):
            if att is None:
                continue
                
            vals = attvalues[att]
            partition = [[0]*numClasses for _ in range(len(vals))]
            class_idx = {cls: idx for idx, cls in enumerate(attvalues[atts[0]])}
            
            for cls, dpoints in node_data.items():
                for dp in dpoints:
                    val_idx = vals.index(dp[i])
                    partition[val_idx][class_idx[cls]] += 1
            
            part_total = sum(sum(row) for row in partition)
            if part_total == 0:
                continue
                
            total_entropy = 0
            for row in partition:
                row_sum = sum(row)
                if row_sum > 0:
                    total_entropy += (row_sum/part_total) * entropy(row)
            
            gain = entropy(list(class_counts.values())) - total_entropy
            if gain > best_gain:
                best_gain = gain
                best_att = att
                best_att_idx = i

        if best_att is None:
            node = TreeNode()
            node.label = max(class_counts.items(), key=lambda x: x[1])[0]
            return node
            
        node = TreeNode()
        node.attribute = best_att
        free_attrs[best_att_idx] = None
        
        for val in attvalues[best_att]:
            subset = {}
            for cls in node_data:
                subset[cls] = [dp for dp in node_data[cls] if dp[best_att_idx] == val]
            
            if sum(len(d) for d in subset.values()) == 0:
                child = TreeNode()
                child.label = max(class_counts.items(), key=lambda x: x[1])[0]
            else:
                child = build_tree(subset, free_attrs.copy())
            
            node.children[val] = child
            
        return node

    def entropy(counts):
        """Safe entropy calculation"""
        total = sum(counts)
        if total == 0:
            return 0
        return -sum((c/total) * math.log2(c/total) if c != 0 else 0 for c in counts)

    root = build_tree(datamap, atts[1:].copy())

    with open(model, 'w') as f:
        f.write(' '.join(atts[1:]) + '\n')
        
        def write_node(node):
            if node.label is not None:
                f.write(f"[{node.label}]")
                return
            
            f.write(f"{node.attribute} (")
            for val in attvalues[node.attribute]:
                if val in node.children:
                    f.write(f" {val} ")
                    write_node(node.children[val])
                else:
                    f.write(f" {val} [undefined]")
            f.write(" )")
        
        write_node(root)

def DTpredict(data, model, prediction):
    """
    This is the main function to make predictions on the test dataset. It will load saved model file,
    and also load testing data TestDataNoLabel.txt, and apply the trained model to make predictions.
    You should save your predictions in prediction file, each line would be a label, such as:
    1
    0
    0
    1
    ...
    """

    # implement your code here
    def __init__(self):
        self.root = None
        self.att_arr = []
        self.predictions = []

    def read_model(modelFile):
        """Reads the decision tree model from the file"""
        try:
            with open(modelFile, "r") as file:
                atts = file.readline().strip().split()
                att_arr = atts
                root = read_node(file)
                return att_arr, root
        except IOError as e:
            print(f"Error reading model: {e}")
            exit(1)

    def read_node(inFile):
        """Recursively builds tree from the file"""
        # read until next token (handles possible line breaks)
        while True:
            line = inFile.readline()
            print(line)
            if not line:
                print("End of file here 1")
                break  # end of file
            tokens = line.strip().split()
            if not tokens:
                continue  # skip empty lines

            for token in tokens:
                if token.startswith('['):  # build leaf node
                    new = TreeNode()
                    new.label = token[1:-1]
                    return new
                elif token == ')':  # end of current node's children
                    continue
                else:  # internal node
                    # create empty node and set attribute
                    node = TreeNode()
                    node.attribute = token

                    # The next token should be '(' to start children
                    next_token = None
                    while True:
                        if not tokens:  # read next line
                            line = inFile.readline()
                            if not line:
                                break
                            tokens = line.strip().split()
                            if not tokens:
                                continue
                        next_token = tokens.pop(0)
                        if next_token == '(':
                            break

                    # read child nodes until it reaches ')'
                    val = None
                    while True:
                        if not tokens:
                            line = inFile.readline()
                            if not line:
                                print("End of file here 2")
                                break
                            tokens = line.strip().split()
                            if not tokens:
                                continue
                        val = tokens.pop(0)
                        if val == ')':
                            break
                        # value is followed by a child node
                        child_node = read_node(inFile)
                        node.children[val] = child_node

                    return node

        raise ValueError("Unexpected end of file while parsing node")

    def trace_tree(node, data, att_arr):
        """Traverses tree to make prediction for one data instance"""
        if node.label is not None:
            return node.label
        att = node.attribute
        val = data[att_arr.index(att)]
        t = node.children.get(val)
        return trace_tree(t, data, att_arr)

    # Main execution flow- combination of methods predictFromModel and savePredictions.

    try:
        # 1. Load model
        att_arr, root = read_model(model)

        # 2. Read test data and make predictions
        predictions = []
        with open(data, "r") as testfile:
            for line in testfile:
                test_data = line.strip().split()
                if not test_data:
                    continue

                test_data = test_data[1:]   # skip first element; 'consume -1' and take the rest
                if len(test_data) != len(att_arr):
                    raise ValueError("Test data doesn't match model attributes")

                pred = trace_tree(root, data, att_arr)
                predictions.append(pred)

        # 3. Save predictions to file
        with open(prediction, "w") as outfile:
            for pred in predictions:
                outfile.write(f"{pred}\n")

        # 4. Return successful status
        return True

    except Exception as e:
        print(f"Error during prediction: {e}")
        return False


def EvaDT(predictionLabel, realLabel, output):
    """
    This is the main function. You should compare line by line,
     and calculate how many predictions are correct, how many predictions are not correct. The output could be:

    In total, there are ??? predictions. ??? are correct, and ??? are not correct.

    """
    correct, incorrect, length = 0, 0, 0
    with open(predictionLabel, 'r') as file1, open(realLabel, 'r') as file2:
        pred = [line for line in file1]
        real = [line for line in file2]
        length = len(pred)
        for i in range(length):
            if pred.pop(0) == real.pop(0):
                correct += 1
            else:
                incorrect += 1
    Rate = correct / length

    result = "In total, there are " + str(length) + " predictions. " + str(correct) + " are correct and " + str(
        incorrect) + " are incorrect. The percentage is " + str(Rate)
    with open(output, "w") as fh:
        fh.write(result)


def main():
    options = parser.parse_args()
    mode = options.mode  # first get the mode
    print("mode is " + mode)
    if mode == "T":
        """
        The training mode
        """
        inputFile = options.input
        outModel = options.output
        if inputFile == '' or outModel == '':
            showHelper()
        DTtrain(inputFile, outModel)
    elif mode == "P":
        """
        The prediction mode
        """
        inputFile = options.input
        modelPath = options.modelPath
        outPrediction = options.output
        if inputFile == '' or modelPath == '' or outPrediction == '':
            showHelper()
        DTpredict(inputFile, modelPath, outPrediction)
    elif mode == "E":
        """
        The evaluating mode
        """
        predictionLabel = options.input
        trueLabel = options.trueLabel
        outPerf = options.output
        if predictionLabel == '' or trueLabel == '' or outPerf == '':
            showHelper()
        EvaDT(predictionLabel, trueLabel, outPerf)
    pass


def showHelper():
    parser.print_help(sys.stderr)
    print("Please provide input augument. Here are examples:")
    print("python " + sys.argv[0] + " --mode T --input TrainingData.txt --output DTModel.txt")
    print("python " + sys.argv[
        0] + " --mode P --input TestDataNoLabel.txt --modelPath DTModel.txt --output TestDataLabelPrediction.txt")
    print("python " + sys.argv[
        0] + " --mode E --input TestDataLabelPrediction.txt --trueLabel LabelForTest.txt --output Performance.txt")
    sys.exit(0)


if __name__ == "__main__":
    # ------------------------arguments------------------------------#
    # Shows help to the users                                        #
    # ---------------------------------------------------------------#
    parser = argparse.ArgumentParser()
    parser._optionals.title = "Arguments"
    parser.add_argument('--mode', dest='mode',
                        default='',  # default empty!
                        help='Mode: T for training, and P for making predictions, and E for evaluating the machine learning model')
    parser.add_argument('--input', dest='input',
                        default='',  # default empty!
                        help='The input file. For T mode, this is the training data, for P mode, this is the test data without label, for E mode, this is the predicted labels')
    parser.add_argument('--output', dest='output',
                        default='',  # default empty!
                        help='The output file. For T mode, this is the model path, for P mode, this is the prediction result, for E mode, this is the final result of evaluation')
    parser.add_argument('--modelPath', dest='modelPath',
                        default='',  # default empty!
                        help='The path of the machine learning model ')
    parser.add_argument('--trueLabel', dest='trueLabel',
                        default='',  # default empty!
                        help='The path of the correct label ')
    if len(sys.argv) < 3:
        showHelper()
    main()
