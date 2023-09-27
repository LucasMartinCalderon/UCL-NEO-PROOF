import re
import os
import numpy as np
import torch
from MLP_layer import MLP

# Define regex patterns for identifying potential unsafe delegatecall vulnerabilities
patterns = {
    "direct_delegatecall": re.compile(r"delegatecall\("),  # Direct usage of delegatecall
    "msg_data_delegatecall": re.compile(r"delegatecall\(\s*msg\.data"),  # Usage of delegatecall with msg.data
    "user_controlled_data_delegatecall": re.compile(r"delegatecall\(\s*[a-zA-Z0-9_]+\s*\)")  # delegatecall with user-controlled parameters
}

def split_function(filepath):
    """Splits the Solidity contract into individual functions for easier pattern matching."""
    function_list = []
    with open(filepath, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    flag = -1

    for line in lines:
        text = line.strip()
        if text and text != "\n":
            # Check if line starts a new function or constructor or fallback
            if text.split()[0] in ["function", "constructor", "fallback"]:
                function_list.append([text])
                flag += 1
            # Continue appending lines to the current function/constructor/fallback block
            elif function_list and ("function" in function_list[flag][0] or "constructor" in function_list[flag][0] or "fallback" in function_list[flag][0]):
                function_list[flag].append(text)
    return function_list

def extract_pattern(filepath):
    """Scans the contract to identify any of the specified patterns."""
    allFunctionList = split_function(filepath)
    pattern_detected = {key: 0 for key in patterns.keys()}

    for function in allFunctionList:
        function_str = ' '.join(function)
        for key, pattern in patterns.items():
            if pattern.search(function_str):
                pattern_detected[key] = 1  # Mark pattern as detected

    return list(pattern_detected.values())

def extract_feature_with_fc(outputPathFC, pattern1, pattern2, pattern3):
    """Transforms the detected patterns into a fixed-size feature using the MLP model."""
    pattern1 = torch.Tensor(pattern1)
    pattern2 = torch.Tensor(pattern2)
    pattern3 = torch.Tensor(pattern3)
    model = MLP(4, 100, 250)

    pattern1FC = model(pattern1).detach().numpy().tolist()
    pattern2FC = model(pattern2).detach().numpy().tolist()
    pattern3FC = model(pattern3).detach().numpy().tolist()
    pattern_final = np.array([pattern1FC, pattern2FC, pattern3FC])

    np.savetxt(outputPathFC, pattern_final, fmt="%.6f")

if __name__ == "__main__":
    # Paths for input files and output directories
    inputFileDir = "../data_example/unsafe_delegatecall/source_code/"
    outputfeatureDir = "../pattern_feature/feature_zeropadding/unsafe_delegatecall/"
    outputfeatureFCDir = "../pattern_feature/feature_FNN/unsafe_delegatecall/"
    outputlabelDir = "../pattern_feature/label_by_extractor/unsafe_delegatecall/"
    dirs = os.listdir(inputFileDir)
    
    for file in dirs:
        # Initial pattern values
        pattern1 = [1, 0, 0]
        pattern2 = [0, 1, 0]
        pattern3 = [0, 0, 1]

        print(file)
        inputFilePath = inputFileDir + file
        name = file.split(".")[0]
        pattern_list = extract_pattern(inputFilePath)

        label = 1 if any(pattern_list) else 0  # Label as vulnerable if any pattern is detected

        pattern1.append(pattern_list[0])
        pattern2.append(pattern_list[1])
        pattern3.append(pattern_list[2])

        outputPathFC = outputfeatureFCDir + name + ".txt"
        extract_feature_with_fc(outputPathFC, pattern1, pattern2, pattern3)

        pattern1 = np.array(np.pad(pattern1, (0, 246), 'constant'))
        pattern2 = np.array(np.pad(pattern2, (0, 246), 'constant'))
        pattern3 = np.array(np.pad(pattern3, (0, 246), 'constant'))

        pattern_final = np.array([pattern1, pattern2, pattern3])
        outputPath = outputfeatureDir + name + ".txt"
        np.savetxt(outputPath, pattern_final, fmt="%.6f")

        # Overwrite the label in the file instead of appending
        outputlabelPath = outputlabelDir + file
        with open(outputlabelPath, 'w') as f_outlabel:  # 'w' mode to overwrite the content
            f_outlabel.write(str(label))
