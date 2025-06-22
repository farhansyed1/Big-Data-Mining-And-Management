import sys, os
import re
import json
from main import main
from typing import Dict, List
from argparse import ArgumentParser
import pandas as pd

class TerminalColors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    RESET = '\033[0m'
    
def check_results(type, response, gold_label):
    if type == "rating":
        pattern = r'\d+\.\d+'
        match = re.search(pattern, response)
        if match:
            rating = float(match.group())
            # check if the rating is within the tolerance
            gold_label = float(gold_label)
            if abs(rating - gold_label) <= 0.2:
                return True
            else:
                return False
        else:
            return False
    elif type == "compare":
        restraunt_names = response.split("has a higher rating than")[0].strip()
        if restraunt_names == gold_label:
            return True
        else:
            return False
    elif type == "review":
        return len(response) > 0
    else:
        raise ValueError(f"Unknown type: {type}")

def public_tests():
    queries = [
        "What's the user rating for SKYHY?",
        "Which is the better pick, Olive Garden or SKYHY?",
        "What's the rating of Paradise?",
        "Should I choose Udipi's Upahar or SKYHY?"
    ]
    queries_types = [
        "rating",
        "compare",
        "rating",
        "compare"
    ]
    labels = [3.457, "SKYHY", 3.897, "SKYHY"]
    num_passed = 0
    for i in range(len(queries)):
        response = main(queries[i])
        result = check_results(queries_types[i], response, labels[i])
        if result:
            print(TerminalColors.GREEN + f"Test {i+1} Passed." + TerminalColors.RESET, "Expected: ", labels[i], "Query: ", queries[i])
        else:
            print(TerminalColors.RED + f"Test {i+1} Failed." + TerminalColors.RESET, "Expected: ", labels[i], "Query: ", queries[i])
        num_passed += 1 if result else 0
    print(f"{num_passed}/{len(queries)} Tests Passed")
    
def generate_predictions():
    df = pd.read_csv("predictions.csv")
    queries = df['Query'].tolist()
    responses = []
    for i, query in enumerate(queries):
        print(f"Processing query {i+1}/{len(queries)}")
        response = main(query)
        responses.append(response)
    predictions = pd.DataFrame({
        "Query": queries,
        "Response": responses
    })
    predictions.to_csv("predictions.csv", index=False)
    
def grading():
    predictions = pd.read_csv("predictions.csv")
    gold_labels = pd.read_csv("labels.csv")
    num_passed = 0
    for i in range(len(predictions)):
        response = predictions.iloc[i]['Response']
        label = gold_labels.iloc[i]['label']
        query_type = gold_labels.iloc[i]['category']
        result = check_results(query_type, response, label)
        num_passed += 1 if result else 0
    print(f"{num_passed}/{len(predictions)} Tests Passed")
        

if __name__ == "__main__":
    parser = ArgumentParser(description="Run public tests.")
    parser.add_argument(
        "--generate",
        action="store_true",
        help="Generate predictions.csv file."
    )
    parser.add_argument(
        "--grade",
        action="store_true",
        help="Grade the predictions.csv file."
    )
    args = parser.parse_args()
    
    if args.grade:
        grading()
    else:
        if args.generate:
        # Generate predictions.csv file
            generate_predictions()
        else:
            # Run public tests
            public_tests()