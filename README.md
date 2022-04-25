# Language-classifier
Detects whether a sentence is English or Dutch using decision trees and ada boost.

# Usage

## Training:

Usage: python3 main.py train <examples> <hypothesisOut> <learning-type>
(To change Max Depth and Number of Stumps, variables at the beginning of the code MAX_TREE_DEPTH = 2
NO_OF_STUMPS = 1 need to be changed)

Optimal eg: python3 main.py train trainVeryBig.txt trainedModelDt dt

Output: trainedModel serialized object created


## Testing for Training Purposes:

Usage: python3 main.py test <trainingData> <testData> <learningType>

Optimal eg: python3 main.py test trainVeryBig.txt testBig.txt dt

Output: Prints Error Rate


## Predicting:

Usage: python3 main.py predict trainedModelDt <file>

Optimal eg: python3 main.py predict trainedModelDt predict.txt

Output: Prints original sentences followed by Prediction (English or Dutch)
