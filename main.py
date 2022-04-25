"""
@author : Janhavi Doshi
Detects whether a sentence is English or Dutch
Usage: Refer ReadMe.txt
"""
import math
import sys
import pickle


MAX_TREE_DEPTH = 2
NO_OF_STUMPS = 1


class DTNode:
    def __init__(self):
        self.feature = None
        self.leftChild = None
        self.rightChild = None
        self.adaHypoWt = None


class DecisionTree:
    def __init__(self):
        self.rootNode = DTNode()
        self.treeDepth = None


class AdaBoost:
    def __init__(self):
        self.listStumps = None
        self.noOfStumps = None


class Model:
    def __init__(self):
        self.learning = None
        self.decisionTree = None
        self.adaBoost = None
        self.dt = None


#  Feature 1: Is the 15 word sentence DEVOID OF ANY English Function Word?
def f1(words):
    for word in words:
        englishFunctionWords = {"from", "an", "he", "she", "at", "on", "it", "for", "this", "it", "her",
                                "him", "a", "under", "above", "before", "and", "or", "but", "then", "would", "could",
                                "should", "did", "has", "yes", "no", "okay", "ok", "oh", "ah", "after", "to", "which",
                                "by", "its", "hers", "his", "when", "there", "than", "the"}
        if word in englishFunctionWords:
            return False
    return True


#  Feature 2: Do the 15 words contain a Dutch Function Word?
def f2(word):
    dutchFunctionWords = {"en", "een", "eeuw", "uit", "de", "het", "hij", "ze", "bij", "aan", "voor", "om", "ze", "wat"
                          , "op"}
    if word in dutchFunctionWords:
        return True
    else:
        return False


#  Feature 3: Do the 15 words contain Dutch Vowels and diphthongs?
def f3(word: str):
    if word.rfind("ij") != -1:
        return True
    elif word.rfind("tsj") != -1:
        return True
    elif word.rfind("aaij") != -1:
        return True
    elif word.rfind("aei") != -1:
        return True
    elif word.rfind("ooy") != -1:
        return True
    elif word.rfind("oey") != -1:
        return True
    else:
        return False


#  Feature 4: Do the 15 words contain special characters?
def f4(word):
    letters = {"ë", "ê", "è", "ô"}
    for letter in word:
        if letter in letters:
            return True
    return False


#  Feature 5: Do the 15 words contain at least 2 words with length greater than 10 letters?
def f5(words: list[str]):
    wordsGreaterThan = 0
    for word in words:
        if len(word) >= 12:
            wordsGreaterThan += 1
    if wordsGreaterThan > 1:
        return True
    else:
        return False


def parseData(filename, case):
    inputArray = []
    sentencesArray = []
    with open(filename) as file:
        for line in file:
            sentencesArray.append(line)
            temp = []
            if case == "train" or case == "test":
                firstSplit = line.split("|")
                if firstSplit[0] == "en":
                    temp = [True, False, False, False, False, "E"]
                else:
                    temp = [True, False, False, False, False, "D"]
                words = firstSplit[1].strip().split(" ")
            else:
                temp = [True, False, False, False, False, "Prediction"]
                words = line.strip().split(" ")

            for word in words:
                tempWord = word.strip().lower()
                if (not temp[1]) and f2(tempWord):
                    temp[1] = True
                if (not temp[2]) and f3(tempWord):
                    temp[2] = True
                if (not temp[3]) and f4(tempWord):
                    temp[3] = True
            if (not temp[4]) and f5(words):
                temp[4] = True
            if not f1(words):
                temp[0] = False
            inputArray.append(temp)
    if case == "predict":
        return inputArray, sentencesArray
    return inputArray


def calculateEntropy(inputArray):
    p = 0
    n = 0
    for item in inputArray:
        if item[5] == 'E':
            p += item[6]
        else:
            n += item[6]
    q = p / (p + n)
    if q == 1 or q == 0:
        return 0
    entropy = -1 * ((q * math.log2(q)) + ((1-q) * math.log2(1-q)))
    return entropy


def calculateGain(attributeIndex, inputArray):
    d1 = []
    d2 = []

    for item in inputArray:
        if item[attributeIndex]:
            d1.append(item)
        else:
            d2.append(item)
    if len(d1) != 0:
        entropy_d1 = calculateEntropy(d1)
    else:
        entropy_d1 = 0
    if len(d2) != 0:
        entropy_d2 = calculateEntropy(d2)
    else:
        entropy_d2 = 0

    remainder_d1 = len(d1) / len(inputArray) * entropy_d1
    remainder_d2 = len(d2) / len(inputArray) * entropy_d2

    remainder = remainder_d1 + remainder_d2
    gain = calculateEntropy(inputArray) - remainder
    return gain


def getNode(inputArray, attributesCovered: list):
    gainList = []
    for i in range(0, 5):
        if attributesCovered[i]:
            gainList.append(-10000)
            continue
        gainList.append(calculateGain(i, inputArray))
    maxVal = max(gainList)
    featureToTest = gainList.index(maxVal)

    E = 0
    D = 0

    for item in inputArray:
        if item[0] == 'E':
            E += 1
        else:
            D += 1

    if E > D:
        featureTrue = "E"
        featureFalse = "D"
    else:
        featureTrue = "D"
        featureFalse = "E"

    rootNode = DTNode()
    rootNode.feature = featureToTest
    rootNode.leftChild = featureTrue
    rootNode.rightChild = featureFalse
    return rootNode


def dtAlgoHelper(inputArray, currDepth, attributesCovered):
    rootNode = getNode(inputArray, attributesCovered)

    currDepth += 1
    attributesCovered[rootNode.feature] = True

    if currDepth == MAX_TREE_DEPTH:
        return rootNode
    else:
        inputArrayLeft = []
        inputArrayRight = []
        for item in inputArray:
            if item[rootNode.feature]:
                inputArrayLeft.append(item)
            else:
                inputArrayRight.append(item)
        attributesCoveredLeft = []
        attributesCoveredRight = []
        for item in attributesCovered:
            if item:
                attributesCoveredLeft.append(True)
                attributesCoveredRight.append(True)
            else:
                attributesCoveredLeft.append(False)
                attributesCoveredRight.append(False)
        if len(inputArrayLeft) > 0:
            rootNode.leftChild = dtAlgoHelper(inputArrayLeft, currDepth, attributesCoveredLeft)
        if len(inputArrayRight) > 0:
            rootNode.rightChild = dtAlgoHelper(inputArrayRight, currDepth, attributesCoveredRight)
        return rootNode


def dtAlgo(inputTrainingArray):
    for item in inputTrainingArray:
        item.append(1)
    attributesCovered = [False, False, False, False, False]
    dt = DecisionTree()
    dt.treeDepth = MAX_TREE_DEPTH
    dt.rootNode = dtAlgoHelper(inputTrainingArray, 0, attributesCovered)
    return dt


def adaAlgoHelper(inputTrainingArray, attributesCovered, listOfStumps, noOfStumps):
    node = getNode(inputTrainingArray, attributesCovered)
    attributesCovered[node.feature] = True

    noOfStumps += 1

    errorAdaBoost = 0
    numberOfErrors = 0
    for item in inputTrainingArray:
        if item[node.feature] and item[5] != 'D':
            errorAdaBoost += item[6]
            numberOfErrors += 1
        if item[node.feature] == 'False' and item[5] != 'E':
            errorAdaBoost += item[6]
            numberOfErrors += 1

    hypothesisWeight = math.log((1 - (errorAdaBoost + sys.float_info.epsilon)) / (errorAdaBoost + sys.float_info.epsilon))
    node.adaHypoWt = hypothesisWeight
    listOfStumps.append(node)

    numberOfCorrect = len(inputTrainingArray) - numberOfErrors
    update = errorAdaBoost / (1 - errorAdaBoost)
    for item in inputTrainingArray:
        if item[node.feature] and item[5] == 'D':
            item[6] = item[6] * update
        if item[node.feature] == 'False' and item[5] == 'E':
            item[6] = item[6] * update
    totalWeight = 0
    for item in inputTrainingArray:
        totalWeight += item[6]
    for item in inputTrainingArray:
        item[6] = item[6] / totalWeight

    if noOfStumps < NO_OF_STUMPS:
        listOfStumps = adaAlgoHelper(inputTrainingArray, attributesCovered, listOfStumps, noOfStumps)

    return listOfStumps


def adaAlgo(inputTrainingArray):
    initialWeight = 1 / len(inputTrainingArray)
    for item in inputTrainingArray:
        item.append(initialWeight)

    attributesCovered = [False, False, False, False, False]

    ada = AdaBoost()
    ada.listStumps = adaAlgoHelper(inputTrainingArray, attributesCovered, [], 0)
    ada.noOfStumps = NO_OF_STUMPS

    return ada


def predictLine(rootNode: DTNode, singleInputArray: list):
    isFeatureTrue = singleInputArray[rootNode.feature]
    if isFeatureTrue:
        if rootNode.leftChild == "E":
            return "E"
        elif rootNode.leftChild == "D":
            return "D"
        else:
            prediction = predictLine(rootNode.leftChild, singleInputArray)
    else:
        if rootNode.rightChild == "E":
            return "E"
        elif rootNode.rightChild == "D":
            return "D"
        else:
            prediction = predictLine(rootNode.rightChild, singleInputArray)
    return prediction


def train():
    trainingData = sys.argv[2]
    hypoOutFile = sys.argv[3]
    learningType = sys.argv[4]

    inputTrainingArray = parseData(trainingData, "train")

    trainedModel = Model()
    if learningType == "dt":
        dt = dtAlgo(inputTrainingArray)
        trainedModel.learning = "dt"
        trainedModel.dt = dt
    else:
        ada = adaAlgo(inputTrainingArray)
        trainedModel.learning = "ada"
        trainedModel.adaBoost = ada
    
    file = open(hypoOutFile, "wb")
    pickle.dump(trainedModel, file)


def testDTLine(line, dt):
    prediction = predictLine(dt.rootNode, line)
    if prediction == line[5]:
        return True
    return False


def testADALine(line, ada: AdaBoost):
    predictsE = 0
    predictsD = 0
    for i in range(0, ada.noOfStumps):
        stump = ada.listStumps[i]
        prediction = predictLine(stump, line)
        hypoWt = stump.adaHypoWt
        if prediction == "E":
            predictsE += hypoWt
        if prediction == "D":
            predictsD += hypoWt
    if predictsD > predictsE:
        finalPrediction = "D"
    else:
        finalPrediction = "E"
    if finalPrediction == line[5]:
        return True
    return False


def testDT(inputTestingArray, dt):
    correct = 0
    incorrect = 0
    total = len(inputTestingArray)
    for line in inputTestingArray:
        isCorrect = testDTLine(line, dt)
        if isCorrect:
            correct += 1
        else:
            incorrect += 1
    errorRate = incorrect / total * 100
    print(str(round(errorRate, 2)) + "%")


def testADA(inputTestingArray, ada):
    correct = 0
    incorrect = 0
    total = len(inputTestingArray)
    for line in inputTestingArray:
        isCorrect = testADALine(line, ada)
        if isCorrect:
            correct += 1
        else:
            incorrect += 1
    errorRate = incorrect / total * 100
    print(str(round(errorRate, 2)) + "%")


def test():
    trainingData = sys.argv[2]
    testData = sys.argv[3]
    learningType = sys.argv[4]

    inputTrainingArray = parseData(trainingData, "train")
    inputTestingArray = parseData(testData, "test")

    # Train the data
    if learningType == "dt":
        dt = dtAlgo(inputTrainingArray)
        testDT(inputTestingArray, dt)
    else:
        ada = adaAlgo(inputTrainingArray)
        testADA(inputTestingArray, ada)


def predictDT(dt: DecisionTree, inputPredictArray, sentencesArray):
    for i in range(0, len(inputPredictArray)):
        prediction = predictLine(dt.rootNode, inputPredictArray[i])
        if prediction == "E":
            prediction = "English"
        else:
            prediction = "Dutch"
        print(sentencesArray[i].strip() + " | Prediction: " + prediction)


def predictADA(ada: AdaBoost, inputPredictArray, sentencesArray):
    for j in range(0, len(inputPredictArray)):
        predictsE = 0
        predictsD = 0
        for i in range(0, ada.noOfStumps):
            stump = ada.listStumps[i]
            prediction = predictLine(stump, inputPredictArray[j])
            hypoWt = stump.adaHypoWt
            if prediction == "E":
                predictsE += hypoWt
            if prediction == "D":
                predictsE += hypoWt
        if predictsD > predictsE:
            finalPrediction = "D"
        else:
            finalPrediction = "E"

        if finalPrediction == "E":
            finalPrediction = "English"
        else:
            finalPrediction = "Dutch"
        print(sentencesArray[j].strip() + " | Prediction: " + finalPrediction)


def predict():
    hypoTrained = sys.argv[2]
    fileWordsInput = sys.argv[3]

    inputPredictArray, sentencesArray = parseData(fileWordsInput, "predict")

    file = open(hypoTrained, "rb")
    trainedModel = pickle.load(file)

    if trainedModel.learning == "dt":
        predictDT(trainedModel.dt, inputPredictArray, sentencesArray)
    else:
        predictADA(trainedModel.adaBoost, inputPredictArray, sentencesArray)


def main():
    # Getting Command Line Arguments
    action = sys.argv[1]
    if action == "train":
        train()
    elif action == "test":
        test()
    elif action == "predict":
        predict()


if __name__ == '__main__':
    main()
