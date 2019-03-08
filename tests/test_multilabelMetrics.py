import pytest
import numpy as np
import unittest
import os
from multilabelMetrics.exampleBasedClassification import eb_accuracy, eb_fbeta, eb_precision, eb_recall, subsetAccuracy, hammingLoss
from multilabelMetrics.labelBasedRanking import aucMicro, aucMacro, aucInstance
from multilabelMetrics.exampleBasedRanking import oneError, coverage, rankingLoss, averagePrecision
from multilabelMetrics.labelBasedClassification import accuracyMicro, accuracyMacro, precisionMicro, precisionMacro, recallMacro, recallMicro, fbetaMicro, fbetaMacro
from skmultilearn.adapt import MLkNN
from .auxiliaryFunctions import readParams, readDataFromFile

TRAINDATA_FILE = os.path.join(os.path.dirname(__file__), 'emotions0.train')
TESTDATA_FILE = os.path.join(os.path.dirname(__file__), 'emotions0.gen')
RESULTS_FILE = os.path.join(os.path.dirname(__file__), 'results.txt')

class multilabelMetricsTest(unittest.TestCase):
    #We will test all the measures here agains a proper file that we have
    #For testing this measure and next iterations and commits, we will test the measures agains some measures that we already know are ok
    #From the emotions dataset

    def test_exampleBasedAccuracy(self):
        Xtrain, ytrain = readDataFromFile(TRAINDATA_FILE)
        Xtest, ytest = readDataFromFile(TESTDATA_FILE)
        params = readParams(RESULTS_FILE)
        classifier = MLkNN(k=10)
        classifier.fit(Xtrain, ytrain)
        predictions = classifier.predict(Xtest)

        self.assertAlmostEqual(params['Accuracy'], eb_accuracy(ytest, predictions))

    def test_exampleBasedPrecision(self):
        Xtrain, ytrain = readDataFromFile(TRAINDATA_FILE)
        Xtest, ytest = readDataFromFile(TESTDATA_FILE)
        params = readParams(RESULTS_FILE)
        classifier = MLkNN(k=10)
        classifier.fit(Xtrain, ytrain)
        predictions = classifier.predict(Xtest)

        self.assertAlmostEqual(params['Precision'], eb_precision(ytest, predictions))

    def test_exampleBasedRecall(self):
        Xtrain, ytrain = readDataFromFile(TRAINDATA_FILE)
        Xtest, ytest = readDataFromFile(TESTDATA_FILE)
        params = readParams(RESULTS_FILE)
        classifier = MLkNN(k=10)
        classifier.fit(Xtrain, ytrain)
        predictions = classifier.predict(Xtest)

        self.assertAlmostEqual(params['Recall'], eb_recall(ytest, predictions))

    def test_exampleBasedFBeta(self):
        Xtrain, ytrain = readDataFromFile(TRAINDATA_FILE)
        Xtest, ytest = readDataFromFile(TESTDATA_FILE)
        params = readParams(RESULTS_FILE)
        classifier = MLkNN(k=10)
        classifier.fit(Xtrain, ytrain)
        predictions = classifier.predict(Xtest)

        self.assertAlmostEqual(params['FBeta'], eb_fbeta(ytest, predictions))
    
    def test_subsetAccuracy(self):
        Xtrain, ytrain = readDataFromFile(TRAINDATA_FILE)
        Xtest, ytest = readDataFromFile(TESTDATA_FILE)
        params = readParams(RESULTS_FILE)
        classifier = MLkNN(k=10)
        classifier.fit(Xtrain, ytrain)
        predictions = classifier.predict(Xtest)

        self.assertAlmostEqual(params['SubsetAccuracy'], subsetAccuracy(ytest, predictions))

    def test_hammingLoss(self):
        Xtrain, ytrain = readDataFromFile(TRAINDATA_FILE)
        Xtest, ytest = readDataFromFile(TESTDATA_FILE)
        params = readParams(RESULTS_FILE)
        classifier = MLkNN(k=10)
        classifier.fit(Xtrain, ytrain)
        predictions = classifier.predict(Xtest)

        self.assertAlmostEqual(params['HammingLoss'], hammingLoss(ytest, predictions))


    ##Now the labelBasedRanking
    def test_aucMicro(self):
        Xtrain, ytrain = readDataFromFile(TRAINDATA_FILE)
        Xtest, ytest = readDataFromFile(TESTDATA_FILE)
        params = readParams(RESULTS_FILE)
        classifier = MLkNN(k=10)
        classifier.fit(Xtrain, ytrain)
        probabilities = classifier.predict_proba(Xtest)

        self.assertAlmostEqual(params['aucMicro'], aucMicro(ytest, probabilities))

    def test_aucMacro(self):
        Xtrain, ytrain = readDataFromFile(TRAINDATA_FILE)
        Xtest, ytest = readDataFromFile(TESTDATA_FILE)
        params = readParams(RESULTS_FILE)
        classifier = MLkNN(k=10)
        classifier.fit(Xtrain, ytrain)
        probabilities = classifier.predict_proba(Xtest)

        self.assertAlmostEqual(params['aucMacro'], aucMacro(ytest, probabilities))

    def test_aucInstance(self):
        Xtrain, ytrain = readDataFromFile(TRAINDATA_FILE)
        Xtest, ytest = readDataFromFile(TESTDATA_FILE)
        params = readParams(RESULTS_FILE)
        classifier = MLkNN(k=10)
        classifier.fit(Xtrain, ytrain)
        probabilities = classifier.predict_proba(Xtest)

        self.assertAlmostEqual(params['aucInstance'], aucInstance(ytest, probabilities))

    #And then the tests for the exampleBasedRanking metrics
    def test_oneError(self):
        Xtrain, ytrain = readDataFromFile(TRAINDATA_FILE)
        Xtest, ytest = readDataFromFile(TESTDATA_FILE)
        params = readParams(RESULTS_FILE)
        classifier = MLkNN(k=10)
        classifier.fit(Xtrain, ytrain)
        probabilities = classifier.predict_proba(Xtest)

        self.assertAlmostEqual(params['oneError'], oneError(ytest, probabilities))

    def test_coverage(self):
        Xtrain, ytrain = readDataFromFile(TRAINDATA_FILE)
        Xtest, ytest = readDataFromFile(TESTDATA_FILE)
        params = readParams(RESULTS_FILE)
        classifier = MLkNN(k=10)
        classifier.fit(Xtrain, ytrain)
        probabilities = classifier.predict_proba(Xtest)

        self.assertAlmostEqual(params['Coverage'], coverage(ytest, probabilities))

    def test_averagePrecision(self):
        Xtrain, ytrain = readDataFromFile(TRAINDATA_FILE)
        Xtest, ytest = readDataFromFile(TESTDATA_FILE)
        params = readParams(RESULTS_FILE)
        classifier = MLkNN(k=10)
        classifier.fit(Xtrain, ytrain)
        probabilities = classifier.predict_proba(Xtest)

        self.assertAlmostEqual(params['AveragePrecision'], averagePrecision(ytest, probabilities))

    def test_rankingLoss(self):
        Xtrain, ytrain = readDataFromFile(TRAINDATA_FILE)
        Xtest, ytest = readDataFromFile(TESTDATA_FILE)
        params = readParams(RESULTS_FILE)
        classifier = MLkNN(k=10)
        classifier.fit(Xtrain, ytrain)
        probabilities = classifier.predict_proba(Xtest)

        self.assertAlmostEqual(params['RankingLoss'], rankingLoss(ytest, probabilities))

    #metrics for labelBasedClassification
    def test_accuracyMicro(self):
        Xtrain, ytrain = readDataFromFile(TRAINDATA_FILE)
        Xtest, ytest = readDataFromFile(TESTDATA_FILE)
        params = readParams(RESULTS_FILE)
        classifier = MLkNN(k=10)
        classifier.fit(Xtrain, ytrain)
        predictions = classifier.predict(Xtest)

        self.assertAlmostEqual(params['AccuracyMicro'], accuracyMicro(ytest, predictions))
    
    def test_accuracyMacro(self):
        Xtrain, ytrain = readDataFromFile(TRAINDATA_FILE)
        Xtest, ytest = readDataFromFile(TESTDATA_FILE)
        params = readParams(RESULTS_FILE)
        classifier = MLkNN(k=10)
        classifier.fit(Xtrain, ytrain)
        predictions = classifier.predict(Xtest)

        self.assertAlmostEqual(params['AccuracyMacro'], accuracyMacro(ytest, predictions))

    def test_precisionMicro(self):
        Xtrain, ytrain = readDataFromFile(TRAINDATA_FILE)
        Xtest, ytest = readDataFromFile(TESTDATA_FILE)
        params = readParams(RESULTS_FILE)
        classifier = MLkNN(k=10)
        classifier.fit(Xtrain, ytrain)
        predictions = classifier.predict(Xtest)

        self.assertAlmostEqual(params['PrecisionMicro'], precisionMicro(ytest, predictions))
    
    def test_precisionMacro(self):
        Xtrain, ytrain = readDataFromFile(TRAINDATA_FILE)
        Xtest, ytest = readDataFromFile(TESTDATA_FILE)
        params = readParams(RESULTS_FILE)
        classifier = MLkNN(k=10)
        classifier.fit(Xtrain, ytrain)
        predictions = classifier.predict(Xtest)

        self.assertAlmostEqual(params['PrecisionMacro'], precisionMacro(ytest, predictions))
    
    def test_recallMicro(self):
        Xtrain, ytrain = readDataFromFile(TRAINDATA_FILE)
        Xtest, ytest = readDataFromFile(TESTDATA_FILE)
        params = readParams(RESULTS_FILE)
        classifier = MLkNN(k=10)
        classifier.fit(Xtrain, ytrain)
        predictions = classifier.predict(Xtest)

        self.assertAlmostEqual(params['RecallMicro'], recallMicro(ytest, predictions))

    def test_recallMacro(self):
        Xtrain, ytrain = readDataFromFile(TRAINDATA_FILE)
        Xtest, ytest = readDataFromFile(TESTDATA_FILE)
        params = readParams(RESULTS_FILE)
        classifier = MLkNN(k=10)
        classifier.fit(Xtrain, ytrain)
        predictions = classifier.predict(Xtest)

        self.assertAlmostEqual(params['RecallMacro'], recallMacro(ytest, predictions))

    def test_fbetaMicro(self):
        Xtrain, ytrain = readDataFromFile(TRAINDATA_FILE)
        Xtest, ytest = readDataFromFile(TESTDATA_FILE)
        params = readParams(RESULTS_FILE)
        classifier = MLkNN(k=10)
        classifier.fit(Xtrain, ytrain)
        predictions = classifier.predict(Xtest)

        self.assertAlmostEqual(params['FBetaMicro'], fbetaMicro(ytest, predictions))

    def test_fbetaMacro(self):
        Xtrain, ytrain = readDataFromFile(TRAINDATA_FILE)
        Xtest, ytest = readDataFromFile(TESTDATA_FILE)
        params = readParams(RESULTS_FILE)
        classifier = MLkNN(k=10)
        classifier.fit(Xtrain, ytrain)
        predictions = classifier.predict(Xtest)

        self.assertAlmostEqual(params['FBetaMacro'], fbetaMacro(ytest, predictions))