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
        y_pred = classifier.predict(Xtest)
        y_pred = y_pred.todense()

        self.assertAlmostEqual(float(params['Accuracy']), float(eb_accuracy(ytest, y_pred)))

    def test_exampleBasedPrecision(self):
        Xtrain, ytrain = readDataFromFile(TRAINDATA_FILE)
        Xtest, ytest = readDataFromFile(TESTDATA_FILE)
        params = readParams(RESULTS_FILE)
        classifier = MLkNN(k=10)
        classifier.fit(Xtrain, ytrain)
        y_pred = classifier.predict(Xtest)
        y_pred = y_pred.todense()
        y_pred = np.asarray(y_pred,dtype=int)

        self.assertAlmostEqual(float(params['Precision']), float(eb_precision(ytest, y_pred)))

    def test_exampleBasedRecall(self):
        Xtrain, ytrain = readDataFromFile(TRAINDATA_FILE)
        Xtest, ytest = readDataFromFile(TESTDATA_FILE)
        params = readParams(RESULTS_FILE)
        classifier = MLkNN(k=10)
        classifier.fit(Xtrain, ytrain)
        y_pred = classifier.predict(Xtest)
        y_pred = y_pred.todense()

        self.assertAlmostEqual(float(params['Recall']), float(eb_recall(ytest, y_pred)))

    def test_exampleBasedFBeta(self):
        Xtrain, ytrain = readDataFromFile(TRAINDATA_FILE)
        Xtest, ytest = readDataFromFile(TESTDATA_FILE)
        params = readParams(RESULTS_FILE)
        classifier = MLkNN(k=10)
        classifier.fit(Xtrain, ytrain)
        y_pred = classifier.predict(Xtest)
        y_pred = y_pred.todense()
        
        self.assertAlmostEqual(float(params['FBeta']), float(eb_fbeta(ytest, y_pred)))
    
    def test_subsetAccuracy(self):
        Xtrain, ytrain = readDataFromFile(TRAINDATA_FILE)
        Xtest, ytest = readDataFromFile(TESTDATA_FILE)
        params = readParams(RESULTS_FILE)
        classifier = MLkNN(k=10)
        classifier.fit(Xtrain, ytrain)
        y_pred = classifier.predict(Xtest)
        y_pred = y_pred.todense()

        self.assertAlmostEqual(float(params['SubsetAccuracy']), float(subsetAccuracy(ytest, y_pred)))

    def test_hammingLoss(self):
        Xtrain, ytrain = readDataFromFile(TRAINDATA_FILE)
        Xtest, ytest = readDataFromFile(TESTDATA_FILE)
        params = readParams(RESULTS_FILE)
        classifier = MLkNN(k=10)
        classifier.fit(Xtrain, ytrain)
        y_pred = classifier.predict(Xtest)
        y_pred = y_pred.todense()

        self.assertAlmostEqual(float(params['HammingLoss']), float(hammingLoss(ytest, y_pred)))

    ##Now the labelBasedRanking
    def test_aucMicro(self):
        Xtrain, ytrain = readDataFromFile(TRAINDATA_FILE)
        Xtest, ytest = readDataFromFile(TESTDATA_FILE)
        params = readParams(RESULTS_FILE)
        classifier = MLkNN(k=10)
        classifier.fit(Xtrain, ytrain)
        probabilities = classifier.predict_proba(Xtest)
        probabilities = probabilities.todense()

        self.assertAlmostEqual(float(params['AUCmicro']), float(aucMicro(ytest, probabilities)))

    def test_aucMacro(self):
        Xtrain, ytrain = readDataFromFile(TRAINDATA_FILE)
        Xtest, ytest = readDataFromFile(TESTDATA_FILE)
        params = readParams(RESULTS_FILE)
        classifier = MLkNN(k=10)
        classifier.fit(Xtrain, ytrain)
        probabilities = classifier.predict_proba(Xtest)
        probabilities = probabilities.todense()

        self.assertAlmostEqual(float(params['AUCmacro']), float(aucMacro(ytest, probabilities)))

    def test_aucInstance(self):
        Xtrain, ytrain = readDataFromFile(TRAINDATA_FILE)
        Xtest, ytest = readDataFromFile(TESTDATA_FILE)
        params = readParams(RESULTS_FILE)
        classifier = MLkNN(k=10)
        classifier.fit(Xtrain, ytrain)
        probabilities = classifier.predict_proba(Xtest)
        probabilities = probabilities.todense()

        self.assertAlmostEqual(float(params['AUCInstance']), float(aucInstance(ytest, probabilities)))

    #And then the tests for the exampleBasedRanking metrics
    def test_oneError(self):
        Xtrain, ytrain = readDataFromFile(TRAINDATA_FILE)
        Xtest, ytest = readDataFromFile(TESTDATA_FILE)
        params = readParams(RESULTS_FILE)
        classifier = MLkNN(k=10)
        classifier.fit(Xtrain, ytrain)
        probabilities = classifier.predict_proba(Xtest)
        probabilities = probabilities.todense()

        self.assertAlmostEqual(float(params['OneError']), float(oneError(ytest,probabilities)))

    def test_coverage(self):
        Xtrain, ytrain = readDataFromFile(TRAINDATA_FILE)
        Xtest, ytest = readDataFromFile(TESTDATA_FILE)
        params = readParams(RESULTS_FILE)
        classifier = MLkNN(k=10)
        classifier.fit(Xtrain, ytrain)
        probabilities = classifier.predict_proba(Xtest)
        probabilities = probabilities.todense()

        self.assertAlmostEqual(float(params['Coverage']), float(coverage(ytest, probabilities)))

    def test_averagePrecision(self):
        Xtrain, ytrain = readDataFromFile(TRAINDATA_FILE)
        Xtest, ytest = readDataFromFile(TESTDATA_FILE)
        params = readParams(RESULTS_FILE)
        classifier = MLkNN(k=10)
        classifier.fit(Xtrain, ytrain)
        probabilities = classifier.predict_proba(Xtest)
        probabilities = probabilities.todense()

        self.assertAlmostEqual(float(params['AveragePrecision']), float(averagePrecision(ytest, probabilities)))

    def test_rankingLoss(self):
        Xtrain, ytrain = readDataFromFile(TRAINDATA_FILE)
        Xtest, ytest = readDataFromFile(TESTDATA_FILE)
        params = readParams(RESULTS_FILE)
        classifier = MLkNN(k=10)
        classifier.fit(Xtrain, ytrain)
        probabilities = classifier.predict_proba(Xtest)
        probabilities = probabilities.todense()
        
        self.assertAlmostEqual(float(params['RankingLoss']), float(rankingLoss(ytest, probabilities)))

    #metrics for labelBasedClassification
    def test_accuracyMicro(self):
        Xtrain, ytrain = readDataFromFile(TRAINDATA_FILE)
        Xtest, ytest = readDataFromFile(TESTDATA_FILE)
        params = readParams(RESULTS_FILE)
        classifier = MLkNN(k=10)
        classifier.fit(Xtrain, ytrain)
        y_pred = classifier.predict(Xtest)
        y_pred = y_pred.todense()

        self.assertAlmostEqual(float(params['AccuracyMicro']), float(accuracyMicro(ytest, y_pred)))
    
    def test_accuracyMacro(self):
        Xtrain, ytrain = readDataFromFile(TRAINDATA_FILE)
        Xtest, ytest = readDataFromFile(TESTDATA_FILE)
        params = readParams(RESULTS_FILE)
        classifier = MLkNN(k=10)
        classifier.fit(Xtrain, ytrain)
        y_pred = classifier.predict(Xtest)
        y_pred = y_pred.todense()

        self.assertAlmostEqual(float(params['AccuracyMacro']), float(accuracyMacro(ytest, y_pred)))

    def test_precisionMicro(self):
        Xtrain, ytrain = readDataFromFile(TRAINDATA_FILE)
        Xtest, ytest = readDataFromFile(TESTDATA_FILE)
        params = readParams(RESULTS_FILE)
        classifier = MLkNN(k=10)
        classifier.fit(Xtrain, ytrain)
        y_pred = classifier.predict(Xtest)
        y_pred = y_pred.todense()

        self.assertAlmostEqual(float(params['PrecisionMicro']), float(precisionMicro(ytest, y_pred)))
    
    def test_precisionMacro(self):
        Xtrain, ytrain = readDataFromFile(TRAINDATA_FILE)
        Xtest, ytest = readDataFromFile(TESTDATA_FILE)
        params = readParams(RESULTS_FILE)
        classifier = MLkNN(k=10)
        classifier.fit(Xtrain, ytrain)
        y_pred = classifier.predict(Xtest)
        y_pred = y_pred.todense()
        y_pred = np.asarray(y_pred)

        self.assertAlmostEqual(float(params['PrecisionMacro']), float(precisionMacro(ytest, y_pred)))
    
    def test_recallMicro(self):
        Xtrain, ytrain = readDataFromFile(TRAINDATA_FILE)
        Xtest, ytest = readDataFromFile(TESTDATA_FILE)
        params = readParams(RESULTS_FILE)
        classifier = MLkNN(k=10)
        classifier.fit(Xtrain, ytrain)
        y_pred = classifier.predict(Xtest)
        y_pred = y_pred.todense()

        self.assertAlmostEqual(float(params['RecallMicro']), float(recallMicro(ytest, y_pred)))

    def test_recallMacro(self):
        Xtrain, ytrain = readDataFromFile(TRAINDATA_FILE)
        Xtest, ytest = readDataFromFile(TESTDATA_FILE)
        params = readParams(RESULTS_FILE)
        classifier = MLkNN(k=10)
        classifier.fit(Xtrain, ytrain)
        y_pred = classifier.predict(Xtest)
        y_pred = y_pred.todense()

        self.assertAlmostEqual(float(params['RecallMacro']), float(recallMacro(ytest, y_pred)))

    def test_fbetaMicro(self):
        Xtrain, ytrain = readDataFromFile(TRAINDATA_FILE)
        Xtest, ytest = readDataFromFile(TESTDATA_FILE)
        params = readParams(RESULTS_FILE)
        classifier = MLkNN(k=10)
        classifier.fit(Xtrain, ytrain)
        y_pred = classifier.predict(Xtest)
        y_pred = y_pred.todense()

        self.assertAlmostEqual(float(params['FBetaMicro']), float(fbetaMicro(ytest, y_pred)))

    def test_fbetaMacro(self):
        Xtrain, ytrain = readDataFromFile(TRAINDATA_FILE)
        Xtest, ytest = readDataFromFile(TESTDATA_FILE)
        params = readParams(RESULTS_FILE)
        classifier = MLkNN(k=10)
        classifier.fit(Xtrain, ytrain)
        y_pred = classifier.predict(Xtest)
        y_pred = y_pred.todense()

        self.assertAlmostEqual(float(params['FBetaMacro']), float(fbetaMacro(ytest, y_pred)))