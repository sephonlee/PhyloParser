# Main program to train model
import sys
import os
import cv2 as cv
import time, datetime
import pickle
import csv
import numpy as np
sys.path.append("..")



import matplotlib.pyplot as plt

from sklearn.externals import joblib
from sklearn import datasets

from sklearn import svm
from sklearn import cross_validation
from sklearn import grid_search
from sklearn import metrics
from sklearn.preprocessing import LabelBinarizer
from sklearn.cross_validation import KFold

from sklearn import cross_validation
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import (RandomTreesEmbedding, RandomForestClassifier,
                              GradientBoostingClassifier)

from sklearn.multiclass import OneVsRestClassifier

from sklearn.metrics import roc_curve
# from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
import arff


class Common():
    
    @staticmethod
    def makeDir(dst, dirName):
        path = os.path.join(dst, dirName)        
        try:
            os.makedirs(path)
            print "Create new directory " + path
            return path    
        except OSError as exc: # Python >2.5
            if exc.errno == errno.EEXIST and os.path.isdir(path):
                print  path + ' is existed.'
                pass
            else: raise
            return path
        
    @staticmethod
    def getModelPath(path, dirName):
        dirName = dirName + datetime.datetime.now().strftime("%Y-%m-%d")
        return Common.makeDir(path, dirName)
    
    
    @staticmethod
    def getFileNameAndSuffix(filePath):
        filename = filePath.split('/')[-1]
        suffix = filename.split('.')[-1]
        return filename, suffix
    
        # File format: paper_id_image_id.{png, jpg,....}
    @ staticmethod
    def getIDsFromFilename(filename):
        filename = filename.split('.')
        filename = filename[0].split('_')
        return filename[1], filename[3]
    
    @ staticmethod
    def saveCSV(path, filename, content = None, header = None, mode = 'wb', consoleOut = True):

        if consoleOut:
            print 'Saving image information...'
        filePath = os.path.join(path, filename) + '.csv'
        with open(filePath, mode) as outcsv:
            writer = csv.writer(outcsv, dialect='excel')
            if header is not None:
                writer.writerow(header)
            if content is not None:
                for c in content:
                    writer.writerow(c)
        
        if consoleOut:  
            print filename, 'were saved in', filePath, '\n'
    
    
    @ staticmethod
    def saveArff(path, filename, X, y):
        
        data = X.tolist()
        for i, row in enumerate(data):
            row.append(str(y[i]))
            
        attributes = ['centroid_%d'%(i+1) for i in range(X.shape[1])]
        attributes.append('class_name')
        outFilePath= os.path.join(path, filename)
        
        infile = open(outFilePath, 'wb')
        arff.dump(outFilePath, data, relation="whatever", names=attributes)
        
        print '.arff file saved in %s'%outFilePath
        
        




def binarizeLabel(y, pos_label):
    new_y = []
    for i, label in enumerate(y):
        if label == pos_label:
            new_y.append(1)
        else:
            new_y.append(0)
    return new_y


def getTenFoldConfusionMatrix(model, X, Y, path = None):
    

    k_fold = KFold(n=X.shape[0], shuffle = True, n_folds = 10)
    count = 0
    
    all_y_test = None
    all_y_pred = None
    for train_indices, test_indices in k_fold:
        print "Computing fold No. %d" %count
        count += 1
               
        x_train = X[train_indices]
        y_train = Y[train_indices]
        
        x_test = X[test_indices]
        y_test = Y[test_indices]
        
        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)
        
        if count == 1:
            all_y_pred = y_pred
            all_y_test = y_test
        else:
            all_y_pred = np.hstack([all_y_pred, y_pred])
            all_y_test = np.hstack([all_y_test, y_test])
                      
    print "10-fold testing data report:"
    print metrics.classification_report(all_y_test, all_y_pred), '\n'
           

    print 'Model has not been trained.\n'
            
def trainModel(X, y, outModelPath = None):
    
#     if outModelPath is None:
#         outModelPath = self.Opt.modelPath
        
    print 'Training Model...'
    startTime = time.time()

#     lb = LabelBinarizer()
#     y = lb.fit_transform(y)
    
    # Split into training and test set (e.g., 75/25)
    X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.2, random_state=0)
    
#     print "y_train", y_train
#     print "y_train_", binarizeLabel(y_train, "scheme")
    # Choose estimator
#     self.estimator = svm.SVC(probability = True)
#         self.estimator = svm.SVC(kernel = 'linear', probability = True)

    names = {"SVM_RBF" : "SVM (RBF Kernel)", 
             "SVM_Linear" : "SVM (Linear Kernel)",
             "SVM_Poly" : "SVM (Poly Kernel)",
             "RF" : "Random Forest", 
             "DT" : "CART", 
             "LR" : "Logistic Regression", 
             "KNN" : "K-nearest Neighbors" }
    
    models = {"SVM_RBF" : svm.SVC(kernel = "rbf", gamma = 1e-3, C=1000, probability = True),
              "SVM_Linear" : svm.SVC(kernel = "linear", gamma = 1e-3, C=1000, probability = True),
              "SVM_Poly" : svm.SVC(kernel = "poly", gamma = 1e-3, C=1000, probability = True),
            "RF" : RandomForestClassifier(),
            "DT" : DecisionTreeClassifier(),
            "LR" : LogisticRegression(),
            "KNN" : KNeighborsClassifier()
              }
    
    ROC_keys = {"Equation_ROC" : "Equation", 
                "Photo_ROC" : "Photo", 
                "Diagram_ROC" : "Diagram", 
                "Table_ROC" : "Table", 
                "Plot_ROC" : "Plot"}
    
    results = {}
    scores = []
    for name in models:
        
        print "train %s"%name
        model = models[name]
        print "model", model
        
        m = model.fit(X_train, y_train)
        holdout_result = {}
        cv_result = {}
        
        y_pred = model.predict(X_test)
#         y_score = model.predict_proba(X_test)

        holdout_result["accuracy"] = model.score(X_test, y_test)
        
        y_score = model.predict_proba(X_test)
#         print "y_score", y_score[:, 2]

#         y_test_ = binarizeLabel(y_test, "scheme")
#         holdout_result["Equation_ROC"] = roc_curve(y_test, y_score[:, 0], pos_label = "equation")
#         holdout_result["Photo_ROC"] = roc_curve(y_test, y_score[:, 1], pos_label = "photo")
#         holdout_result["Diagram_ROC"] = roc_curve(y_test, y_score[:, 2], pos_label = "scheme")
#         holdout_result["Table_ROC"] = roc_curve(y_test, y_score[:, 3], pos_label = "table")
#         holdout_result["Plot_ROC"] = roc_curve(y_test, y_score[:, 4], pos_label = "visualization")
        
        
        
#         print "holdout_result", holdout_result
        
        cv_result["accuracy_list"] = cross_validation.cross_val_score(model, X, y, cv=10,  scoring="accuracy")
        cv_result["accuracy"] = cv_result["accuracy_list"].mean()
        scores.append(cv_result["accuracy_list"])
        
        print "10-fold cross-validation, %s, acc: %f"%(name, cv_result["accuracy"])
        
#         getTenFoldConfusionMatrix(model, X, y)
        
        results[name] = (holdout_result, cv_result)
        
        model.fit(X, y)
        clfFilePath = os.path.join(outModelPath, '%s.pkl'%name) 
        joblib.dump(model, clfFilePath)
#         pickle.dumps(model, clfFilePath)
        model = joblib.load(clfFilePath) 
        getTenFoldConfusionMatrix(model, X, y)
        
        
    estimator = svm.SVC(probability = True)
    cv = cross_validation.ShuffleSplit(X_train.shape[0], n_iter=10, test_size=0.25, random_state=0)
    tuned_parameters = [{'kernel': ['rbf', 'linear', 'poly'], 'gamma': [1e-2, 1e-3, 1e-4, 1e-5], 'C': [1, 10, 100, 1000]},]
    classifier = grid_search.GridSearchCV(estimator = estimator, cv = 10, param_grid = tuned_parameters)
    classifier.fit(X_train, y_train)
    print classifier.best_estimator_, '\n'
    accuracy = cross_validation.cross_val_score(classifier.best_estimator_, X, y, cv=10)
    print "accuracy", accuracy.mean()
    scores[1] = accuracy
    
    getTenFoldConfusionMatrix(model, X, y)
    
    classifier.fit(X, y)
    clfFilePath = os.path.join(outModelPath, 'SVM.pkl') 
    joblib.dump(classifier, clfFilePath)
    

   
    
    
    endTime = time.time()
    print 'Complete training model in ',  endTime - startTime, 'sec\n'
    
#     return self.saveSVMModel(path = outModelPath) 


if __name__ == '__main__':
    
    
    negPath = '/Users/sephon/Desktop/Research/VizioMetrics/Corpus/Phylogenetic/CNN_corpus/line_patch/final_negatives'
    posPath = '/Users/sephon/Desktop/Research/VizioMetrics/Corpus/Phylogenetic/CNN_corpus/line_patch/final_positives'
    
    arffOutPath = '/Users/sephon/Desktop/Research/VizioMetrics/Corpus/Phylogenetic/CNN_corpus/line_patch/train_data.arff'
    outModelPath = '/Users/sephon/Desktop/Research/VizioMetrics/Corpus/Phylogenetic/CNN_corpus/line_patch/models/'
    
    data_hor = joblib.load('/Users/sephon/Desktop/Research/VizioMetrics/Corpus/Phylogenetic/CNN_corpus/line_patch/horline_feature.dat', mmap_mode=None)
    data_anchor = joblib.load('/Users/sephon/Desktop/Research/VizioMetrics/Corpus/Phylogenetic/CNN_corpus/line_patch/anchorline_feature.dat', mmap_mode=None)
    data_for_species_hor = joblib.load('/Users/sephon/Desktop/Research/VizioMetrics/Corpus/Phylogenetic/CNN_corpus/line_patch/farspecies_horline_feature.dat', mmap_mode=None)
    data_for_species_hor_resize = joblib.load('/Users/sephon/Desktop/Research/VizioMetrics/Corpus/Phylogenetic/CNN_corpus/line_patch/farspecies_horline_resize_feature.dat', mmap_mode=None)
    
    dic = {}
#     dic = {}
    for d in data_hor:
        dic[d[0]] = d[2]
    
    for d in data_anchor:
        dic[d[0]] = d[2]
        
    for d in data_for_species_hor:
        dic[d[0]] = d[2]
        
    for d in data_for_species_hor_resize:
        dic[d[0]] = d[2]
    
    data = []
    fileList = []
    
    Y = []
    X = []
    neg_count = 0
    pos_count = 0
    
    for dirPath, dirNames, fileNames in os.walk(negPath):   
        for f in fileNames:
            extension = f.split('.')[-1].lower()
            if extension in ["jpg", "png"]:
                fileList.append(os.path.join(f))
#                 print os.path.join(f)
                
                y = 1
                if dirPath.split("/")[-1].split("_")[-1] == "negatives":
                    
                    y = 0
                
                    
                if dic.has_key(f):
                    feature = dic[f]
#                 else:
#                     feature = dic_anchor[f]
                    
                Y.append(y)
#                 print len(np.array(feature).tolist())
                X.append(np.array(feature).tolist())
                neg_count += 1
                


    for dirPath, dirNames, fileNames in os.walk(posPath):   
        for f in fileNames:
            extension = f.split('.')[-1].lower()
            if extension in ["jpg", "png"]:
                fileList.append(os.path.join(f))
#                 print os.path.join(f)
                
                y = 1
                if dirPath.split("/")[-1].split("_")[-1] == "negatives":
                    y = 0
                
                    
                if dic.has_key(f):
                    feature = dic[f]

                    
                Y.append(y)
#                 print feature
#                 print np.array(feature)
                X.append(np.array(feature).tolist())
                pos_count += 1
                
    
    posPath = '/Users/sephon/Desktop/Research/VizioMetrics/Corpus/Phylogenetic/CNN_corpus/line_patch/far_species_horline_positivs'
    for dirPath, dirNames, fileNames in os.walk(posPath):   
        for f in fileNames:
            extension = f.split('.')[-1].lower()
            if extension in ["jpg", "png"]:
                fileList.append(os.path.join(f))
#                 print os.path.join(f)
                 
                y = 1
                if dirPath.split("/")[-1].split("_")[-1] == "negatives":
                    y = 0
                 
                     
                if dic.has_key(f):
                    feature = dic[f]
 
                     
                Y.append(y)
#                 print feature
#                 print np.array(feature)
                X.append(np.array(feature).tolist())
                pos_count += 1
                
    posPath = '/Users/sephon/Desktop/Research/VizioMetrics/Corpus/Phylogenetic/CNN_corpus/line_patch/far_species_horline_positivs'
    for dirPath, dirNames, fileNames in os.walk(posPath):   
        for f in fileNames:
            extension = f.split('.')[-1].lower()
            if extension in ["jpg", "png"]:
                fileList.append(os.path.join(f))
#                 print os.path.join(f)
                 
                y = 1
                if dirPath.split("/")[-1].split("_")[-1] == "negatives":
                    y = 0
                 
                     
                if dic.has_key(f):
                    feature = dic[f]
 
                     
                Y.append(y)
#                 print feature
#                 print np.array(feature)
                X.append(np.array(feature).tolist())
                pos_count += 1
    
    print "positive data size:", pos_count
    print "negative data size:", neg_count

    print Y[0], X[0]
    
    trainModel(np.array(X),np.array(Y), outModelPath)





#     Common.saveArff(, 'train_data.arff', X, y)

#     data = np.load('/Users/sephon/Desktop/Research/VizioMetrics/Corpus/Phylogenetic/CNN_corpus/line_patch/horline_feature.npy')
    

#     y = allCatNames    
#     Common.saveArff(Opt_train.modelPath, 'train_data.arff', X, y)
#   
# #     SVM_train = SVMClassifier(Opt_train, isTrain = True)
# #     SVM_train.trainModel(X, y)
#     trainModel(X, y)
#     print 'Model has been trained'






