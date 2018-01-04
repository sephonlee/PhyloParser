# Phylogenetic Tree Parser Tutorial

# from readCorners import readCorners
# import textDetector
# import textRemover
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
from ete3 import Tree
import operator, os
import random
from os import listdir
from os.path import isfile, join
from GroundTruthConverter import *

from sklearn.externals import joblib



from PhyloParser import *

def getFilesInFolder(folderPath):
    fileNameList = [f for f in listdir(folderPath) if isfile(join(folderPath, f)) and f.split(".")[-1] == "jpg"]

    return fileNameList

def saveResult(data, output):
    with open(output, 'wb') as outcsv:
        writer = csv.writer(outcsv, dialect='excel')
        header = ['filename', 'distance', 'num_node', "score", "num_leaf", "count_contourBoxes", "count_shareBox"]
        writer.writerow(header)
        
        for row in data:
            writer.writerow(row)


if __name__ == '__main__':
    
    clfPath = "/Users/sephon/Desktop/Research/VizioMetrics/Corpus/Phylogenetic/CNN_corpus/line_patch/models/RF.pkl"
    
    phyloParser = PhyloParser(clfPath = clfPath)
    ground_truth_path  = "/Users/sephon/Desktop/Research/VizioMetrics/Corpus/Phylogenetic/CNN_corpus/hq_ground_truth0228.csv"
    folderPath = '/Users/sephon/Desktop/Research/VizioMetrics/Corpus/Phylogenetic/CNN_corpus/high_quality_tree'
    ground_truth = {}
    fileNameList = []
    with open(ground_truth_path ,'rb') as incsv:
        reader = csv.reader(incsv, dialect='excel')
        reader.next()
    
        for row in reader:
            ground_truth[row[0]] = row[1]    
#             print row[0], row[1]
            fileNameList.append(row[0])
    
    
    
    
#     fileNameList = getFilesInFolder(folderPath)
    outFileName = '/Users/sephon/Desktop/Research/VizioMetrics/Corpus/Phylogenetic/CNN_corpus/high_quality_tree_20170307_rline_new_git.csv'     
    results = []

#     fileNameList = ["PMC1474148_ijbsv02p0133g05.jpg"]
    fileNameList = ["PMC2233639_1471-2199-8-95-2.jpg"]
#     fileNameList = ['PMC2697986_1471-2148-9-107-3.jpg']

    for index in range(0, len(fileNameList)):

        filePath = os.path.join(folderPath, fileNameList[index])
        print 
        print "%s (%d / %d)"%(fileNameList[index], index, len(fileNameList))
        
        if isfile(filePath) :
            image = cv.imread(filePath,0)
            

        image, w = PhyloParser.resizeImageByLineWidth(image)
        PhyloParser.displayImage(image)
        
        
         
        image_data = ImageData(image)
        image_data = phyloParser.preprocces(image_data, debug= False)

        image_data = phyloParser.detectLines(image_data, debug = True)
        image_data = phyloParser.getCorners(image_data, debug = False)   
        image_data = phyloParser.makeLinesFromCorner(image_data, debug = False)
        image_data = phyloParser.includeLinesFromCorners(image_data)
        image_data = phyloParser.postProcessLines(image_data)
        
    #         image_data = phyloParser.refineLinesByCorners(image_data)
    
        image_data = phyloParser.groupLines(image_data)
        print "matchLineGroup"
        image_data = phyloParser.matchLineGroups(image_data, debug = False)
        
        # image_data = phyloParser.getSpecies(image_data, debug = True)
        print "getSpecies_v3"
        image_data = phyloParser.getSpecies(image_data, debug = False)
        print "end getSpecies_v3"
        image_data = phyloParser.constructTree(image_data, tracing = False , debug = False)
    
        
    
        # PhyloParser1.0
    #         image_data = phyloParser.matchLines(image_data, debug = False)
    #         image_data = phyloParser.makeTree(image_data, tracing = False, debug = False, showResult = False)
    
    
        ####### evaluation
        
        truth_string = string2TreeString(ground_truth[fileNameList[index]], rename = True)
        print image_data.treeStructure
        t1 = PhyloTree(image_data.treeStructure + ";")
        t2 = PhyloTree(truth_string + ";")
        PhyloTree.rename_node(t1, rename_all=True)
        PhyloTree.rename_node(t2, rename_all=True)
        
        print t1
#         t1.drawTree()
        print t2
        distance = PhyloTree.zhang_shasha_distance(t1, t2)
        num_node = PhyloTree.getNodeNum(t2)
        num_leaf = t2.getLeafCount()
        score = distance/float(num_node)
        
        print "distance %d/%d = %f, leave=%d" %(distance, num_node, score, num_leaf)
        print "contour count=%d , sharebox count=%d"%(image_data.count_contourBoxes, image_data.count_shareBox)
        results.append([fileNameList[index], distance, num_node, score, num_leaf, image_data.count_contourBoxes, image_data.count_shareBox])
        
#         except:
# # #             truth_string = string2TreeString(ground_truth[fileNameList[index]], rename = True)
# # #             t2 = PhyloTree(truth_string + ";")
# # #             num_node = count_node(t2)
#                 results.append([fileNameList[index], -1, -1, -1])
         
#     saveResult(results, outFileName)