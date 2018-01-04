# Phylogenetic Tree Parser Tutorial

# from readCorners import readCorners
# import textDetector
# import textRemover
import cv2 as cv
import numpy as np
import pickle
from matplotlib import pyplot as plt
from ete3 import Tree
import operator, os
import random
from os import listdir
from os.path import isfile, join
from GroundTruthConverter import *

from PhyloParser import *
from boto.ec2.cloudwatch.dimension import Dimension

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
    
    ground_truth_path  = "/Users/sephon/Desktop/Research/VizioMetrics/Corpus/Phylogenetic/CNN_corpus/TreeRipper_dataset/TreeRipper_dataset.csv"
    folderPath = '/Users/sephon/Desktop/Research/VizioMetrics/Corpus/Phylogenetic/CNN_corpus/TreeRipper_dataset/images/'
    
#     ground_truth_path  = "/Users/sephon/Desktop/Research/VizioMetrics/Corpus/Phylogenetic/CNN_corpus/TreeRipper_dataset/TreeRipper_multi_dataset.csv"
#     folderPath = '/Users/sephon/Desktop/Research/VizioMetrics/Corpus/Phylogenetic/CNN_corpus/TreeRipper_dataset/multi-chart_clip/'   
    
    
    ground_truth = {}
    fileNameList = []
    with open(ground_truth_path ,'rb') as incsv:
        reader = csv.reader(incsv, dialect='excel', delimiter='\t')
        reader.next()
    
        for row in reader:
            ground_truth[row[0]] = row[1]    
#             print row[0], row[1]
            fileNameList.append(row[0])
    
    
    
    
#     fileNameList = getFilesInFolder(folderPath)
    outFileName = '/Users/sephon/Desktop/Research/VizioMetrics/Corpus/Phylogenetic/CNN_corpus/TreeRipper_multichart_line_corner_refineAnchor_v2_linethres4_20170304_git.csv'     
    results = []


    fileNameList = ["1471-2148-6-99-2-l.jpg"]
#     fileNameList = ["1471-2148-10-52-2-l.jpg"]
#     fileNameList = ["1471-2148-9-287-8-l.jpg", "1471-2148-7-224-1-l.jpg", "1471-2148-9-287-5-l.jpg", "1471-2148-9-287-6-l.jpg", "1471-2148-8-46-1-l.jpg", "1471-2148-7-227-2-l.jpg", "1471-2148-10-87-4-l.jpg", "1471-2148-10-39-1-l.jpg", "1471-2148-8-112-4-l.jpg", "1471-2148-8-160-1-l.jpg", "1471-2148-10-117-2-l.jpg"]
    for index in range(0, len(fileNameList)):

#         try: 

        filePath = os.path.join(folderPath, fileNameList[index])
        print "%s (%d / %d)"%(fileNameList[index], index, len(fileNameList))
        
        if isfile(filePath) :
            image = cv.imread(filePath,0)
#             PhyloParser.displayImage(image)

            
        image, w = PhyloParser.resizeImageByLineWidth(image)
        PhyloParser.displayImage(image)
        

        image_data = ImageData(image)
        image_data = phyloParser.preprocces(image_data, debug=False)
    
        image_data = phyloParser.detectLines(image_data, debug = False)
        
        image_data = phyloParser.getCorners(image_data, debug = False)   
        image_data = phyloParser.makeLinesFromCorner(image_data, debug = False)
        image_data = phyloParser.includeLinesFromCorners(image_data)
        image_data = phyloParser.postProcessLines(image_data)
    
        
        print "groupLines"
        image_data = phyloParser.groupLines(image_data)
        
        print "matchLineGroups"
        image_data = phyloParser.matchLineGroups(image_data, debug = False)
        

        print "getSpecies_v3"
        image_data = phyloParser.getSpecies(image_data, debug = False)

        print "constructTree"
        image_data = phyloParser.constructTree(image_data, tracing = False , debug = False)
    
        truth_string = ground_truth[fileNameList[index]]
        print truth_string
        print image_data.treeStructure
        t1 = PhyloTree(image_data.treeStructure + ";")
        t2 = PhyloTree(truth_string)
        PhyloTree.rename_node(t1, rename_all=True)
        PhyloTree.rename_node(t2, rename_all=True)
        
        print t1
#             t1.drawTree()
        print t2
        distance = PhyloTree.zhang_shasha_distance(t1, t2)
        num_node = PhyloTree.getNodeNum(t2) 
        num_leaf = t2.getLeafCount()
        score = distance/float(num_node)
        
        print "distance %d/%d = %f, leave=%d" %(distance, num_node, score, num_leaf)
        print "contour count=%d , sharebox count=%d"%(image_data.count_contourBoxes, image_data.count_shareBox)
        results.append([fileNameList[index], distance, num_node, score, num_leaf, image_data.count_contourBoxes, image_data.count_shareBox])
        
  
#         PhyloParser.displayImage(image)
        
#         except:
#             truth_string = string2TreeString(ground_truth[fileNameList[index]], rename = True)
#             t2 = PhyloTree(truth_string + ";")
#             num_node = count_node(t2)
#             results.append([fileNameList[index], distance, num_node, score, num_leaf, image_data.count_contourBoxes, image_data.count_shareBox])

        
#     saveResult(results, outFileName)
    
    
 