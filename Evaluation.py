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

            
def saveCSV(data, output, mode = 'wb'):
    with open(output, mode) as outcsv:
        writer = csv.writer(outcsv, dialect='excel')
        for row in data:
            writer.writerow(row)


if __name__ == '__main__':
    
    rootPath = "/Users/sephon/Desktop/Research/VizioMetrics/Corpus/Phylogenetic/CNN_corpus/"
    
    clfPath = os.path.join(rootPath, "line_patch/models/RF.pkl")
    phyloParser = PhyloParser(clfPath = clfPath)
    
    
    
    
    
    
    
    ground_truth_path  = os.path.join(rootPath, "hq_ground_truth0228.csv")
    folderPath = os.path.join(rootPath, "high_quality_tree")
    ground_truth = {}
    fileNameList = []
    with open(ground_truth_path ,'rb') as incsv:
        reader = csv.reader(incsv, dialect='excel')
        reader.next()
       
        for row in reader:
            ground_truth[row[0]] = row[1]    
            fileNameList.append(row[0])
#      
#      
# ##---------------------------- BASE --------------------------------##
#  
#     outFileName = os.path.join(rootPath, "hqtree_base_20170307.csv")
#     results = []
#    
#    
#     for index in range(0, len(fileNameList)):
#    
#         try:
#             filePath = os.path.join(folderPath, fileNameList[index])
#             print 
#             print "%s (%d / %d)"%(fileNameList[index], index, len(fileNameList))
#                
#             if isfile(filePath) :
#                 image = cv.imread(filePath,0)
#                    
#     #         PhyloParser.displayImage(image)
#             image, w = PhyloParser.resizeImageByLineWidth(image)
#    
#                 
#             image_data = ImageData(image)
#             image_data = phyloParser.preprocces(image_data, debug=False)
#            
#             image_data = phyloParser.detectLines(image_data, debug = False)
#             image_data = phyloParser.getCorners(image_data, debug = False)   
#             image_data = phyloParser.makeLinesFromCorner(image_data, debug = False)
#             image_data = phyloParser.includeLinesFromCorners(image_data)
#             image_data = phyloParser.postProcessLines(image_data)
#                
#     
#             image_data = phyloParser.groupLines(image_data)
#             image_data = phyloParser.matchLineGroups(image_data, debug = False)
#                
#    
#             image_data = phyloParser.getSpecies_v3(image_data, debug = False)
#    
#    
#             image_data = phyloParser.constructTree_eval(image_data, fixTree = False, tracing = False, orphanHint = False, debug = False)
#            
#    
#             ####### evaluation
#                
#             truth_string = string2TreeString(ground_truth[fileNameList[index]], rename = True)
#             print image_data.treeStructure
#             t1 = PhyloTree(image_data.treeStructure + ";")
#             t2 = PhyloTree(truth_string + ";")
#             PhyloTree.rename_node(t1, rename_all=True)
#             PhyloTree.rename_node(t2, rename_all=True)
#                
#             distance = PhyloTree.zhang_shasha_distance(t1, t2)
#             num_node = PhyloTree.getNodeNum(t2)
#             num_leaf = t2.getLeafCount()
#             score = distance/float(num_node)
#                
#             print "distance %d/%d = %f, leave=%d" %(distance, num_node, score, num_leaf)
#             print "contour count=%d , sharebox count=%d"%(image_data.count_contourBoxes, image_data.count_shareBox)
#             results.append([fileNameList[index], distance, num_node, score, num_leaf, image_data.count_contourBoxes, image_data.count_shareBox])
#            
#         except:
#             results.append([fileNameList[index], -1, -1, -1])
#             
#     saveResult(results, outFileName)
#        
#        
# ##---------------------------- fix tree only --------------------------------##
#    
#     outFileName = os.path.join(rootPath, "hqtree_base_fixTree_20170307.csv")
#     results = []
#    
#    
#     for index in range(0, len(fileNameList)):
#    
#         try:
#             filePath = os.path.join(folderPath, fileNameList[index])
#             print 
#             print "%s (%d / %d)"%(fileNameList[index], index, len(fileNameList))
#                
#             if isfile(filePath) :
#                 image = cv.imread(filePath,0)
#                    
#     #         PhyloParser.displayImage(image)
#             image, w = PhyloParser.resizeImageByLineWidth(image)
#    
#                 
#             image_data = ImageData(image)
#             image_data = phyloParser.preprocces(image_data, debug=False)
#            
#             image_data = phyloParser.detectLines(image_data, debug = False)
#             image_data = phyloParser.getCorners(image_data, debug = False)   
#             image_data = phyloParser.makeLinesFromCorner(image_data, debug = False)
#             image_data = phyloParser.includeLinesFromCorners(image_data)
#             image_data = phyloParser.postProcessLines(image_data)
#                
#     
#             image_data = phyloParser.groupLines(image_data)
#             image_data = phyloParser.matchLineGroups(image_data, debug = False)
#                
#    
#             image_data = phyloParser.getSpecies_v3(image_data, debug = False)
#    
#    
#             image_data = phyloParser.constructTree_eval(image_data, fixTree = True, tracing = False, orphanHint = False, debug = False)
#            
#            
#    
#             ####### evaluation
#                
#             truth_string = string2TreeString(ground_truth[fileNameList[index]], rename = True)
#             print image_data.treeStructure
#             t1 = PhyloTree(image_data.treeStructure + ";")
#             t2 = PhyloTree(truth_string + ";")
#             PhyloTree.rename_node(t1, rename_all=True)
#             PhyloTree.rename_node(t2, rename_all=True)
#                
#             distance = PhyloTree.zhang_shasha_distance(t1, t2)
#             num_node = PhyloTree.getNodeNum(t2)
#             num_leaf = t2.getLeafCount()
#             score = distance/float(num_node)
#                
#             print "distance %d/%d = %f, leave=%d" %(distance, num_node, score, num_leaf)
#             print "contour count=%d , sharebox count=%d"%(image_data.count_contourBoxes, image_data.count_shareBox)
#             results.append([fileNameList[index], distance, num_node, score, num_leaf, image_data.count_contourBoxes, image_data.count_shareBox])
#            
#         except:
#             results.append([fileNameList[index], -1, -1, -1])
#             
#     saveResult(results, outFileName)
#        
#        
#    
# ##---------------------------- fixTree + orphanHint --------------------------------##
#     
#     outFileName = os.path.join(rootPath, "hqtree_base_fixTree_orphanHint_20170307.csv")
#     results = []
#     
#     
#     for index in range(0, len(fileNameList)):
#     
#         try:
#             filePath = os.path.join(folderPath, fileNameList[index])
#             print 
#             print "%s (%d / %d)"%(fileNameList[index], index, len(fileNameList))
#                 
#             if isfile(filePath) :
#                 image = cv.imread(filePath,0)
#                     
#     #         PhyloParser.displayImage(image)
#             image, w = PhyloParser.resizeImageByLineWidth(image)
#     
#                  
#             image_data = ImageData(image)
#             image_data = phyloParser.preprocces(image_data, debug=False)
#             
#             image_data = phyloParser.detectLines(image_data, debug = False)
#             image_data = phyloParser.getCorners(image_data, debug = False)   
#             image_data = phyloParser.makeLinesFromCorner(image_data, debug = False)
#             image_data = phyloParser.includeLinesFromCorners(image_data)
#             image_data = phyloParser.postProcessLines(image_data)
#                 
#      
#             image_data = phyloParser.groupLines(image_data)
#             image_data = phyloParser.matchLineGroups(image_data, debug = False)
#                 
#     
#             image_data = phyloParser.getSpecies_v3(image_data, debug = False)
#     
#     
#             image_data = phyloParser.constructTree_eval(image_data, fixTree = True, tracing = False, orphanHint = True, debug = False)
#             
#             
#     
#             ####### evaluation
#                 
#             truth_string = string2TreeString(ground_truth[fileNameList[index]], rename = True)
#             print image_data.treeStructure
#             t1 = PhyloTree(image_data.treeStructure + ";")
#             t2 = PhyloTree(truth_string + ";")
#             PhyloTree.rename_node(t1, rename_all=True)
#             PhyloTree.rename_node(t2, rename_all=True)
#                 
#             distance = PhyloTree.zhang_shasha_distance(t1, t2)
#             num_node = PhyloTree.getNodeNum(t2)
#             num_leaf = t2.getLeafCount()
#             score = distance/float(num_node)
#                 
#             print "distance %d/%d = %f, leave=%d" %(distance, num_node, score, num_leaf)
#             print "contour count=%d , sharebox count=%d"%(image_data.count_contourBoxes, image_data.count_shareBox)
#             results.append([fileNameList[index], distance, num_node, score, num_leaf, image_data.count_contourBoxes, image_data.count_shareBox])
#             
#         except:
#             results.append([fileNameList[index], -1, -1, -1])
#              
#     saveResult(results, outFileName)
  
  
# ##---------------------------- line + tracing  --------------------------------##
#    
#     outFileName = os.path.join(rootPath, "hqtree_base_tracing_20170313_73-141.csv") 
#     saveCSV([['filename', 'distance', 'num_node', "score", "num_leaf", "count_contourBoxes", "count_shareBox", "line_width"]], outFileName)
#       
#    
#     for index in range(0, len(fileNameList)):
#    
#         try:
#             filePath = os.path.join(folderPath, fileNameList[index])
#             print 
#             print "%s (%d / %d)"%(fileNameList[index], index, len(fileNameList))
#                
#             if isfile(filePath) :
#                 image = cv.imread(filePath,0)
#                    
#     #         PhyloParser.displayImage(image)
#             image, w = PhyloParser.resizeImageByLineWidth(image)
#    
#                 
#             image_data = ImageData(image)
#             image_data = phyloParser.preprocces(image_data, debug=False)
#            
#             image_data = phyloParser.detectLines(image_data, debug = False)
#             image_data = phyloParser.getCorners(image_data, debug = False)   
#             image_data = phyloParser.makeLinesFromCorner(image_data, debug = False)
#             image_data = phyloParser.includeLinesFromCorners(image_data)
#             image_data = phyloParser.postProcessLines(image_data)
#                
#     
#             image_data = phyloParser.groupLines(image_data)
#             image_data = phyloParser.matchLineGroups(image_data, debug = False)
#                
#    
#             image_data = phyloParser.getSpecies_v3(image_data, debug = False)
#    
#    
#             image_data = phyloParser.constructTree_eval(image_data, fixTree = False, tracing = True, orphanHint = False, debug = False)
#            
#            
#    
#             ####### evaluation
#                
#             truth_string = string2TreeString(ground_truth[fileNameList[index]], rename = True)
#             print image_data.treeStructure
#             t1 = PhyloTree(image_data.treeStructure + ";")
#             t2 = PhyloTree(truth_string + ";")
#             PhyloTree.rename_node(t1, rename_all=True)
#             PhyloTree.rename_node(t2, rename_all=True)
#                
#             distance = PhyloTree.zhang_shasha_distance(t1, t2)
#             num_node = PhyloTree.getNodeNum(t2)
#             num_leaf = t2.getLeafCount()
#             score = distance/float(num_node)
#                
#             print "distance %d/%d = %f, leave=%d" %(distance, num_node, score, num_leaf)
#             print "contour count=%d , sharebox count=%d"%(image_data.count_contourBoxes, image_data.count_shareBox)
#             saveCSV([[fileNameList[index], distance, num_node, score, num_leaf, image_data.count_contourBoxes, image_data.count_shareBox, w]], outFileName, mode='ab')
#         except:
# #             results.append([[fileNameList[index], distance, num_node, score, num_leaf, image_data.count_contourBoxes, image_data.count_shareBox]])
#             saveCSV([[fileNameList[index], -1, -1, -1, -1,-1,-1, -1]], outFileName, mode='ab')
           

##---------------------------- line + fixTree + tracing  --------------------------------##
   
    outFileName = os.path.join(rootPath, "hqtree_base_fixTree_tracing2_20170313.csv") 
    saveCSV([['filename', 'distance', 'num_node', "score", "num_leaf", "count_contourBoxes", "count_shareBox", "line_width"]], outFileName)
      
   
    for index in range(0, len(fileNameList)):
   
        try:
            filePath = os.path.join(folderPath, fileNameList[index])
            print 
            print "%s (%d / %d)"%(fileNameList[index], index, len(fileNameList))
               
            if isfile(filePath) :
                image = cv.imread(filePath,0)
                   
    #         PhyloParser.displayImage(image)
            image, w = PhyloParser.resizeImageByLineWidth(image)
   
                
            image_data = ImageData(image)
            image_data = phyloParser.preprocces(image_data, debug=False)
           
            image_data = phyloParser.detectLines(image_data, debug = False)
            image_data = phyloParser.getCorners(image_data, debug = False)   
            image_data = phyloParser.makeLinesFromCorner(image_data, debug = False)
            image_data = phyloParser.includeLinesFromCorners(image_data)
            image_data = phyloParser.postProcessLines(image_data)
               
    
            image_data = phyloParser.groupLines(image_data)
            image_data = phyloParser.matchLineGroups(image_data, debug = False)
               
   
            image_data = phyloParser.getSpecies_v3(image_data, debug = False)
   
   
            image_data = phyloParser.constructTree_eval(image_data, fixTree = True, tracing = True, orphanHint = False, debug = False)
           
           
   
            ####### evaluation
               
            truth_string = string2TreeString(ground_truth[fileNameList[index]], rename = True)
            print image_data.treeStructure
            t1 = PhyloTree(image_data.treeStructure + ";")
            t2 = PhyloTree(truth_string + ";")
            PhyloTree.rename_node(t1, rename_all=True)
            PhyloTree.rename_node(t2, rename_all=True)
               
            distance = PhyloTree.zhang_shasha_distance(t1, t2)
            num_node = PhyloTree.getNodeNum(t2)
            num_leaf = t2.getLeafCount()
            score = distance/float(num_node)
               
            print "distance %d/%d = %f, leave=%d" %(distance, num_node, score, num_leaf)
            print "contour count=%d , sharebox count=%d"%(image_data.count_contourBoxes, image_data.count_shareBox)
            saveCSV([[fileNameList[index], distance, num_node, score, num_leaf, image_data.count_contourBoxes, image_data.count_shareBox, w]], outFileName, mode='ab')
        except:
#             results.append([[fileNameList[index], distance, num_node, score, num_leaf, image_data.count_contourBoxes, image_data.count_shareBox]])
            saveCSV([[fileNameList[index], -1, -1, -1, -1,-1,-1, -1]], outFileName, mode='ab')


##---------------------------- line + fixTree + tracing + orphanHint--------------------------------##
  
    outFileName = os.path.join(rootPath, "hqtree_base_fixTree_tracing2_orphanHint_20170313.csv") 
    saveCSV([['filename', 'distance', 'num_node', "score", "num_leaf", "count_contourBoxes", "count_shareBox", "line_width"]], outFileName)
      
   
    for index in range(0, len(fileNameList)):
   
        try:
            filePath = os.path.join(folderPath, fileNameList[index])
            print 
            print "%s (%d / %d)"%(fileNameList[index], index, len(fileNameList))
               
            if isfile(filePath) :
                image = cv.imread(filePath,0)
                   
    #         PhyloParser.displayImage(image)
            image, w = PhyloParser.resizeImageByLineWidth(image)
   
                
            image_data = ImageData(image)
            image_data = phyloParser.preprocces(image_data, debug=False)
           
            image_data = phyloParser.detectLines(image_data, debug = False)
            image_data = phyloParser.getCorners(image_data, debug = False)   
            image_data = phyloParser.makeLinesFromCorner(image_data, debug = False)
            image_data = phyloParser.includeLinesFromCorners(image_data)
            image_data = phyloParser.postProcessLines(image_data)
               
    
            image_data = phyloParser.groupLines(image_data)
            image_data = phyloParser.matchLineGroups(image_data, debug = False)
               
   
            image_data = phyloParser.getSpecies_v3(image_data, debug = False)
   
   
            image_data = phyloParser.constructTree_eval(image_data, fixTree = True, tracing = True, orphanHint = True, debug = False)
           
           
   
            ####### evaluation
               
            truth_string = string2TreeString(ground_truth[fileNameList[index]], rename = True)
            print image_data.treeStructure
            t1 = PhyloTree(image_data.treeStructure + ";")
            t2 = PhyloTree(truth_string + ";")
            PhyloTree.rename_node(t1, rename_all=True)
            PhyloTree.rename_node(t2, rename_all=True)
               
            distance = PhyloTree.zhang_shasha_distance(t1, t2)
            num_node = PhyloTree.getNodeNum(t2)
            num_leaf = t2.getLeafCount()
            score = distance/float(num_node)
               
            print "distance %d/%d = %f, leave=%d" %(distance, num_node, score, num_leaf)
            print "contour count=%d , sharebox count=%d"%(image_data.count_contourBoxes, image_data.count_shareBox)
            saveCSV([[fileNameList[index], distance, num_node, score, num_leaf, image_data.count_contourBoxes, image_data.count_shareBox, w]], outFileName, mode='ab')
        except:
#             results.append([[fileNameList[index], distance, num_node, score, num_leaf, image_data.count_contourBoxes, image_data.count_shareBox]])
            saveCSV([[fileNameList[index], -1, -1, -1, -1,-1,-1, -1]], outFileName, mode='ab')
            
            
            
# ##---------------------------- line + tracing + orphanHint --------------------------------##
#   
#     outFileName = os.path.join(rootPath, "hqtree_base_tracing_orphanHint_20170307.csv") 
#     saveCSV([['filename', 'distance', 'num_node', "score", "num_leaf", "count_contourBoxes", "count_shareBox", "line_width"]], outFileName)
#      
#   
#     for index in range(0, len(fileNameList)):
#   
#         try:
#             filePath = os.path.join(folderPath, fileNameList[index])
#             print 
#             print "%s (%d / %d)"%(fileNameList[index], index, len(fileNameList))
#               
#             if isfile(filePath) :
#                 image = cv.imread(filePath,0)
#                   
#     #         PhyloParser.displayImage(image)
#             image, w = PhyloParser.resizeImageByLineWidth(image)
#   
#                
#             image_data = ImageData(image)
#             image_data = phyloParser.preprocces(image_data, debug=False)
#           
#             image_data = phyloParser.detectLines(image_data, debug = False)
#             image_data = phyloParser.getCorners(image_data, debug = False)   
#             image_data = phyloParser.makeLinesFromCorner(image_data, debug = False)
#             image_data = phyloParser.includeLinesFromCorners(image_data)
#             image_data = phyloParser.postProcessLines(image_data)
#               
#    
#             image_data = phyloParser.groupLines(image_data)
#             image_data = phyloParser.matchLineGroups(image_data, debug = False)
#               
#   
#             image_data = phyloParser.getSpecies_v3(image_data, debug = False)
#   
#   
#             image_data = phyloParser.constructTree_eval(image_data, fixTree = False, tracing = True, orphanHint = True, debug = False)
#           
#           
#   
#             ####### evaluation
#               
#             truth_string = string2TreeString(ground_truth[fileNameList[index]], rename = True)
#             print image_data.treeStructure
#             t1 = PhyloTree(image_data.treeStructure + ";")
#             t2 = PhyloTree(truth_string + ";")
#             PhyloTree.rename_node(t1, rename_all=True)
#             PhyloTree.rename_node(t2, rename_all=True)
#               
#             distance = PhyloTree.zhang_shasha_distance(t1, t2)
#             num_node = PhyloTree.getNodeNum(t2)
#             num_leaf = t2.getLeafCount()
#             score = distance/float(num_node)
#               
#             print "distance %d/%d = %f, leave=%d" %(distance, num_node, score, num_leaf)
#             print "contour count=%d , sharebox count=%d"%(image_data.count_contourBoxes, image_data.count_shareBox)
#             saveCSV([[fileNameList[index], distance, num_node, score, num_leaf, image_data.count_contourBoxes, image_data.count_shareBox, w]], outFileName, mode='ab')
#         except:
# #             results.append([[fileNameList[index], distance, num_node, score, num_leaf, image_data.count_contourBoxes, image_data.count_shareBox]])
#             saveCSV([[fileNameList[index], -1, -1, -1, -1,-1,-1, -1]], outFileName, mode='ab')
     
     
           
     
     
     
# ##---------------------------- tracing only --------------------------------##
#   
#     outFileName = os.path.join(rootPath, "hqtree_pure_tracing_20170307.csv") 
#     saveCSV([['filename', 'distance', 'num_node', "score", "num_leaf", "count_contourBoxes", "count_shareBox", "line_width"]], outFileName)
#      
#   
#     for index in range(0, len(fileNameList)):
#   
#         try:
#             filePath = os.path.join(folderPath, fileNameList[index])
#             print 
#             print "%s (%d / %d)"%(fileNameList[index], index, len(fileNameList))
#               
#             if isfile(filePath) :
#                 image = cv.imread(filePath,0)
#                   
#     #         PhyloParser.displayImage(image)
#             image, w = PhyloParser.resizeImageByLineWidth(image)
#   
#                
#             image_data = ImageData(image)
#             image_data = phyloParser.preprocces(image_data, debug=False)
#           
#             image_data = phyloParser.detectLines(image_data, debug = False)
#             image_data = phyloParser.getCorners(image_data, debug = False)   
#             image_data = phyloParser.makeLinesFromCorner(image_data, debug = False)
#             image_data = phyloParser.includeLinesFromCorners(image_data)
#             image_data = phyloParser.postProcessLines(image_data)
#               
#    
#             image_data = phyloParser.groupLines(image_data)
#             image_data = phyloParser.matchLineGroups(image_data, debug = False)
#               
#   
#             image_data = phyloParser.getSpecies_v3(image_data, debug = False)
#   
#             ######
#             image_data = phyloParser.constructTreeByTracing(image_data, debug = False, tracing = True)
#           
#   
#             ####### evaluation
#               
#             truth_string = string2TreeString(ground_truth[fileNameList[index]], rename = True)
#             print image_data.treeStructure
#             t1 = PhyloTree(image_data.treeStructure + ";")
#             t2 = PhyloTree(truth_string + ";")
#             PhyloTree.rename_node(t1, rename_all=True)
#             PhyloTree.rename_node(t2, rename_all=True)
#               
#             distance = PhyloTree.zhang_shasha_distance(t1, t2)
#             num_node = PhyloTree.getNodeNum(t2)
#             num_leaf = t2.getLeafCount()
#             score = distance/float(num_node)
#               
#             print "distance %d/%d = %f, leave=%d" %(distance, num_node, score, num_leaf)
#             print "contour count=%d , sharebox count=%d"%(image_data.count_contourBoxes, image_data.count_shareBox)
#             saveCSV([[fileNameList[index], distance, num_node, score, num_leaf, image_data.count_contourBoxes, image_data.count_shareBox, w]], outFileName, mode='ab')
#         except:
# #             results.append([[fileNameList[index], distance, num_node, score, num_leaf, image_data.count_contourBoxes, image_data.count_shareBox]])
#             saveCSV([[fileNameList[index], -1, -1, -1, -1,-1,-1, -1]], outFileName, mode='ab')
            
                
     
     
 
# ############################################ Ripper Dataset ###################################################
# ##---------------------------- fixTree only --------------------------------##
#      
#  
#  
#  
#     ground_truth_path  = os.path.join(rootPath, "TreeRipper_dataset/TreeRipper_dataset.csv")
#     folderPath = os.path.join(rootPath, "TreeRipper_dataset/images/")
#      
# #     ground_truth_path  = "/Users/sephon/Desktop/Research/VizioMetrics/Corpus/Phylogenetic/CNN_corpus/TreeRipper_dataset/TreeRipper_multi_dataset.csv"
# #     folderPath = '/Users/sephon/Desktop/Research/VizioMetrics/Corpus/Phylogenetic/CNN_corpus/TreeRipper_dataset/multi-chart_clip/'   
#      
#      
#     ground_truth = {}
#     fileNameList = []
#     with open(ground_truth_path ,'rb') as incsv:
#         reader = csv.reader(incsv, dialect='excel', delimiter='\t')
#         reader.next()
#      
#         for row in reader:
#             ground_truth[row[0]] = row[1]    
#             fileNameList.append(row[0])
#  
#  
#     outFileName = os.path.join(rootPath, "ripper_base_fixTree_20170307.csv")
#     results = []
#  
#     saveCSV([['filename', 'distance', 'num_node', "score", "num_leaf", "count_contourBoxes", "count_shareBox", "line_width"]], outFileName)
#      
#     for index in range(0, len(fileNameList)):
#  
#         try: 
#  
#             filePath = os.path.join(folderPath, fileNameList[index])
#             print "%s (%d / %d)"%(fileNameList[index], index, len(fileNameList))
#              
#             if isfile(filePath) :
#                 image = cv.imread(filePath,0)    
#              
#      
#             image, w = PhyloParser.resizeImageByLineWidth(image)
#  
#  
#             image_data = ImageData(image)
#             image_data = phyloParser.preprocces(image_data, debug=False)
#  
#             image_data = phyloParser.detectLines(image_data, debug = False)
#             image_data = phyloParser.getCorners(image_data, debug = False)   
#             image_data = phyloParser.makeLinesFromCorner(image_data, debug = False)
#             image_data = phyloParser.includeLinesFromCorners(image_data)
#             image_data = phyloParser.postProcessLines(image_data)
#              
#          
#          
#             image_data = phyloParser.groupLines(image_data)
#             image_data = phyloParser.matchLineGroups(image_data, debug = False)
#              
#  
#             image_data = phyloParser.getSpecies_v3(image_data, debug = False)
#  
#             image_data = phyloParser.constructTree_eval(image_data, fixTree = True, tracing = False, orphanHint = False, debug = False)
#          
#          
#             truth_string = ground_truth[fileNameList[index]]
#             t1 = PhyloTree(image_data.treeStructure + ";")
#             t2 = PhyloTree(truth_string)
#             PhyloTree.rename_node(t1, rename_all=True)
#             PhyloTree.rename_node(t2, rename_all=True)
#              
#             distance = PhyloTree.zhang_shasha_distance(t1, t2)
#             num_node = PhyloTree.getNodeNum(t2) 
#             num_leaf = t2.getLeafCount()
#             score = distance/float(num_node)
#              
#             print "distance %d/%d = %f, leave=%d" %(distance, num_node, score, num_leaf)
#             print "contour count=%d , sharebox count=%d"%(image_data.count_contourBoxes, image_data.count_shareBox)
# #             results.append([fileNameList[index], distance, num_node, score, num_leaf, image_data.count_contourBoxes, image_data.count_shareBox])
#             saveCSV([[fileNameList[index], distance, num_node, score, num_leaf, image_data.count_contourBoxes, image_data.count_shareBox, w]], outFileName, mode='ab')
#         except:
# #             results.append([[fileNameList[index], distance, num_node, score, num_leaf, image_data.count_contourBoxes, image_data.count_shareBox]])
#             saveCSV([[fileNameList[index], -1, -1, -1, -1,-1,-1, -1]], outFileName, mode='ab')
#         
# 
# 
# 


# ############################################ Ripper Dataset ###################################################
# ##---------------------------- fixTree only --------------------------------##
#     
# 
# 
# 
# #     ground_truth_path  = "/Users/sephon/Desktop/Research/VizioMetrics/Corpus/Phylogenetic/CNN_corpus/TreeRipper_dataset/TreeRipper_dataset.csv"
# #     folderPath = '/Users/sephon/Desktop/Research/VizioMetrics/Corpus/Phylogenetic/CNN_corpus/TreeRipper_dataset/images/'
#     
#     ground_truth_path  = os.path.join(rootPath, "TreeRipper_dataset/TreeRipper_multi_dataset.csv")
#     folderPath = os.path.join(rootPath, "TreeRipper_dataset/multi-chart_clip/")
#     
#     
#     ground_truth = {}
#     fileNameList = []
#     with open(ground_truth_path ,'rb') as incsv:
#         reader = csv.reader(incsv, dialect='excel', delimiter='\t')
#         reader.next()
#     
#         for row in reader:
#             ground_truth[row[0]] = row[1]    
#             fileNameList.append(row[0])
# 
# 
#     outFileName = os.path.join(rootPath, "ripper_multiclip_base_fixTree_20170307.csv")
#     results = []
# 
#     saveCSV([['filename', 'distance', 'num_node', "score", "num_leaf", "count_contourBoxes", "count_shareBox", "line_width"]], outFileName)
#     
#     for index in range(0, len(fileNameList)):
# 
#         try: 
# 
#             filePath = os.path.join(folderPath, fileNameList[index])
#             print "%s (%d / %d)"%(fileNameList[index], index, len(fileNameList))
#             
#             if isfile(filePath) :
#                 image = cv.imread(filePath,0)    
#             
#     
#             image, w = PhyloParser.resizeImageByLineWidth(image)
# 
#             image_data = ImageData(image)
#             image_data = phyloParser.preprocces(image_data, debug=False)
#         
#             image_data = phyloParser.detectLines(image_data, debug = False)
#             image_data = phyloParser.getCorners(image_data, debug = False)   
#             image_data = phyloParser.makeLinesFromCorner(image_data, debug = False)
#             image_data = phyloParser.includeLinesFromCorners(image_data)
#             image_data = phyloParser.postProcessLines(image_data)
#         
#         
#             image_data = phyloParser.groupLines(image_data)
#             image_data = phyloParser.matchLineGroups(image_data, debug = False)
#             
# 
#             image_data = phyloParser.getSpecies_v3(image_data, debug = False)
# 
#             image_data = phyloParser.constructTree_eval(image_data, fixTree = True, tracing = False, orphanHint = False, debug = False)
#         
#         
#             truth_string = ground_truth[fileNameList[index]]
#             t1 = PhyloTree(image_data.treeStructure + ";")
#             t2 = PhyloTree(truth_string)
#             PhyloTree.rename_node(t1, rename_all=True)
#             PhyloTree.rename_node(t2, rename_all=True)
#             
#             distance = PhyloTree.zhang_shasha_distance(t1, t2)
#             num_node = PhyloTree.getNodeNum(t2) 
#             num_leaf = t2.getLeafCount()
#             score = distance/float(num_node)
#             
#             print "distance %d/%d = %f, leave=%d" %(distance, num_node, score, num_leaf)
#             print "contour count=%d , sharebox count=%d"%(image_data.count_contourBoxes, image_data.count_shareBox)
#             saveCSV([[fileNameList[index], distance, num_node, score, num_leaf, image_data.count_contourBoxes, image_data.count_shareBox, w]], outFileName, mode='ab')
#         except:
# #             results.append([[fileNameList[index], distance, num_node, score, num_leaf, image_data.count_contourBoxes, image_data.count_shareBox]])
#             saveCSV([[fileNameList[index], -1, -1, -1, -1,-1,-1, -1]], outFileName, mode='ab')
        


# ############################################ Ripper Dataset ###################################################
# ##---------------------------- fixTree only --------------------------------##
#       
#   
#   
#   
#     ground_truth_path  = os.path.join(rootPath, "TreeRipper_dataset/TreeRipper_dataset.csv")
#     folderPath = os.path.join(rootPath, "TreeRipper_dataset/images/")
#       
# #     ground_truth_path  = "/Users/sephon/Desktop/Research/VizioMetrics/Corpus/Phylogenetic/CNN_corpus/TreeRipper_dataset/TreeRipper_multi_dataset.csv"
# #     folderPath = '/Users/sephon/Desktop/Research/VizioMetrics/Corpus/Phylogenetic/CNN_corpus/TreeRipper_dataset/multi-chart_clip/'   
#       
#       
#     ground_truth = {}
#     fileNameList = []
#     with open(ground_truth_path ,'rb') as incsv:
#         reader = csv.reader(incsv, dialect='excel', delimiter='\t')
#         reader.next()
#       
#         for row in reader:
#             ground_truth[row[0]] = row[1]    
#             fileNameList.append(row[0])
#   
#   
#     outFileName = os.path.join(rootPath, "ripper_base_fixTree_orphanHint_20170307.csv")
#     results = []
#   
#     saveCSV([['filename', 'distance', 'num_node', "score", "num_leaf", "count_contourBoxes", "count_shareBox", "line_width"]], outFileName)
#       
#     for index in range(0, len(fileNameList)):
#   
#         try: 
#   
#             filePath = os.path.join(folderPath, fileNameList[index])
#             print "%s (%d / %d)"%(fileNameList[index], index, len(fileNameList))
#               
#             if isfile(filePath) :
#                 image = cv.imread(filePath,0)    
#               
#       
#             image, w = PhyloParser.resizeImageByLineWidth(image)
#   
#   
#             image_data = ImageData(image)
#             image_data = phyloParser.preprocces(image_data, debug=False)
#   
#             image_data = phyloParser.detectLines(image_data, debug = False)
#             image_data = phyloParser.getCorners(image_data, debug = False)   
#             image_data = phyloParser.makeLinesFromCorner(image_data, debug = False)
#             image_data = phyloParser.includeLinesFromCorners(image_data)
#             image_data = phyloParser.postProcessLines(image_data)
#               
#           
#           
#             image_data = phyloParser.groupLines(image_data)
#             image_data = phyloParser.matchLineGroups(image_data, debug = False)
#               
#   
#             image_data = phyloParser.getSpecies_v3(image_data, debug = False)
#   
#             image_data = phyloParser.constructTree_eval(image_data, fixTree = True, tracing = False, orphanHint = True, debug = False)
#           
#           
#             truth_string = ground_truth[fileNameList[index]]
#             t1 = PhyloTree(image_data.treeStructure + ";")
#             t2 = PhyloTree(truth_string)
#             PhyloTree.rename_node(t1, rename_all=True)
#             PhyloTree.rename_node(t2, rename_all=True)
#               
#             distance = PhyloTree.zhang_shasha_distance(t1, t2)
#             num_node = PhyloTree.getNodeNum(t2) 
#             num_leaf = t2.getLeafCount()
#             score = distance/float(num_node)
#               
#             print "distance %d/%d = %f, leave=%d" %(distance, num_node, score, num_leaf)
#             print "contour count=%d , sharebox count=%d"%(image_data.count_contourBoxes, image_data.count_shareBox)
# #             results.append([fileNameList[index], distance, num_node, score, num_leaf, image_data.count_contourBoxes, image_data.count_shareBox])
#             saveCSV([[fileNameList[index], distance, num_node, score, num_leaf, image_data.count_contourBoxes, image_data.count_shareBox, w]], outFileName, mode='ab')
#         except:
# #             results.append([[fileNameList[index], distance, num_node, score, num_leaf, image_data.count_contourBoxes, image_data.count_shareBox]])
#             saveCSV([[fileNameList[index], -1, -1, -1, -1,-1,-1, -1]], outFileName, mode='ab')
#          
#  
#  
#  
# 
# 
# 
# ############################################ Ripper Dataset ###################################################
# ##---------------------------- fixTree only --------------------------------##
#      
#  
#  
#  
# #     ground_truth_path  = "/Users/sephon/Desktop/Research/VizioMetrics/Corpus/Phylogenetic/CNN_corpus/TreeRipper_dataset/TreeRipper_dataset.csv"
# #     folderPath = '/Users/sephon/Desktop/Research/VizioMetrics/Corpus/Phylogenetic/CNN_corpus/TreeRipper_dataset/images/'
#      
#     ground_truth_path  = os.path.join(rootPath, "TreeRipper_dataset/TreeRipper_multi_dataset.csv")
#     folderPath = os.path.join(rootPath, "TreeRipper_dataset/multi-chart_clip/")
#      
#      
#     ground_truth = {}
#     fileNameList = []
#     with open(ground_truth_path ,'rb') as incsv:
#         reader = csv.reader(incsv, dialect='excel', delimiter='\t')
#         reader.next()
#      
#         for row in reader:
#             ground_truth[row[0]] = row[1]    
#             fileNameList.append(row[0])
#  
#  
#     outFileName = os.path.join(rootPath, "ripper_multi_base_fixTree_orphanHint_20170307.csv")
#     results = []
#  
#     saveCSV([['filename', 'distance', 'num_node', "score", "num_leaf", "count_contourBoxes", "count_shareBox", "line_width"]], outFileName)
#      
#     for index in range(0, len(fileNameList)):
#  
#         try: 
#  
#             filePath = os.path.join(folderPath, fileNameList[index])
#             print "%s (%d / %d)"%(fileNameList[index], index, len(fileNameList))
#              
#             if isfile(filePath) :
#                 image = cv.imread(filePath,0)    
#              
#      
#             image, w = PhyloParser.resizeImageByLineWidth(image)
#  
#             image_data = ImageData(image)
#             image_data = phyloParser.preprocces(image_data, debug=False)
#          
#             image_data = phyloParser.detectLines(image_data, debug = False)
#             image_data = phyloParser.getCorners(image_data, debug = False)   
#             image_data = phyloParser.makeLinesFromCorner(image_data, debug = False)
#             image_data = phyloParser.includeLinesFromCorners(image_data)
#             image_data = phyloParser.postProcessLines(image_data)
#          
#          
#             image_data = phyloParser.groupLines(image_data)
#             image_data = phyloParser.matchLineGroups(image_data, debug = False)
#              
#  
#             image_data = phyloParser.getSpecies_v3(image_data, debug = False)
#  
#             image_data = phyloParser.constructTree_eval(image_data, fixTree = True, tracing = False, orphanHint = True, debug = False)
#          
#          
#             truth_string = ground_truth[fileNameList[index]]
#             t1 = PhyloTree(image_data.treeStructure + ";")
#             t2 = PhyloTree(truth_string)
#             PhyloTree.rename_node(t1, rename_all=True)
#             PhyloTree.rename_node(t2, rename_all=True)
#              
#             distance = PhyloTree.zhang_shasha_distance(t1, t2)
#             num_node = PhyloTree.getNodeNum(t2) 
#             num_leaf = t2.getLeafCount()
#             score = distance/float(num_node)
#              
#             print "distance %d/%d = %f, leave=%d" %(distance, num_node, score, num_leaf)
#             print "contour count=%d , sharebox count=%d"%(image_data.count_contourBoxes, image_data.count_shareBox)
#             saveCSV([[fileNameList[index], distance, num_node, score, num_leaf, image_data.count_contourBoxes, image_data.count_shareBox, w]], outFileName, mode='ab')
#         except:
# #             results.append([[fileNameList[index], distance, num_node, score, num_leaf, image_data.count_contourBoxes, image_data.count_shareBox]])
#             saveCSV([[fileNameList[index], -1, -1, -1, -1,-1,-1, -1]], outFileName, mode='ab')
        


