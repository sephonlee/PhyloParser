# Phylogenetic Tree Parser Tutorial

# from readCorners import readCorners
# import textDetector
# import textRemover
import cv2 as cv
import numpy as np
import joblib
import pickle
from matplotlib import pyplot as plt
from ete3 import Tree
import operator, os
import random
from os import listdir
from os.path import isfile, join
from GroundTruthConverter import *

from PhyloParser import *

def getFilesInFolder(folderPath):
    fileNameList = [f for f in listdir(folderPath) if isfile(join(folderPath, f)) and f.split(".")[-1] == "jpg"]

    return fileNameList

def saveResult(data, output):
    with open(output, 'wb') as outcsv:
        writer = csv.writer(outcsv, dialect='excel')
        header = ['filename', 'distance', 'num_node', "score"]
        writer.writerow(header)
        
        for row in data:
            writer.writerow(row)



def getViewPatch(image, line, size = (10, 40)):
    
    right_end = (line[3], line[2])
    top =  right_end[0]-size[0]/2
    down = right_end[0]+size[0]/2
    left = right_end[1]-15
    right  = right_end[1]+size[1]-15
    
    if top < 0:
        down = down - top
        top = 0
    
    if left < 0:
        right = right - left
        left = 0
        
    if down > image.shape[0]:
        top - (down - image.shape[0])
        down = image.shape[0]
        
    if right > image.shape[1]:
        left - (right - image.shape[1])
        right = image.shape[1]
        
    patch = image[top:down, left:right]
    return patch

def getAvgRightEndPointOfLines(lines):
    x = 0
    if len(lines) > 0:
        for line in lines:
            x += line[2]
        
        avg_x = x/float(len(lines))
    else:
        avg_x = 0
    return avg_x
    
    
# def getDomain(dim, patch_dim, top, bot, left, right):
    
#     if top < 0:
#         
#         np.zeros((-top, ))
#         
#     if bot > dim[0]:
#         
# #     if left < 0:
#     if right > dim[1]
    
    
def getLineFeature(image, line, avg_line_x, hpatch_size = (3,15), vpatch_size = (9,3), x_margin = 7):
    
#     print "line", line

    border = 50
    extend_image = cv.copyMakeBorder(image, border, border, border, border, cv.BORDER_CONSTANT, value = 255)
#     PhyloParser.displayLines(extend_image, [[line[0]+border, line[1]+border, line[2]+border, line[3]+border, line[4]+border]])
    
    right_end = (line[3], line[2])
    hh = hpatch_size[0] / 2
    
#     hpatch = image[right_end[0]-hh:right_end[0]+hh+1, right_end[1] + x_margin: right_end[1] + x_margin + hpatch_size[1]]
    
    hpatch = extend_image[right_end[0]-hh + border:right_end[0]+hh+1 + border, right_end[1] + x_margin + border: border + right_end[1] + x_margin + hpatch_size[1]]
#     print hpatch
#     print hpatch.flatten()
    
    hh = vpatch_size[0] / 2
    vpatch = extend_image[border+right_end[0]-hh:border+right_end[0]+hh+1, border+right_end[1]-vpatch_size[1]: border+right_end[1]]
#     print vpatch
#     print vpatch.T.flatten()
    
    
    distanceToAvgX = (right_end[1] - avg_line_x)/image.shape[1]
    
    feature = np.hstack((hpatch.flatten(), vpatch.T.flatten())) / float(255)
    feature = np.hstack((distanceToAvgX, feature))

    viewPatch = getViewPatch(image, line)
    
#     avgline = [int(avg_line_x), 0, int(avg_line_x), image.shape[0], 0]
#     print avgline
#     PhyloParser.displayLines(image, [avgline])
     
#     PhyloParser.displayImage(viewPatch)
#     PhyloParser.displayImage(hpatch)
#     PhyloParser.displayImage(vpatch)
    

    
    
    return feature, viewPatch

if __name__ == '__main__':
    
    
    phyloParser = PhyloParser()
    
    folderPath = '/Users/sephon/Desktop/Research/VizioMetrics/Corpus/Phylogenetic/CNN_corpus/far_species'
    ground_truth = {}
    fileNameList = []
    
    fileList = []
    for dirPath, dirNames, fileNames in os.walk(folderPath):   
        for f in fileNames:
            extension = f.split('.')[-1].lower()
            if extension in ["jpg", "png"]:
                fileList.append(os.path.join(dirPath, f))
    
    
    
#     fileNameList = getFilesInFolder(folderPath)
    pathFolder = '/Users/sephon/Desktop/Research/VizioMetrics/Corpus/Phylogenetic/CNN_corpus/line_patch'     
#     datasetFlieName = '/Users/sephon/Desktop/Research/VizioMetrics/Corpus/Phylogenetic/CNN_corpus/line_feature.npy'
#     datasetFlieName = '/Users/sephon/Desktop/Research/VizioMetrics/Corpus/Phylogenetic/CNN_corpus/line_feature.npy'
    
    
#     np.save('/Users/sephon/Desktop/Research/VizioMetrics/TSNE/CNN_VizioNet_8cat/AlexNet_feature_CNN_corpus_testset.npy', X)
# np.save('/Users/sephon/Desktop/Research/VizioMetrics/TSNE/CNN_VizioNet_8cat/AlexNet_label_CNN_corpus_testset.npy', label)
# 
# X = np.load('/Users/sephon/Desktop/Research/VizioMetrics/TSNE/CNN_VizioNet_8cat/AlexNet_feature_CNN_corpus_testset.npy')
# label = np.load('/Users/sephon/Desktop/Research/VizioMetrics/TSNE/CNN_VizioNet_8cat/AlexNet_label_CNN_corpus_testset.npy')
     
    data = []
    patchNameList = []

#     fileNameList = ["PMC1978502_1471-2148-7-139-2.jpg"] ## infinite loop

    for index in range(0, len(fileList)):

        filePath = os.path.join(folderPath, fileList[index])
        print 
        print "%s (%d / %d)"%(fileList[index], index, len(fileList))
        
        if isfile(filePath) :
            image = cv.imread(filePath,0)
            
#         PhyloParser.displayImage(image)
        
        image = PhyloParser.resizeImageByLineWidth(image)
        image_data = ImageData(image)
        image_data = phyloParser.preprocces(image_data, debug=False)

        image_data = phyloParser.detectLines(image_data, debug = False)
        image_data = phyloParser.getCorners(image_data, debug = False)   
        image_data = phyloParser.makeLinesFromCorner(image_data, debug = False)
        image_data = phyloParser.includeLinesFromCorners(image_data)
        image_data = phyloParser.postProcessLines(image_data)
        image_data = phyloParser.groupLines(image_data)
        image_data = phyloParser.matchLineGroups(image_data, debug = False)

#         print "show anchor line"
#         PhyloParser.displayLines(image, image_data.anchorLines)

            
      
#         image_data = phyloParser.constructTree(image_data, tracing = False , debug = True)
    
    
    
    
#         PhyloParser.displayImage(image_data.image_preproc)
        avg_line_x = getAvgRightEndPointOfLines(image_data.anchorLines)
        
        for j, line in enumerate(image_data.horLines):
            filename = fileList[index]
            patchName = filename.split("/")[-1][0:-4] + "_horline_resize_" + str(j) + "." + filename.split(".")[-1]
            patchNameList.append([patchName, filename])
#             print [patchName, filename]
            feature, viewPatch = getLineFeature(image_data.image_preproc, line, avg_line_x)
#             print feature

            patchVarience = np.var(feature[0:46])
            
            if patchVarience <= 0.05:
                finalPatchName = os.path.join(pathFolder, "negative", patchName)
            else:
                finalPatchName = os.path.join(pathFolder, "positive", patchName)
            print finalPatchName

            cv.imwrite(finalPatchName, viewPatch)
#             print patchName
#             print filename
#             print feature.tolist()
#             print len(feature.tolist())
            print patchName
            data.append([patchName, filename, feature.tolist()])
#             print data[-1]
        


        joblib.dump(data, '/Users/sephon/Desktop/Research/VizioMetrics/Corpus/Phylogenetic/CNN_corpus/line_patch/farspecies_horline_resize_feature.dat') 
        
#         print "show anchor line"
#         PhyloParser.displayLines(image, image_data.horLines)

        

# np.save('/Users/sephon/Desktop/Research/VizioMetrics/TSNE/CNN_VizioNet_8cat/AlexNet_label_CNN_corpus_testset.npy', label)