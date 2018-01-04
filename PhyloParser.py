from os import listdir
from os.path import isfile, join
import csv
import cv2 as cv
try:
    import Image
except:
    from PIL import Image
    
import numpy as np
import peakutils
import codecs

from sklearn.externals import joblib
from sklearn.cluster import KMeans, MiniBatchKMeans
from skimage.transform import probabilistic_hough_line
from skimage.feature import canny
from scipy import stats


import pytesseract
import time
import math
from cgitb import text
from sets import Set
from itertools import groupby
from operator import itemgetter


from Node import *
from ImageData import *
from TrunkNode import TrunkNode
import GroundTruthConverter





class PhyloParser():
    
    def __init__(self, clfPath = None):
        
        self.classifier = None
        
        if clfPath is not None:
            try:
                print 'Loading classifier from', clfPath
                self.classifier = joblib.load(clfPath)
            except:
                print 'model not loaded', clfPath
            

    # Remove Background
    # Create Mask for text and tree
    # Sharpen Edge
    @staticmethod
    def preprocces(image_data, debug = False):
        
        image = image_data.image

        if debug:
            print "Preprocessing image ..."
            print "Input image:"
            PhyloParser.displayImage(image)

        #save original image
        image_data.originalImage = image.copy() 
        
        # Remove background
        image, edgeMask, hasBackground= PhyloParser.removeBackground(image)
        if debug:
            print "Removed background"
            PhyloParser.displayImage(image)
        
        # Separate tree and text
        image_data.treeMask, image_data.nonTreeMask, image_data.contours, image_data.hierarchy = PhyloParser.findContours(255 - PhyloParser.negateImage(image)) 
        if debug:
            print "display tree mask"
            PhyloParser.displayImage(image_data.treeMask)
            print "display non-tree mask"
            PhyloParser.displayImage(image_data.nonTreeMask)
        
        # Sharpen edges
        image = PhyloParser.bilateralFilter(image)
        if debug:
            print "bilateralFilter image"
            PhyloParser.displayImage(image)


        image_data.image_preproc = image
        image_data.hasBackground = hasBackground
        image_data.preprocessed = True

        return image_data
        
        
    ## static method for preprocessing ##

    @staticmethod
    def removeBackground(image):
        hist1, bins = np.histogram(image.ravel(),256,[0,256])

        CANNY_THRESH_1 = 100
        CANNY_THRESH_2 = 200

        edges = cv.Canny(image, CANNY_THRESH_1, CANNY_THRESH_2)
        kernel = np.ones((5,5),np.uint8)
        edges = cv.morphologyEx(edges, cv.MORPH_CLOSE, kernel)
        newImage = image.copy()
        newImage[np.where(edges != 255)] = 255        
        hist2, bins = np.histogram(newImage.ravel(),256,[0,256])


        hasBackground, backgroundList = PhyloParser.findBackgroundPixelPeak(hist1, hist2, bins)
        if hasBackground:
            newnewImage = np.zeros(image.shape,np.uint8)
            for start, end in backgroundList:
                newnewImage[np.where((newImage>=start) & (newImage<end))] = 255

            testing = newImage.copy()

            horLines, verLines = PhyloParser.detectLines_light(newnewImage)
            for line in horLines:
                x1, y1, x2, y2, length = line
                newnewImage[y1:y2+1, x1:x2] = 0
            for line in verLines:
                x1, y1, x2, y2, length = line
                newnewImage[y1:y2, x1:x2+1] = 0

            testing[np.where(newnewImage==255)] = 255
            kernel = np.ones((3,3),np.uint8)
            image = testing

        return image, edges, hasBackground
    

    @staticmethod
    def findBackgroundPixelPeak(hist1, hist2, bins, peakThreshold = 0.01, minDistThreshold = 1, peakRangeThreshold = 0.01):
        diff = hist2 - hist1
        sumDiff = -sum(diff[0:255])
        diff = - ((diff+0.0)/sumDiff)
        maxPeakValue = max(diff)

        peakThreshold = (maxPeakValue + 0.0) / 10
        backgroundList = []
        bins = bins[0:255]
        diff = diff[0:255]
   
        indexes = peakutils.indexes(diff, thres=peakThreshold, min_dist=minDistThreshold)
        # print indexes
        tmp = []
        for index in indexes:
            if index > 10 and index < 250:
                tmp.append(index)
        indexes = tmp[:]
        # slope, intercept, r_value, p_value, std_err = stats.linregress(bins,diff)
        # print slope, intercept

        for index in indexes:
            peakRange = [index, index]
            thres = diff[index] * peakRangeThreshold
            stack = []
            stack.append(index+1)
            while stack:
                pos = stack.pop()
                if diff[pos] > thres and diff[pos]<diff[pos-1]:
                    peakRange[1] = pos
                    if pos+1<250:
                        stack.append(pos+1)
            stack.append(index-1)
            while stack:
                pos = stack.pop()
                if diff[pos] > thres and diff[pos]<diff[pos+1]:
                    peakRange[0] = pos
                    stack.append(pos-1)

            peakRange[1] +=1
            if abs(diff[peakRange[0]] - diff[peakRange[1]]) < (maxPeakValue + 0.0) / 100 :
                backgroundList.append(tuple(peakRange))
            elif peakRange[0] == index and abs(diff[peakRange[0] - 1] - diff[peakRange[1]]) < (maxPeakValue + 0.0) / 100:
                backgroundList.append(tuple(peakRange))

        return len(backgroundList) > 0, backgroundList
    
#     @staticmethod
#     def sobelFilter(image, k=5, sigma = 3):
#         image  = PhyloParser.gaussianBlur(image, (k,k), sigma)
#         sobelx = cv.Sobel(image, cv.CV_64F, 1, 0, ksize = k)
#         sobely = cv.Sobel(image, cv.CV_64F, 0, 1, ksize = k)
#         image = cv.addWeighted(sobelx, 0.5, sobely, 0.5, 0)
#         return image
    
#     @staticmethod 
#     def gaussianBlur(image, kernelSize = (3,3), sigma= 1.0):
#         image = cv.GaussianBlur(image, kernelSize, sigma)
#         return image

    @staticmethod 
    ## Old bolding, duplicate with threshold
    def binarize(image, thres=180, mode = 0):
        ret, image = cv.threshold(image, thres, 255, mode)
        return image
    
    @staticmethod
    def bilateralFilter(image, radius = 3, sigmaI = 30.0, sigmaS = 3.0):
        image = cv.bilateralFilter(image, radius, sigmaI, sigmaS)
        return image

    @staticmethod
    def downSample(image, parameter = 500.0):
        height, width = image.shape
        if height>700:
            if height > 1000:
                parameter = 700.0
            ratio  = (parameter + 0.0) / height
            image = cv.resize(image, (0,0), fx = ratio, fy=ratio)

        return image
        
    @staticmethod
    def displayImage(image):
        plt.imshow(image, cmap='Greys_r')
        plt.show()


    @staticmethod
    def resizeImageByLineWidth(image, linewidth = 4):
        w = PhyloParser.findLineWidth(image)        
        if abs(w) > linewidth :
            ratio = linewidth / abs(float(w))
            image = cv.resize(image,(0,0), fx = ratio, fy = ratio, interpolation = cv.INTER_CUBIC)
        return image, w

    @staticmethod
    # return line width of the tree
    # approach: 
    # 1. use contour finder to separate tree
    # 2. Iteratively Use opening until detecting a change of overall pixel values 
    def findLineWidth(image, upper_bond = 15):
        mask, nonTreeMask, contours, hierarchy = PhyloParser.findContours(255 - PhyloParser.negateImage(image)) 
        
        linewidth = 1
        original_pixel_sum = np.sum(mask)
                
        while linewidth <= upper_bond:
            
            kernel = np.ones((linewidth,linewidth), np.uint8)
            result = cv.morphologyEx(mask, cv.MORPH_OPEN, kernel)
            pixel_sum = np.sum(result)

            if pixel_sum == 0 or original_pixel_sum / pixel_sum >= 2:
                break
            else:
                linewidth += 1
        

        if linewidth - 1 == upper_bond:
            return -1
        else:
            return linewidth - 1

    @staticmethod
    # return a mask of the tree, a mask of text and contours
    def findContours(var_mask1, var_mask2 = None):

        var_mask1 = 255 - var_mask1

        height, width = var_mask1.shape
        var_mask1 = cv.copyMakeBorder(var_mask1, 1, 1, 1, 1, cv.BORDER_CONSTANT, value = 0)
        _, contours, hierarchy= cv.findContours(var_mask1.copy(), cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
        
        lenghtList = []
        for cnts in contours:
            lenghtList.append(len(cnts))
            for index, points in enumerate(cnts):
                cnts[index] = points - 1 #shift back (because of padding)

        # label the largest contour as the tree
        maxValue = 0
        maxIndex = 0
        for index, cnts in enumerate(contours):
            if len(cnts)> maxValue:
                maxIndex = index
                maxValue = len(cnts)

        
        mask = np.zeros((height,width), dtype=np.uint8)
        cv.drawContours(mask, contours, maxIndex, (255), thickness = -1, hierarchy = hierarchy, maxLevel = 1)
        
#         print "mask"
#         PhyloParser.displayImage(mask)
        
        kernel = np.ones((5,5),np.uint8)
        tmpMask = cv.morphologyEx(mask, cv.MORPH_CLOSE, kernel)

        compensateMask = np.zeros((height, width), dtype = np.uint8)
        cv.drawContours(compensateMask, contours, maxIndex, (255), thickness = -1)

        nonTreeMask = np.zeros((height, width), dtype = np.uint8)
        cv.drawContours(nonTreeMask, contours, -1, (255), thickness = -1)
        nonTreeMask[np.where(compensateMask == 255)] = 0
        cv.drawContours(nonTreeMask, contours, -1, (255), thickness = -1, hierarchy = hierarchy, maxLevel = 5)
        nonTreeMask[np.where(tmpMask == 255)] = 0
        
        return mask, nonTreeMask, contours, hierarchy

    @staticmethod
    def removeLabels(image, mask):
        image[np.where(mask == 0)] = 255
        return image
    


    @staticmethod
    def postProcessLines(image_data):

        image_data.horLines = PhyloParser.cutLines(image_data.horLines, image_data.verLines)

        horLines = PhyloParser.purifyLines(image_data.horLines, image_data.image_preproc_for_line_detection, PhyloParser.negateImage(image_data.image_preproc_for_line_detection), 'hor')
        verLines = PhyloParser.purifyLines(image_data.verLines, image_data.image_preproc_for_line_detection, PhyloParser.negateImage(image_data.image_preproc_for_line_detection), 'ver')
       
        image_data.horLines = horLines
        image_data.verLines = verLines

        return image_data

    # Group lines that represents the same branches
    @staticmethod
    def groupLines(image_data, debug = False):
        horLines = image_data.horLines[:]
        verLines = image_data.verLines[:]
        if image_data.preprocessed:    
            image = image_data.image_preproc.copy()
        else:
            image = image_data.image.copy()
        height, width = image.shape

        horLineMask = np.zeros((height, width, 2), dtype = int)
        verLineMask = np.zeros((height, width, 2), dtype = int)

        horLineMappingDict = {}
        horLineMappingDict['lineMapping'] = {}
        horLineMappingDict['overlapMapping'] = {}
        verLineMappingDict = {}
        verLineMappingDict['lineMapping'] = {}
        verLineMappingDict['overlapMapping'] = {} 

        image_data.horLines, image_data.horLineMask, image_data.horLineMappingDict = PhyloParser.getUniqueLinesList(horLines, horLineMask, horLineMappingDict, image_data, mode = 'hor')
        image_data.verLines, image_data.verLineMask, image_data.verLineMappingDict = PhyloParser.getUniqueLinesList(verLines, verLineMask, verLineMappingDict, image_data, mode = 'ver')
        image_data.lineGrouped = True

        # print image_data.getLineGroupGivenDot((492,23), 'ver')
        # print image_data.verLineMappingDict
                # PhyloParser.displayLines(image_data.image, image_data.verLineMappingDict['lineMapping'][image_data.getLineGroupGivenDot((388,320), 'ver')]['lineGroup'])


        if debug:
            print 'groupLines debugging'
            image_data.displayTargetLines('horLines')
            image_data.displayTargetLines('verLines')



        # for key, value in horLineMappingDict.items():
        #     if key == 'overlapMapping':
        #         print valuef
        #     elif key == 'lineMapping':
        #         for indexKey, mapValue in value.items():
        #             print '----------------------------'
        #             print 'Index = ',indexKey
        #             print 'length = ', mapValue['length']
        #             print 'overlap = ', mapValue['overlap']
        #             print 'linegroup: ', mapValue['lineGroup']
        #             print 'numNoiseGroup:' , len(mapValue['noise'])
        #             print 'noiseIndexes:', 
        #             for key, whatever in mapValue['noise'].items():
        #                 print key,
        #             print
        # PhyloParser.displayImage(horLineMask[:,:,0])
        # PhyloParser.displayImage(horLineMask[:,:,1])

        # PhyloParser.displayImage(image_data.horLineMask[:,:,0])
        # PhyloParser.displayImage(image_data.verLineMask[:,:,0])

        return image_data


    ### State methods for group lines ###

    @staticmethod
    def isLineNoisy(length, aveLength):
        if length < aveLength -5:
            return True
        else:
            return False


    @staticmethod
    def getMidPoint(line):
        x1, y1, x2, y2, length = line
        return ((y1+y2)/2, (x1+x2)/2)


    @staticmethod
    def getUniqueLinesList(lineList, mask, mappingDict, image_data, mode):
        if mode == 'hor':
            lineList = sorted(lineList, key = lambda x: x[1])
        elif mode == 'ver':
            lineList = sorted(lineList, key = lambda x: x[0])


        image_lineDetection = image_data.image_preproc_for_line_detection    
        mapIndex = 1
        overlapIndex = 1

        for line in lineList:

            x1, y1, x2, y2, length = line
            if mode == 'hor':
                overlapNumber = PhyloParser.getVoteNumber(mask[y1:y2+1, x1:x2, 1], mode = 'overlap')
            elif mode == 'ver':
                overlapNumber = PhyloParser.getVoteNumber(mask[y1:y2, x1:x2+1, 1], mode = 'overlap')
            if overlapNumber == 0:
                if mode == 'hor':
                    voteNumber=PhyloParser.getVoteNumber(mask[y1:y2+1, x1:x2, 0])
                elif mode == 'ver':
                    voteNumber = PhyloParser.getVoteNumber(mask[y1:y2, x1:x2+1, 0])
                if voteNumber == 0:
                    mappingDict['lineMapping'][mapIndex] = {}
                    mappingDict['lineMapping'][mapIndex]['length'] = length
                    mappingDict['lineMapping'][mapIndex]['lineGroup'] = [line]
                    mappingDict['lineMapping'][mapIndex]['overlap'] = []
                    overlapIndex, mappingDict = PhyloParser.drawLine_lineVersion(line, mappingDict, mask, mapIndex, overlapIndex, mode = mode)
                    mapIndex +=1
                else:
                    if PhyloParser.isSameLineGroup(line, mappingDict['lineMapping'][voteNumber]['lineGroup']):
                        if not PhyloParser.isLineNoisy(length, mappingDict['lineMapping'][voteNumber]['length']):
                            mappingDict['lineMapping'][voteNumber]['lineGroup'].append(line)
                            PhyloParser.updateAverageLength(mappingDict, voteNumber, line)

                            overlapIndex, mappingDict = PhyloParser.drawLine_lineVersion(line, mappingDict, mask, voteNumber, overlapIndex, mode = mode)
                    else:
                        
                        mappingDict['lineMapping'][mapIndex] = {}
                        mappingDict['lineMapping'][mapIndex]['length'] = length
                        mappingDict['lineMapping'][mapIndex]['lineGroup'] = [line]
                        mappingDict['lineMapping'][mapIndex]['overlap'] = [overlapIndex]
                        if overlapIndex not in mappingDict['lineMapping'][voteNumber]['overlap']:
                            mappingDict['lineMapping'][voteNumber]['overlap'].append(overlapIndex)
                        mappingDict['overlapMapping'][overlapIndex] = [mapIndex, voteNumber]

                        overlapIndex, mappingDict = PhyloParser.drawLine_lineVersion(line, mappingDict, mask, mapIndex, overlapIndex+1, mode = mode, overlapIndex = overlapIndex)
                        mapIndex+=1
            else:
                isFound = False

                for lineIndex in mappingDict['overlapMapping'][overlapNumber]:                 
                    if PhyloParser.isSameLineGroup(line, mappingDict['lineMapping'][lineIndex]['lineGroup']):
                        if not PhyloParser.isLineNoisy(length, mappingDict['lineMapping'][lineIndex]['length']):
                            mappingDict['lineMapping'][lineIndex]['lineGroup'].append(line)
                            PhyloParser.updateAverageLength(mappingDict, lineIndex, line)
                            overlapIndex, mappingDict = PhyloParser.drawLine_lineVersion(line, mappingDict, mask, lineIndex, overlapIndex, mode, overlapIndex = overlapNumber)
                        isFound = True

                        

                if not isFound:
                    mappingDict['lineMapping'][mapIndex] = {}
                    mappingDict['lineMapping'][mapIndex]['length'] = length
                    mappingDict['lineMapping'][mapIndex]['lineGroup'] = [line]
                    mappingDict['lineMapping'][mapIndex]['overlap'] = [overlapNumber]
                    mappingDict['overlapMapping'][overlapNumber].append(mapIndex)
                    overlapIndex, mappingDict = PhyloParser.drawLine_lineVersion(line, mappingDict, mask, mapIndex, overlapIndex, mode, overlapIndex = overlapNumber)
                    mapIndex +=1
        lineList = []
        noiseIndexes = []
        noiseOverlapIndexes = []


        for lineIndex, lineDict in mappingDict['lineMapping'].items():

            lineDict['noise'] = {}
            lineDict['parent'] = []
            lineDict['children'] = []
            lineDict['type'] = None
            
            ## new version ####
#             if mode =='hor':
#                 lineDict['lineGroup'] = sorted(lineDict['lineGroup'], key= lambda x: x[1])
#                 lineDict['rline_upper'] = lineDict['lineGroup'][0]
#                 lineDict['rline_lower'] = lineDict['lineGroup'][-1]
#                 mmax = lineDict['lineGroup'][-1][1]
#                 mmin = lineDict['lineGroup'][0][1]
#                 x1, y1, x2, y2, length = lineDict['lineGroup'][len(lineDict['lineGroup'])/2]
#                 lineDict['midPoint'] = ((mmax+mmin)/2, (x1+x2)/2)
#                 lineDict['rline'] = (x1, (mmax+mmin)/2, x2, (mmax+mmin)/2, length)
#             else:
#                 lineDict['lineGroup'] = sorted(lineDict['lineGroup'], key= lambda x: x[0])
#                 targetIndex = len(lineDict['lineGroup'])/2
#                 targetLine = lineDict['lineGroup'][targetIndex]
#                 x1, y1, x2, y2, length = targetLine
#                 lineDict['midPoint'] = ((y1+y2)/2, (x1+x2)/2)
#                 lineDict['rline'] = targetLine
                
                ## new worse version
#                 lineDict['rline'] = lineDict['lineGroup'][-1]
#                 mmax = lineDict['lineGroup'][-1][0]
#                 mmin = lineDict['lineGroup'][0][0]
#                 lineDict['midPoint'] = ((y1+y2)/2, (mmax + mmin)/2)


            ##### old version ######
            if mode =='hor':
                lineDict['lineGroup'] = sorted(lineDict['lineGroup'], key= lambda x: x[1])
            else:
                lineDict['lineGroup'] = sorted(lineDict['lineGroup'], key= lambda x: x[0])
                
            targetIndex = len(lineDict['lineGroup'])/2
            targetLine = lineDict['lineGroup'][targetIndex]
            x1, y1, x2, y2, length = targetLine

            lineDict['midPoint'] = ((y1+y2)/2, (x1+x2)/2)
            lineDict['rline'] = targetLine
            lineDict['rline_upper'] = targetLine
            lineDict['rline_lower'] = targetLine



        for lineIndex, lineDict in mappingDict['lineMapping'].items():
            isNoise = False
            if lineIndex not in noiseIndexes:
                noisePool = []
                for overlapMappingIndex in mappingDict['lineMapping'][lineIndex]['overlap']:
                    
                    for overlapIndex in mappingDict['overlapMapping'][overlapMappingIndex]:
                        if lineIndex != overlapIndex:
                            ox1, oy1, ox2, oy2, olength = mappingDict['lineMapping'][overlapIndex]['rline']
                            tx1, ty1, tx2, ty2, tlength = mappingDict['lineMapping'][lineIndex]['rline']

                            if mode == 'hor':
                                if ox1 >= tx1 - tlength/10 and ox2 <= tx2 + tlength/10:
                                    midPoint = (ty1+oy1) / 2
                                    pixelValues = image_lineDetection[midPoint:midPoint + 1,ox1:ox2]

                                    values, counts = np.unique(pixelValues, return_counts=True)
                                    pixelValueDict = dict(zip(values, counts))
                                    blackPts = 0
                                    allPts = sum(pixelValueDict.values())
                                    for key, count in pixelValueDict.items():
                                        if key < 235:
                                            blackPts +=count

                                    if blackPts > allPts/2:
                                        isNoise = True
                                        if overlapIndex not in noisePool:
                                            noisePool.append(overlapIndex)
                            elif mode == 'ver':
                                if oy1 >= ty1 - tlength/10 and oy2 <= ty2 + tlength/10:
                                    # if lineIndex == 17 and overlapIndex == 15:
                                    #     print oy1, ty1, oy2, ty2
                                    midPoint = (tx1+ox1)/2
                                    pixelValues = image_lineDetection[oy1:oy2,midPoint :midPoint +1]

                                    values, counts = np.unique(pixelValues, return_counts=True)
                                    pixelValueDict = dict(zip(values, counts))
                                    
                                    blackPts = 0
                                    allPts = sum(pixelValueDict.values())
                                    for key, count in pixelValueDict.items():
                                        if key < 235:
                                            blackPts +=count
                                    
                                    if blackPts>allPts/2:
                                        isNoise = True
                                        if overlapIndex not in noisePool:
                                            noisePool.append(overlapIndex)

                if isNoise:
                    overlapIndexCheckList = []

                    for noiseIndex in noisePool:
                        noiseIndexes.append(noiseIndex)
                        lineDict['noise'][noiseIndex] = mappingDict['lineMapping'][noiseIndex].copy()
                        for subNoiseIndex, subnoises in mappingDict['lineMapping'][noiseIndex]['noise'].items():
                            lineDict['noise'][subNoiseIndex] = subnoises.copy()

                        mask[PhyloParser.mapping2Dto3D(np.where(mask[:,:,0] == noiseIndex), 0)] = lineIndex
                        for overlapIndex in mappingDict['lineMapping'][noiseIndex]['overlap']:
                            if noiseIndex in mappingDict['overlapMapping'][overlapIndex]:
                                mappingDict['overlapMapping'][overlapIndex].remove(noiseIndex)
                            overlapIndexCheckList.append(overlapIndex)


                    if len(overlapIndexCheckList) >0:
                        for overlapMappingIndex in overlapIndexCheckList:
                            
                            if len(mappingDict['overlapMapping'][overlapMappingIndex])<2:
                                mask[PhyloParser.mapping2Dto3D(np.where(mask[:,:,1] == overlapMappingIndex), 1)] = 0
                                for overlapIndex in mappingDict['overlapMapping'][overlapMappingIndex]:
                                    if overlapMappingIndex in mappingDict['lineMapping'][overlapIndex]['overlap']:
                                        mappingDict['lineMapping'][overlapIndex]['overlap'].remove(overlapMappingIndex)
                                if overlapMappingIndex not in noiseOverlapIndexes:
                                    noiseOverlapIndexes.append(overlapMappingIndex)
                                
                                # print mappingDict
        

        # if mode == 'hor':
        #     print '----------------------------Before deleting-------------------------'
        #     print mappingDict

        # print noiseIndexes
        # print noiseOverlapIndexes


        for noiseIndex in noiseIndexes:
            del mappingDict['lineMapping'][noiseIndex]

        for noiseIndex in noiseOverlapIndexes:
            del mappingDict['overlapMapping'][noiseIndex]

        for lineIndex, lineDict in mappingDict['lineMapping'].items():
            lineList.append(lineDict['rline'])


        return lineList, mask, mappingDict

    @staticmethod
    def getVoteNumber(mask, mode = None):

        hist1, bins = np.histogram(mask.ravel(),256,[0,256])

        if mode == 'overlap':
            nonzeroIndexes = np.nonzero(hist1)[0]
            for nonzeroIndex in nonzeroIndexes:
                if nonzeroIndex != 0:
                    return nonzeroIndex
            return 0

        voteNumber = np.argmax(hist1)
        return voteNumber


    @staticmethod
    def updateAverageLength(mappingDict, mapIndex, line):
        x1, y1, x2, y2, length = line
        countLength = len(mappingDict['lineMapping'][mapIndex]['lineGroup']) 
        sumLength = mappingDict['lineMapping'][mapIndex]['length'] * (countLength - 1)
        mappingDict['lineMapping'][mapIndex]['length'] = (sumLength + length) / countLength

        return mappingDict
 

    @staticmethod
    def isSameLineGroup(line, lineGroup):
        rline = [0,0,0,0,0]
        count = 0
        for gline in lineGroup:
            count +=1
            for index, pixel in enumerate(gline):
                rline[index] +=pixel
        for i in range(len(rline)):
            rline[i] = rline[i] / count
        rline = tuple(rline)

        return PhyloParser.isSameLine(line, rline)

    @staticmethod
    def isSameLine(aline, bline, margin = 5):
        ax1, ay1, ax2, ay2, alength = aline
        bx1, by1, bx2, by2, blength = bline

        if ay1 - margin < by1 and ay2 + margin> by2 and ax1 - margin < bx1 and ax2 + margin > bx2 and alength + margin > blength:
            return True
        elif by1 - margin < ay1 and by2 + margin > ay2 and bx1 - margin < ax1 and bx2 + margin > ax2 and blength + margin > alength:
            return True
        else:
            return False        
        

    @staticmethod
    def drawLine_lineVersion(line, mappingDict, mask, mapIndex, newOverlapIndex, mode, overlapIndex = None):

        x1, y1, x2, y2, length = line
        drawMask = PhyloParser.getLineCoverRange_lineVersion(line, mask, mode, margin = 3)

        # if mapIndex == 9:
        #     hist1, bins = np.histogram(drawMask.ravel(),256,[0,256])
        #     print hist1
        #     print np.where((drawMask != 0) & (mask[:,:,1] != 0))

        #     PhyloParser.displayImage(drawMask)
        #     PhyloParser.displayImage(mask[:,:,0])
        #     PhyloParser.displayImage(mask[:,:,1])

        # if mode == 'hor' and newOverlapIndex == 5:
        #     print mappingDict
        #     print mapIndex, overlapIndex, newOverlapIndex
        # if mode == 'hor' and overlapIndex == 5:
        #     print mappingDict
        #     print mapIndex, overlapIndex, newOverlapIndex
        #     print mappingDict['overlapMapping'][overlapIndex]

        # if mode == 'hor' and newOverlapIndex == 6:
        #     print mappingDict
        #     print mapIndex, overlapIndex, newOverlapIndex
        # if mode == 'hor' and overlapIndex == 6:
        #     print mappingDict
        #     print mapIndex, overlapIndex, newOverlapIndex
        #     print mappingDict['overlapMapping'][overlapIndex]


        overlapCoverRange = np.where(drawMask == 155)
        isCovered = np.where(drawMask == 205)
        basicDrawRange = PhyloParser.mapping2Dto3D(np.where(drawMask == 255), 0)
        mask[basicDrawRange] = mapIndex

            
        if len(isCovered[0]) != 0:
            if overlapIndex == None:
                drawRange = PhyloParser.mapping2Dto3D(isCovered, 0)
                countCoveredIndexes = np.bincount(mask[drawRange])
                nonzeroIndexes = np.nonzero(countCoveredIndexes)[0]
                seen = [] 
                for coverIndex in nonzeroIndexes:
                    if coverIndex != mapIndex:
                        isArchived = False
                        for existedOverlapIndex in mappingDict['lineMapping'][mapIndex]['overlap']:
                            if coverIndex in mappingDict['overlapMapping'][existedOverlapIndex]:
                                isArchived = True

                        if not isArchived:
                            mappingDict['overlapMapping'][newOverlapIndex] = [mapIndex, coverIndex]
                            mappingDict['lineMapping'][mapIndex]['overlap'].append(newOverlapIndex)
                            mappingDict['lineMapping'][coverIndex]['overlap'].append(newOverlapIndex)             
                            drawRange = PhyloParser.mapping2Dto3D(np.where((drawMask == 205) & (mask[:,:,0] == coverIndex)), 1)
                            mask[drawRange] = newOverlapIndex
                            newOverlapIndex+=1


            else:
                drawRange = PhyloParser.mapping2Dto3D(isCovered, 0)
                countCoveredIndexes = np.bincount(mask[drawRange])
                nonzeroIndexes = np.nonzero(countCoveredIndexes)[0]

                for coverIndex in nonzeroIndexes:
                    if coverIndex != mapIndex:
                        if coverIndex in mappingDict['overlapMapping'][overlapIndex]:
                            drawRange = PhyloParser.mapping2Dto3D(np.where((drawMask == 205) & (mask[:,:,0] == coverIndex)), 1)                        
                            mask[drawRange] = overlapIndex
                        else:

                            mappingDict['overlapMapping'][newOverlapIndex] = [coverIndex, mapIndex]
                            mappingDict['lineMapping'][mapIndex]['overlap'].append(newOverlapIndex)
                            mappingDict['lineMapping'][coverIndex]['overlap'].append(newOverlapIndex)                         
                            drawRange = PhyloParser.mapping2Dto3D(np.where((drawMask == 205) & (mask[:,:,0] == coverIndex)), 1)
                            mask[drawRange] = newOverlapIndex
                            newOverlapIndex +=1                       




        if len(overlapCoverRange[0]) != 0:

            drawRange = PhyloParser.mapping2Dto3D(overlapCoverRange, 1)
            countCoveredIndexes = np.bincount(mask[drawRange])
            nonzeroIndexes = np.nonzero(countCoveredIndexes)[0]                


            for nonzeroIndex in nonzeroIndexes:                
                if mapIndex not in mappingDict['overlapMapping'][nonzeroIndex]:
                    mappingDict['overlapMapping'][newOverlapIndex] = [mapIndex]
                    mappingDict['lineMapping'][mapIndex]['overlap'].append(newOverlapIndex)
                    mappingDict['overlapMapping'][newOverlapIndex] += mappingDict['overlapMapping'][nonzeroIndex]
                    for walalalaIndex in mappingDict['overlapMapping'][nonzeroIndex]:
                        mappingDict['lineMapping'][walalalaIndex]['overlap'].append(newOverlapIndex)
                    drawRange = PhyloParser.mapping2Dto3D(np.where((drawMask == 155) & (mask[:,:,1] == nonzeroIndex)), 1)
                    mask[drawRange] = newOverlapIndex
                    newOverlapIndex+=1
                else:
                    drawRange = PhyloParser.mapping2Dto3D(np.where((drawMask == 155) & (mask[:,:,1] == nonzeroIndex)), 1)
                    mask[drawRange] = nonzeroIndex

          
            # else:
            #     print 'wahaha'
                # drawRange = PhyloParser.mapping2Dto3D(overlapRange, 1)
                # countCoveredIndexes = np.bincount(drawRange[0])
                # nonzeroIndexes = np.nonzero(countCoveredIndexes)[0]
                # for coverIndex in nonzeroIndexes:
                #     if coverIndex != overlapIndex:


        return newOverlapIndex, mappingDict
    
    
    @staticmethod
    def mapping2Dto3D(oriRange, dim, mode = None):
        oriRange = list(oriRange)
        lenRange = len(oriRange[0])
        if mode == 'modify':
            oriRange.pop(-1)

        if dim == 0:
            thirdDim = np.zeros(lenRange, dtype = np.uint8)
        elif dim == 1:
            thirdDim = np.ones(lenRange, dtype = np.uint8)

        oriRange.append(thirdDim)
        oriRange = tuple(oriRange)
        return oriRange


        

    @staticmethod
    def getLineCoverRange_lineVersion(line, lineMask, mode, margin = 3):
        shape = lineMask.shape
        height, width, dimension = shape
        mask = np.zeros((shape[0], shape[1]), dtype = np.uint8)
        nonzeroMask_basic = np.zeros((shape[0], shape[1]), dtype = np.uint8)
        nonzeroMask_overlap = np.zeros((shape[0], shape[1]), dtype = np.uint8)
        x1, y1, x2, y2, length = line
        if mode == 'hor':
            xStart = x1
            xEnd = x2
            yStart = y1-margin
            yEnd = y2+1+margin

            if yStart < 0:
                yStart = 0
            if yEnd >=height:
                yEnd = height -1

        elif mode == 'ver':
            xStart = x1-margin
            xEnd = x2+margin+1
            yStart = y1
            yEnd = y2

            if xStart<0:
                xStart = 0
            if xEnd >= width:
                xEnd = width-1


        mask[yStart:yEnd, xStart:xEnd] = 255
        mask[np.where((mask == 255) & (lineMask[:,:,0] !=0))] -= 50
        mask[np.where((mask !=0) & (lineMask[:,:,1] !=0))] -=50

        return mask

 
    ### End static methods for groupLines ####
 
 
    # Used in removeBackground
    @staticmethod
    def detectLines_light(image, debug = False, heightThreshold = 500):
        
        # find vertical lines
        mode = 0
        height, width = image.shape
        oriImage = image.copy()
        
        minVerLine, minHorLine = PhyloParser.getLineThreshold(image)
        verLines = PhyloParser.getLines(image, mode, minLength = minVerLine)

        # find horizontal lines
        image = PhyloParser.rotateImage(image)
        mode = 1
        horLines = PhyloParser.getLines(image, mode, minLength = minHorLine)

        # split horizontal lines that cut by vertical lines
        # to solve the problem of Poseidon trigeminal stick
        
        if debug:
            PhyloParser.displayLines(oriImage, verLines)
            PhyloParser.displayLines(oriImage, horLines)

        return horLines, verLines
   


    @staticmethod 
    def getLineThreshold(image):
        height, width = image.shape
#         if height>heightThreshold:
#             ratio = (height - heightThreshold) * 3 / 200
#             minVerLine = 12 + ratio
#             minHorLine = 7 + ratio
#         else:
        minVerLine = 8 + height / 100
        minHorLine = 3 + width / 100
        return minVerLine, minHorLine

    @staticmethod
    def detectLines(image_data, debug = False, heightThreshold = 500):
        
        # use preprocessed image
        if image_data.preprocessed:    
            image = image_data.image_preproc.copy()
        else:
            image = image_data.image.copy()

        # PhyloParser.displayImage(image)
        # # sub-preprocessing 
        # image = PhyloParser.binarize(image, thres = 180, mode = 3)
        if debug:
            print "detecting lines ..."
            print "binerized image"
            PhyloParser.displayImage(image)

        # image = PhyloParser.negateImage(cv.Canny(image,100,200))
        # PhyloParser.displayImage(image)
        
        # save the preprocessed image into image_data
        image_data.image_preproc_for_line_detection = image

        # remove text information
        if image_data.treeMask is not None:
            print "Found available tree mask! Applied the tree mask"
            image = PhyloParser.removeLabels(image, image_data.treeMask)

        # binarization
        image = PhyloParser.negateImage(image)

        # find vertical lines
        mode = 0
        minVerLine, minHorLine = PhyloParser.getLineThreshold(image_data.image)
        image_data.verLines = PhyloParser.getLines(image, mode, minLength = minVerLine)


        # find horizontal lines
        image = PhyloParser.rotateImage(image)
        mode = 1
        image_data.horLines = PhyloParser.getLines(image, mode, minLength = minHorLine)

        # split horizontal lines that cut by vertical lines
        # to solve the problem of Poseidon trigeminal stick

        if debug:
            print "detectLines debugging"
            image_data.displayTargetLines('horLines')
            image_data.displayTargetLines('verLines')
        image_data.lineDetected = True # Update image_data status
        return image_data
   
        
    ## static method for detectLine ##removeTextoveText
    @staticmethod
    def purifyLines(lineList, image, negatedImage,  mode,  varianceThreshold=60):
        varList = []
        varList_ = []
        aveList = []
        aveList_ = []

        for line in lineList:
            x1, y1, x2, y2, length = line
            if mode == 'hor':
                var = np.sqrt(np.var(image[y1:y2+1, x1:x2]))
                ave = np.mean(image[y1:y2+1, x1:x2])
            if mode == 'ver':
                var = np.sqrt(np.var(image[y1:y2, x1:x2+1]))
                ave = np.mean(image[y1:y2, x1:x2+1])

            if var < varianceThreshold and not PhyloParser.isInRegion(line, negatedImage, mode):
                varList.append(line)
            else:
                varList_.append(line)
            # varList.append(var)
            # aveList.append(ave)
        # varList = sorted(varList, key = lambda x: -x)
        # aveList = sorted(aveList, key = lambda x: -x)
        # PhyloParser.displayLines(image, varList)
        # PhyloParser.displayLines(image, varList_)

        # print varList


        return varList
    @staticmethod
    def isInRegion(line, negatedImage, mode, thres1= 0.8, thres2 = 0.5, marginRatio = 30, consecutiveThres = 10):
        height, width = negatedImage.shape
        # PhyloParser.displayImage(negatedImage)
        x1, y1, x2, y2, length = line
        if mode == 'hor':
            result = []
            margin = int(height/marginRatio)

            for index in range(margin):

                region = negatedImage[y1 + index:y2 + index +1, x1:x2]
                lines = [line]
                # PhyloParser.displayLines(negatedImage, lines)
                indices = np.where(region == 255)
                isWhite = (len(indices[0])+0.0)/(x2-x1)
                # print isWhite
                if isWhite>thres1:
                    result.append(True)
                else:
                    result.append(False)
            result.reverse()
            for index in range(1,margin):

                region = negatedImage[y1 - index:y2 - index +1, x1:x2]
                indices = np.where(region == 255)
                isWhite = (len(indices[0])+0.0)/(x2-x1)
                if isWhite>thres1:
                    result.append(True)
                else:
                    result.append(False)


        if mode == 'ver':
            result = []
            margin = int(width/marginRatio)
            for index in range(margin):

                region = negatedImage[y1:y2, x1 + index:x2 + index + 1]
                indices = np.where(region == 255)
                isWhite = (len(indices[0])+0.0)/(y2-y1)

                if isWhite>thres1:
                    result.append(True)
                else:
                    result.append(False)
            result.reverse()
            for index in range(1,margin):

                region = negatedImage[y1:y2, x1 - index:x2-index+1]
                indices = np.where(region == 255)
                isWhite = (len(indices[0])+0.0)/(y2-y1)
                if isWhite>thres1:
                    result.append(True)
                else:
                    result.append(False)           
        # print result
        # print (sum(result) + 0.0)/len(result)
        # print margin
        consecutiveCount = 0
        current = None
        isConsecutive = False
        for pt in result:
            if consecutiveCount > consecutiveThres:
                isConsecutive = True
                break
            if not current:
                current = pt
            if pt:
                consecutiveCount +=1
                current = pt
            else:
                if current:
                    consecutiveCount = 0
                current = pt


        if (sum(result) + 0.0)/len(result) > thres2 or isConsecutive:
            return True
        else:
            return False



    @staticmethod
    def negateImage(image, thres = 30):
        image = 255 - image
        rat, image = cv.threshold(image, thres, 255, 0)
        return image
    
    @staticmethod
    def rotateImage(image):
        return np.rot90(image)
    
    
    @staticmethod
    # the image is in the image data
    # fill up the list in image_data
    # return the image_data
    def getLines(image, mode, minLength = 10):
        
        tmp = cv.HoughLinesP(image, rho = 100, theta = np.pi, threshold = 10, minLineLength = minLength, maxLineGap = 0)
        image_height, image_width =  image.shape

        lineList = []
        if mode == 0:
            if tmp != None:
                for line in tmp:
                    x1, y1, x2, y2 = list(line[0])
                    lineInfo = (x1, y2, x2, y1, abs(y2-y1))
                    lineList.append(lineInfo)
        elif mode == 1:
            if tmp != None:
                for line in tmp:
                    x1, y1, x2, y2 = line[0]
                    y1 = -y1 + image_height 
                    y2 = -y2 + image_height
                    lineInfo = (y1, x2, y2, x1, abs(y2-y1))
                    lineList.append(lineInfo)
                
        return lineList
    
    @staticmethod
    def cutLines(horLines, verLines, length_threshold = 0, line_ratio_threshold = 0.2):

        newList = []
        margin = 5

        for line in horLines:
            if line[4] < length_threshold:
                newList.append(line)
            else:
                x1, y1, x2, y2, length = line
                isnotcut = True
                for vx1, vy1, vx2, vy2, vlength in verLines:
                    if x1+margin < vx1 and vx1 < x2-margin and vy1 < y1 and y1 < vy2:

                        newline1 = [x1, y1, vx1, y2, vx1-x1]
                        newline2 = [vx1, y1, x2, y2, x2-vx1]

#                         print (vx1-x1+0.0) / length, (vx1 - x1+0.0)/length

                        if (vx1-x1+0.0) / length > line_ratio_threshold and (vx1 - x1+0.0)/length < 1-line_ratio_threshold:
                            newList.append(tuple(newline1))
                            newList.append(tuple(newline2))
                            isnotcut = False
                            break
                if isnotcut:
                    newList.append(line)
          
        return newList

    ## end static method for detectLine ##
    
    
    @staticmethod
    # corner data will be written in the image_data
    def getCorners(image_data, mask = None, debug = False):
        
        if image_data.preprocessed:      
            image = image_data.image_preproc
        else:
            image = image_data.originalImage
        
        if debug:
            print "Getting corners..."
            print "original image"
            PhyloParser.displayCorners(image)
         
        #sub preprocessing
        image = PhyloParser.binarize(image, thres = 180, mode = 0)
        if debug:
            print "binarized image"
            PhyloParser.displayCorners(image)

        # save the preprocessed image into image_data
        image_data.image_preproc_for_corner = image
        
        if mask is not None:
            image = PhyloParser.removeLabels(image, mask)
            if debug:
                print "use given mask"
            
        elif image_data.treeMask is not None:

            image = PhyloParser.removeLabels(image, image_data.treeMask)
            if debug:
                print "use mask generated from preprocessing"
        else:
            if debug:
                print "no mask"
                
#         image = image[100:110,155:165]
        image_data.upCornerList = PhyloParser.detectCorners(image, 1)
        image_data.downCornerList = PhyloParser.detectCorners(image, -1)
        image_data.jointUpList = PhyloParser.detectCorners(image, 2)
        
#         image_data.jointDownList = PhyloParser.detectCorners(image, -2, mask = mask)
        
        if debug:                
            PhyloParser.displayCorners(image_data.image_preproc, [image_data.upCornerList, image_data.downCornerList, image_data.jointUpList, image_data.jointDownList])
        
        image_data.cornerDetected = True
        return image_data
    
    @staticmethod
    # NOT use
    def refinePoint_(pointList, tolerance = 5):
        
        print "refine"
        print pointList
        
        remove_index = []
        for i in range(0, len(pointList)-1):
            j = i + 1
            
            print "index=",i 
            while True and j < len(pointList):

                p = pointList[i]
                next_p = pointList[j]
                
                print i, p
                print j, next_p
                
                if abs(p[0] - next_p[0]) <= tolerance and abs(p[1] - next_p[1]) <= tolerance:
                    if next_p[0] > p[0]:
                        remove_index.append(i)
                        print "remove", i   
                    else:
                        remove_index.append(j)
                        print "remove", j

                
                j += 1
                if j < len(pointList) and abs(p[1] - pointList[j][1]) > tolerance:
                    break                
        
        remove_index = list(Set(remove_index))
        remove_index = sorted(remove_index, reverse=True)
        
        print "pointList"
        print "remove_index", remove_index
        
        for index in remove_index:
            del pointList[index]
            
        return pointList
        
    
#     @staticmethod
#     def refineCorners(upCornerList, downCornerList, jointDownList):
#         removeIndexUp = []
#         removeIndexDown = []
#         
#         for i, p in enumerate(upCornerList):
#             if (p[0]-1, p[1]) in jointDownList:
#                 removeIndexUp.append(i)
#                 
#         removeIndexUp = sorted(removeIndexUp, reverse=True)
#         for index in removeIndexUp:
#             del upCornerList[index]
#         
#         for i, p in enumerate(downCornerList):
#             if (p[0]+1, p[1]) in jointDownList:
#                 removeIndexDown.append(i)
#                 
#         removeIndexDown = sorted(removeIndexDown, reverse=True)
#         for index in removeIndexDown:
#             del downCornerList[index]
#             
#         return upCornerList, downCornerList
#     
#     @staticmethod
#     def refineJoints(jointDownList, upCornerList, downCornerList):
#         
#         removeIndexList = []
#         for i, p in enumerate(jointDownList):
#             if (p[0]+1, p[1]) in upCornerList:
#                 removeIndexList.append(i)
#                 
#         for i, p in enumerate(jointDownList):
#             if (p[0]-1, p[1]) in downCornerList:
#                 removeIndexList.append(i)
#                 
#         removeIndexList = sorted(removeIndexList, reverse=True)
#         for index in removeIndexList:
#             del jointDownList[index]
#             
#         return jointDownList
                
        
        
    @staticmethod
    def detectCorners(image, mode, kernelSize = 3):
        

        line_width = 1##
        (kernel, norPixelValue, mode) = PhyloParser.createKernel(mode, 3)
#         print kernel
        
        filteredImage = cv.filter2D(image, -1, kernel)
        
#         print "filteredImage"
#         print filteredImage
#         print "image"
#         print image
#         
        
        # First threshold to find possible corners from filtering
        mmin = np.amin(filteredImage)
        threshold= norPixelValue * pow(line_width,1.5) * 255 - 1 ####
#         print "threshold: ", threshold
        upperBound = mmin + threshold
                
        indices = np.where(filteredImage < upperBound)
            
        cornerList = zip(indices[0], indices[1])
                
        new_cornerList = []
        for corner in cornerList:
                        
            if(corner[0]-1 >= 0 and corner[0]+2 <= image.shape[0] and corner[1]-1 >=0 and corner[1]+2 <= image.shape[1]):
                patch = image[corner[0]-1:corner[0]+2, corner[1]-1:corner[1]+2].copy().astype("float")/255      
                patch_variance =  np.var(patch)
            
#                 print corner
#                 print patch
#                 print patch_variance
#                 print filteredImage[corner[0]-1:corner[0]+2, corner[1]-1:corner[1]+2].copy().astype("float")
                
                patch_sum = max(np.amax(np.sum(patch, 0)), np.amax(np.sum(patch, 1)))

#                 print "sum, ", patch_sum
#                 print max(np.amax(np.sum(patch, 0)), np.amax(np.sum(patch, 1)))
#                 print "filter value,", filteredImage[corner[0], corner[1]]
#                 print "var,", patch_variance

                if abs(mode) == 1:
                    if patch_variance > 0.2 and patch_sum <= 2:
                        if mode == 1:
                            new_cornerList.append((corner[0]-1, corner[1]-1)) # shift back in line
                        else:
                            new_cornerList.append((corner[0]+1, corner[1]-1)) # shift back in line
                            
                if abs(mode) == 2:
                    if  0.17 < patch_variance < 0.247 and patch_sum <= 2:
                        if mode == 2:
                            new_cornerList.append((corner[0], corner[1]+1)) # shift back in line
                        else:
                            new_cornerList.append((corner[0], corner[1]-1)) # shift back in line
                
#         print "cornerList", len(cornerList)
#         print "new_cornerList", new_cornerList
        
#         print cornerList
        cornerList = new_cornerList
#         print cornerList
        cornerList = sorted(cornerList, key = lambda x: (int(x[1]), x[0]))
#         cornerList = PhyloParser.removeRepeatCorner(cornerList)

        return cornerList
        
    ## static method for detectCorners ##

    ## NOT USE
    @staticmethod
    def removeRepeatCorner(cornerList):
        i=0
        margin = 5
        xList = []
        yList = []
        while i<len(cornerList):
            x, y = cornerList[i]
            xList.append(x)
            yList.append(y)
            j = i
            while j+1<len(cornerList) and x + margin > cornerList[j+1][0] and x-margin < cornerList[j+1][0]:
                if y+margin > cornerList[j+1][1] and y-margin < cornerList[j+1][1]:
                    del cornerList[j+1]
                else:
                    j+=1
            i +=1
        return cornerList
    
    
    @staticmethod
    def displayCorners(image, list_pointList = []):
        
        displayImage = cv.cvtColor(image,cv.COLOR_GRAY2RGB)
        if len(list_pointList) > 0:

            rad = 2            
            colors = [(255, 0 , 0), (0, 255 , 0), (0, 0 , 255), (0, 255 , 255)]
            for i, pointList in enumerate(list_pointList):
                for y, x in pointList:
                    cv.rectangle(displayImage, (x-rad, y - rad), (x + rad, y +rad), color=colors[i], thickness=2)

        plt.imshow(displayImage)
        plt.show()
    
    
    @staticmethod
    def createKernel(mode, kernelSize):
        width = 1##
        kernel = np.zeros((kernelSize, kernelSize), np.float32)
        
        # top left corner
        if mode==1:
            for i in range(width):
                for x in range(kernelSize):
                    kernel[i][x] = 1
                    kernel[x][i] = 1
                    
        # join up kernel
        elif mode == 2:
            for i in range(width):
                for x in range(kernelSize):
                    kernel[x][kernelSize-1 - i] = 1
                    kernel[(kernelSize/2)+i][x] = 1
                    
        # join down kernel
        elif mode == -2:
            for i in range(width):
                for x in range(kernelSize):
                    kernel[x][0] = 1
                    kernel[(kernelSize/2)+i][x] = 1
            
        # bottom left corner
        elif mode == -1:
            for i in range(width):
                for x in range(kernelSize):
                    kernel[x][i] = 1
                    kernel[kernelSize-1 - i][x] = 1
                    
        summ = np.sum(kernel)
        kernel = kernel / summ
        
        if kernel[0][0] != 0:
            norPixelValue = kernel[0][0]
        else:
            norPixelValue = kernel[kernelSize-1][kernelSize-1]

        kernelPackage = (kernel, norPixelValue, mode)

        return kernelPackage
    
    ## end static method for detectCorners ##
    
    
    @staticmethod
    #return a list of dictionary
    #each element is set of point that stand in the same line
    #in each element
    #    key "corners" contains all corners in such line
    #    key "joints" contains the corner and it's corresponding joints, the first point in the list is the anchor corner
    def makeLinesFromCorner(image_data, margin = 5, debug = False):
        
        image = image_data.image_preproc_for_corner.copy()
        
        upCornerList_ver = list(image_data.upCornerList)
        upCornerList_hor = sorted(list(image_data.upCornerList),  key = lambda x: (int(x[0]), x[1]))
        upCornerIndex_ver = 0  
        upCornerIndex_hor = 0  
#         print "upCornerList", upCornerList_ver

        downCornerList_ver = list(image_data.downCornerList)
        downCornerList_hor = sorted(list(image_data.downCornerList),  key = lambda x: (int(x[0]), x[1]))
        downCornerIndex_hor = 0
#         print "downCornerList", downCornerList_ver
        
        jointUpList_ver = list(image_data.jointUpList)
        jointUpList_hor = sorted(list(image_data.jointUpList),  key = lambda x: (int(x[0]), x[1]))
#         print "jointUpList", jointUpList_ver
        
#         jointDownList = list(image_data.jointDownList)
#         print "jointDownList", jointDownList
        
        pointSet_ver = [] #vertical line between corners
        upPointSet_hor = [] #horizontal line between top left corners and corresponding joints
        downPointSet_hor = [] #horizontal line between bottom left corners and corresponding joints
        
        while upCornerIndex_ver < len(upCornerList_ver):
            upCorner = upCornerList_ver[upCornerIndex_ver]
             
            # vertical match
            cornerCandidate, downCornerList_ver = PhyloParser.matchPoints(upCorner, downCornerList_ver, image, 0, margin = margin)
            jointCandidate, jointUpList_ver = PhyloParser.matchPoints(upCorner, jointUpList_ver, image, 0, margin = margin)
                 
            ## find vertical line!
            if len(cornerCandidate) > 1:
                data = {}
                data["corners"] = cornerCandidate
                 
#                 if len(jointCandidate) > 1:
#                     del jointCandidate[0]
                     
                data["joints"] = jointCandidate
                 
                pointSet_ver.append(data)
                del upCornerList_ver[upCornerIndex_ver]
                 
            ## find on line, go next
            else:
                upCornerIndex_ver += 1
            
            
        # match horizontal line on up corner   
        while upCornerIndex_hor < len(upCornerList_hor):
            upCorner = upCornerList_hor[upCornerIndex_hor]

            # horizontal math
            jointCandidate, jointUpList_hor = PhyloParser.matchPoints(upCorner, jointUpList_hor, image, 1, margin = margin)

            ## find horizontal line!
            if len(jointCandidate) > 1:
                data = {}
                data["joints"] = jointCandidate
                 
                upPointSet_hor.append(data)
                del upCornerList_hor[upCornerIndex_hor]
#                 print "find joint candidate", jointCandidate
                 
            ## find no line, go next
            else:
                upCornerIndex_hor += 1

        
        # match horizontal line on down corner
        # keep using the same jointUpList_hor
        while downCornerIndex_hor < len(downCornerList_hor):
            downCorner = downCornerList_hor[downCornerIndex_hor]

            # horizontal math
            jointCandidate, jointUpList_hor = PhyloParser.matchPoints(downCorner, jointUpList_hor, image, 1, margin = margin)

            ## find horizontal line!
            if len(jointCandidate) > 1:
                data = {}
                data["joints"] = jointCandidate
                 
                downPointSet_hor.append(data)
                del downCornerList_hor[downCornerIndex_hor]
#                 print "find joint candidate", jointCandidate
                 
            ## find no line, go next
            else:
                downCornerIndex_hor += 1
        
        pointSet_ver = PhyloParser.removeDuplicatePoint(pointSet_ver, 0, margin = 5)
        upPointSet_hor = PhyloParser.removeDuplicatePoint(upPointSet_hor, 0, margin = 5)
        downPointSet_hor = PhyloParser.removeDuplicatePoint(downPointSet_hor, 0, margin = 5)

        if debug:
            ver_lines = PhyloParser.pointSetToLine(image_data.image_preproc_for_corner, pointSet_ver, type="corners")
            hor_lines_up =  PhyloParser.pointSetToLine(image_data.image_preproc_for_corner, upPointSet_hor, type="joints")
            hor_lines_down =  PhyloParser.pointSetToLine(image_data.image_preproc_for_corner, downPointSet_hor, type="joints")
            
            PhyloParser.displayCornersAndLine(image, [upCornerList_hor, jointUpList_hor], [ver_lines, hor_lines_up, hor_lines_down])
        
                
        #         print "remain upCornerList horizontal", upCornerList_hor
        
        image_data.pointSet_ver = pointSet_ver
        image_data.upPointSet_hor = upPointSet_hor
        image_data.downPointSet_hor = downPointSet_hor
        
        image_data.lineDetectedFromCorners = True
        
        return image_data

    ## static method for makeLinesFromCorner ##

    @staticmethod
    # axis = 0 --> vertically match
    # axis = 1 --> horizontally match 
    def matchPoints(point, candidatePoints, image, axis, margin = 5):
        
        if axis == 0 or axis == 1 :
            
            index_for_margin_test = 1 - axis
            index_for_location_test = axis
            
            matchPoints = [point] ### 
            candidatePointsIndex = 0

            while True and candidatePointsIndex < len(candidatePoints):
                downCorner = candidatePoints[candidatePointsIndex]
#                 print "this downCornerIndex: ", candidatePointsIndex, downCorner
                
                if  (abs(downCorner[1-axis] - point[1-axis]) <= margin) and  (downCorner[axis] - point[axis] > 0) and PhyloParser.isInLine(point, downCorner, image):
                    # find match,  stay in the same index due to removal"
                    matchPoints.append(downCorner)
                    del candidatePoints[candidatePointsIndex]
                
                elif downCorner[1-axis] - point[1-axis] <= margin or downCorner[axis] - point[axis] > 0 or PhyloParser.isInLine(point, downCorner, image):
                    # not match, but close, keep searching next element
                    candidatePointsIndex += 1
                    
                else: 
                    # once margin test fail, the later elements will all fail, so stop iterating
                    break
            
            return matchPoints, candidatePoints
        
        else:
            print "axis must ether 1 or 0"
            return None, candidatePoints

    @staticmethod
    # determine if two points are in the same line
    def isInLine(corner1, corner2, image, threshold = 0.01):
        
        y_min = min(corner1[0], corner2[0])
        y_max = max(corner1[0], corner2[0])      
        x_min = min(corner1[1], corner2[1])
        x_max = max(corner1[1], corner2[1])
        
        subimage = image[y_min:y_max+1, x_min:x_max+1].copy().astype("float")/255  ## not count the later index
        variance =  np.var(subimage)
        
        return variance < threshold
    
    @staticmethod
    # remove duplicate point in each point set
    def removeDuplicatePoint(pointSet, axis, margin = 5):
        for s in pointSet:
            if "corners" in s and len(s["corners"]) > 2:
                s["corners"] = PhyloParser.refinePoint_v2(s["corners"], margin)
            if "joints" in s and len(s["joints"]) > 2:
                s["joints"] = PhyloParser.refinePoint_v2(s["joints"], margin)
                
        return pointSet
                
    @staticmethod 
    def refinePoint_v2(pointList, margin = 5):
        
        # remove continuous point
        remove_index = []
        for i in range(0, len(pointList)-1):
            j = i + 1

            p = pointList[i]
            next_p = pointList[j]
            
#             print abs(p[0] - next_p[0])
#             print abs(p[1] - next_p[1])
            if abs(p[0] - next_p[0]) <= 1 and abs(p[1] - next_p[1]) <= 1:
                remove_index.append(i)
              
              
              
        remove_index = list(Set(remove_index))
        remove_index = sorted(remove_index, reverse=True)  
        for index in remove_index:
            del pointList[index]
        
        
        # remove duplicate in margin
        remove_index = []
        for i in range(0, len(pointList)-1):
            j = i + 1

            p = pointList[i]
            next_p = pointList[j]
            
#             print abs(p[0] - next_p[0])
#             print abs(p[1] - next_p[1])
            if abs(p[0] - next_p[0]) <= margin and abs(p[1] - next_p[1]) <= margin:
                remove_index.append(i)
              
        remove_index = list(Set(remove_index))
        remove_index = sorted(remove_index, reverse=True)  
        for index in remove_index:
            del pointList[index]
             
        return pointList
                
    @staticmethod
    # need sorted
    # called by removeDuplicatePoint
    # select the very bottom or very right point in the margin as the representative point 
    def refinePoint(pointList, margin = 5):
         
        remove_index = []
        for i in range(0, len(pointList)-1):
            j = i + 1

            p = pointList[i]
            next_p = pointList[j]
            
            print abs(p[0] - next_p[0])
            print abs(p[1] - next_p[1])
            if abs(p[0] - next_p[0]) <= margin and abs(p[1] - next_p[1]) <= margin:
                remove_index.append(i)
              
        remove_index = list(Set(remove_index))
        remove_index = sorted(remove_index, reverse=True)
         
        for index in remove_index:
            del pointList[index]
             
        return pointList
    

    
    @staticmethod
    def displayCornersAndLine(image, list_pointList = [], list_lines = []):
        
        displayImage = cv.cvtColor(image,cv.COLOR_GRAY2RGB)
        if len(list_pointList) > 0:
            rad = 2            
            colors = [(255, 0 , 0), (0, 255 , 0), (0, 0 , 255), (0, 255 , 255)]
            for i, pointList in enumerate(list_pointList):
                for y, x in pointList:
                    cv.rectangle(displayImage, (x-rad, y - rad), (x + rad, y +rad), color=colors[i], thickness=2)


        if len(list_lines) > 0:
            colors = [(255, 150 , 0), (150, 255 , 0), (150, 0 , 255)]
            for i, lines in enumerate(list_lines):
                for line in lines:
                    x1, y1, x2, y2, length = line
                    cv.rectangle(displayImage, (x1, y1), (x2, y2), color=colors[i], thickness=2)
            
        plt.imshow(displayImage)
        plt.show()

    ## end static method for makeLinesFromCorner ##
    

    @staticmethod
    # remove duplicate lines
    # for vertical lines: select the very right line 
    # for horizontal lines: select the very bottom line
    def refineLines(image_data, debug = False):
        
        image = image_data.image_preproc_for_corner
                
        if debug:
            PhyloParser.displayLines(image, image_data.horLines)
        
        midPointOfHorLines = PhyloParser.getMidPoints(image_data.horLines)
        midPointOfVerLines = PhyloParser.getMidPoints(image_data.verLines)
        
        
        midPointOfHorLines = sorted(midPointOfHorLines, key = lambda x: (x[0], x[1]))
        midPointOfVerLines = sorted(midPointOfVerLines, key = lambda x: (x[1], x[0]), reverse=True) #take the very right lines
        

        image_data.horLines, image_data.horLineGroup = PhyloParser.getRepresentativeLines(image, midPointOfHorLines)
                    
        if debug:
            PhyloParser.displayLines(image, image_data.horLines)
            
        image_data.verLines, image_data.verLineGroup = PhyloParser.getRepresentativeLines(image, midPointOfVerLines)
                    
        if debug:
            PhyloParser.displayLines(image, image_data.verLines)
            
        image_data.lineRefined = True
        return image_data

    ## static method for refineLines ##

    @staticmethod
    def getMidPoints(lines):
        midPoints = []
        for l in lines:
            midPoints.append((int((l[0] + l[2])/2), int((l[1] + l[3])/2), l[4], l))
        return midPoints
        
    
    @staticmethod
    #axis = 0: vertical lines
    #axis = 1: horizontal lines
    def getRepresentativeLines(image, midPointOfLines, margin = 5):
                
        keep_lines = []
        group_line_indices = []
        group_lines = []
           
        index_head = 0
        line1 = midPointOfLines[index_head]
        max_length = line1[2]
        keep_index = 0
        index_head += 1
        temp_line_indices = [0]
        
        while index_head < len(midPointOfLines):
            
            line2 = midPointOfLines[index_head]
            
            x_in_margin = abs(line1[0] - line2[0]) <= margin
            y_in_margin = abs(line1[1] - line2[1]) <= margin
            
            print "line1", line1
            print "line2", line2, PhyloParser.checkLine(image, line2[3])
            
            if x_in_margin and y_in_margin:
                print "find a continuous line"
                #find a continuous line
                temp_line_indices.append(index_head)            
                
                # check if line2 is better to keep as the main line of the set
                if line2[2] >= max_length and PhyloParser.checkLine(image, line2[3]):
                    print "put line2 index into keep index"
                    max_length = line2[2]
                    keep_index = index_head
                    
                # moving forward
                line1 = line2
                index_head += 1
            
            elif (not x_in_margin and not y_in_margin) or index_head == len(midPointOfLines) - 1:
                print "break"
                #save set
                group_line_indices.append(temp_line_indices)####
                group_lines.append([midPointOfLines[x] for x in temp_line_indices])
                keep_lines.append(midPointOfLines[keep_index][3])
                
                print "this set:", group_lines[-1]
                print "keep lines:", len(keep_lines), keep_lines
                
                #remove found index from the lines
                temp_line_indices = sorted(temp_line_indices, reverse=True)
                for i in temp_line_indices:
                    del midPointOfLines[i]
                    
                print "remain lines:", midPointOfLines
                #start over
                index_head = 0 
                line1 = midPointOfLines[index_head]
                max_length = line1[2]
                keep_index = 0
                index_head += 1
                temp_line_indices = [0]
                
            else:
                #keep moving foward
                print "not matched, keep searching next"
                index_head += 1
            
        
        #pick up last sest
        if midPointOfLines > 0:
            group_lines.append(midPointOfLines)
            keep_lines.append(midPointOfLines[keep_index][3])
            
            
        print "group_lines", len(group_lines), group_lines
        print "keep lines:", len(keep_lines), keep_lines
        
        return keep_lines, group_lines

    @staticmethod
    # determine if the given line is a truly BLACK line
    def checkLine(image, line, var_threshold = 0.01, mean_threshold = 3):

        if line[0] == line[2]:
            array = image[line[1]:line[3], line[0]:line[0]+1]
        else:
            array = image[line[1]:line[1]+1, line[0]:line[2]]
            
            
        variance = np.var(array.astype("float")/255)
        mean = np.mean(array)

        return variance < var_threshold and mean < mean_threshold


    @staticmethod 
    def getColor(count):
        if count %5 == 0:
            return (255, 0, 0)
        elif count % 5 == 1:
            return (0, 255, 0)
        elif count % 5 == 2:
            return (0, 0, 255)
        elif count % 5 == 3:
            return (255, 0, 255)
        else:
            return (0, 255, 255)

    @staticmethod
    #for debug
    def displayLines_v2(image, lines, maxNum, sigma):
                
        if len(image.shape) == 2:
            whatever = cv.cvtColor(image, cv.COLOR_GRAY2RGB)

        count = 0

        for line in lines:
            if line[2] - line[0] < maxNum + sigma:
                count +=1
                color = PhyloParser.getColor(count)
                x1, y1, x2, y2, length = line
                cv.rectangle(whatever, (x1, y1), (x2, y2), color=color, thickness=-1)
        plt.imshow(whatever)
        plt.show()

    
    @staticmethod
    #for debug
    def displayLines(image, lines):
                
        if len(image.shape) == 2:
            img = cv.cvtColor(image, cv.COLOR_GRAY2RGB)

        count = 0

        for line in lines:
            count +=1
            color = PhyloParser.getColor(count)
            x1, y1, x2, y2, length = line
            cv.rectangle(img, (x1, y1), (x2, y2), color=(255, 0, 0), thickness=2)
        plt.imshow(img)
        plt.show()
                
    ## end static method for refineLines ##
    
    
    @staticmethod
    # merge the lines created from corners with lines created from line detection
    def includeLinesFromCorners(image_data):
        if image_data.lineDetectedFromCorners and image_data.lineDetected:

            ver_lines = PhyloParser.pointSetToLine(image_data.image_preproc_for_corner, image_data.pointSet_ver, type="corners")
            hor_lines_up =  PhyloParser.pointSetToLine(image_data.image_preproc_for_corner, image_data.upPointSet_hor, type="joints")
            hor_lines_down =  PhyloParser.pointSetToLine(image_data.image_preproc_for_corner, image_data.downPointSet_hor, type="joints")
            
            hor_lines_up = PhyloParser.purifyLines(hor_lines_up, image_data.image_preproc_for_line_detection, PhyloParser.negateImage(image_data.image_preproc_for_line_detection), 'hor')
            hor_lines_down = PhyloParser.purifyLines(hor_lines_down, image_data.image_preproc_for_line_detection, PhyloParser.negateImage(image_data.image_preproc_for_line_detection), 'hor')
            ver_lines = PhyloParser.purifyLines(ver_lines, image_data.image_preproc_for_line_detection, PhyloParser.negateImage(image_data.image_preproc_for_line_detection), 'ver')

            image_data.horLines += hor_lines_up
            image_data.horLines += hor_lines_down
            image_data.verLines += ver_lines
        
        else:
            print "Found no lines created from corner detection."
            
        return image_data
    
    ## static method for includeLinesFromCorners ##
    
    @staticmethod
    # need sorted already
    def pointSetToLine(image, pointSetList, type = "corners"):
        lineList = []
        for pointSet in pointSetList:
            
            points = pointSet[type] # select the type of point set
            
            # must have at least two points to form a line
            if len(points) > 1:
                y1 = points[0][0]
                x1 = points[0][1]
                
                y2 =  points[-1][0]
                x2 =  points[-1][1]
            
                lineLength = max(abs(x2-x1),abs(y2-y1))
                
                if type == "corners": #vertical
                    # two points not inline, pick the one with lower pixel value as y
                    if x1 != x2 and image[y1, x1] <= image[y2, x2]:
                        lineList.append((x1, y1, x1, y1, lineLength))
                    else:
                        lineList.append((x2, y1, x2, y1, lineLength))
                        
                    lineList.append((x1, y1, x2, y2, lineLength))
                else: #horizontal
                    # two points not inline, pick the one with lower pixel value as y
                    if y1 != y2 and image[y1, x1] <= image[y2, x2]:
                        lineList.append((x1, y1, x2, y1, lineLength))
                    else:
                        lineList.append((x1, y2, x2, y2, lineLength))
        
        return lineList
    
    ## end static method for includeLinesFromCorners ##
    
    
    # match line groups
    def matchLineGroups(self, image_data, debug = False):

        if image_data.lineDetected:

            image_data = self.matchParent(image_data)
            image_data = self.matchChildren(image_data)

            if debug:
                image_data.displayTargetLines('parent')
                image_data.displayTargetLines('children')
                image_data.displayTargetLines('interLines')
                image_data.displayTargetLines('anchorLines')
            
            image_data.lineMatched = True

        else:
            print "Error! Please do detectLines before this method"
        
        return image_data
       
    
    ## static method for matchLineGroups ##

    def matchChildren(self, image_data):
        horLines = image_data.horLines
        verLines = image_data.verLines
        horLineMask = image_data.horLineMask
        verLineMask = image_data.verLineMask
        horLineMappingDict = image_data.horLineMappingDict
        verLineMappingDict = image_data.verLineMappingDict

        height, width, dim = horLineMask.shape

        children = []
        countIndexes = 1
        childrenIndexes = {}
        anchorLines = []
        anchorLinesIndexes = []

        for verLineIndex, verLineGroup in verLineMappingDict['lineMapping'].items():
            overlapArea = np.where((verLineMask[:,:,0] == verLineIndex) & (horLineMask[:,:,1] != 0))
            overlapIndexes = horLineMask[PhyloParser.mapping2Dto3D(overlapArea, 1)]
            intersectionArea = np.where((verLineMask[:,:,0] == verLineIndex) & (horLineMask[:,:,0] != 0))
            intersectionIndexes = list(horLineMask[PhyloParser.mapping2Dto3D(intersectionArea, 0)])
            # print 'step1', intersectionIndexes

            for overlapIndex in overlapIndexes:
                intersectionIndexes += list(horLineMappingDict['overlapMapping'][overlapIndex])
            # print 'step2', intersectionIndexes
            potentialChildren = list(set(intersectionIndexes))
            midPoint = verLineGroup['midPoint']
            leaves = []
            leavesIndex = []
            for potChildIndex in potentialChildren:
                hor_rline = horLineMappingDict['lineMapping'][potChildIndex]['rline']
                horQuarterPt = (hor_rline[1], (hor_rline[0]*3+hor_rline[2])/4)
                if PhyloParser.isPointOnTheRight(midPoint, horQuarterPt):

                    leaves.append(horLineMappingDict['lineMapping'][potChildIndex]['rline'])
                    leavesIndex.append(potChildIndex)
                    if horLineMappingDict['lineMapping'][potChildIndex]['type'] != 'interLine':
                        horLineMappingDict['lineMapping'][potChildIndex]['type'] = 'anchorLine'
                        anchorLines.append(horLineMappingDict['lineMapping'][potChildIndex]['rline'])
                        anchorLinesIndexes.append(potChildIndex)

                    verLineMappingDict['lineMapping'][verLineIndex]['children'].append(potChildIndex)
                    horLineMappingDict['lineMapping'][potChildIndex]['children'].append(verLineIndex)
            
                                  


            leaves = sorted(leaves, key = lambda x: x[1])
            if len(leavesIndex)>=2:
                leavesIndex = sorted(leavesIndex, key = lambda x: horLineMappingDict['lineMapping'][x]['rline'][1])
                tmpLeaves = []
                for leafIndex, leaf in enumerate(leavesIndex):
                    if leafIndex == 0:
                        tmpLeaves.append(horLineMappingDict['lineMapping'][leaf]['rline_upper'])
                    elif leafIndex == len(leavesIndex) - 1:
                        tmpLeaves.append(horLineMappingDict['lineMapping'][leaf]['rline_lower'])
                    else:
                        tmpLeaves.append(horLineMappingDict['lineMapping'][leaf]['rline'])
                children.append(((verLineGroup['rline'], tuple(leaves)), 0))
                
                childrenIndexes[countIndexes] = leavesIndex
                countIndexes +=1
            elif len(leavesIndex) ==1:
                x1, y1, x2, y2, length = leaves[0]
                vx1, vy1, vx2, vy2, vlength = verLineGroup['rline']
                if abs(y1 - vy1) > abs(y1 - vy2):
                    leaves = [None, horLineMappingDict['lineMapping'][leavesIndex[0]]['rline_lower']]
                else:
                    leaves = [horLineMappingDict['lineMapping'][leavesIndex[0]]['rline_upper'], None]

                children.append(((verLineGroup['rline'], tuple(leaves)), 0))

                childrenIndexes[countIndexes] = leavesIndex
                countIndexes +=1


        for horLineIndex, horLineGroup in horLineMappingDict['lineMapping'].items():
            if len(horLineGroup['children'])>1:

                
                removedIndex = []
                for childIndex in horLineGroup['children']:
                    hx1, hy1, hx2, hy2, hlength = horLineMappingDict['lineMapping'][horLineIndex]['rline']
                    x1, y1, x2, y2, length = verLineMappingDict['lineMapping'][childIndex]['rline']
                    if not PhyloParser.isDotWithinLine((hx1, hy1), verLineMappingDict['lineMapping'][childIndex]['rline']):
                        verLineMappingDict['lineMapping'][childIndex]['children'].remove(horLineIndex)
                        removedIndex.append(childIndex)
                for i in removedIndex:
                    horLineGroup['children'].remove(i)

                if len(horLineGroup['children'])>1:
                    isCloserIndex = None
                    isCloserSpot = None
                    for childIndex in horLineGroup['children']:
                        x1, y1, x2, y2, length = verLineMappingDict['lineMapping'][childIndex]['rline']
                        if not isCloserIndex or x1>isCloserSpot:
                            if isCloserIndex:
                                verLineMappingDict['lineMapping'][isCloserIndex]['children'].remove(horLineIndex)
                            isCloserIndex = childIndex
                            isCloserSpot = x1
                        else:
                            verLineMappingDict['lineMapping'][childIndex]['children'].remove(horLineIndex)
                    newChildren = [isCloserIndex]
                    horLineGroup['children'] = newChildren



        image_data.children = children
        image_data.childrenIndexes = childrenIndexes
        image_data.anchorLines = anchorLines
        image_data.anchorLinesIndexes = anchorLinesIndexes

        return image_data

    @staticmethod
    def isPointOnTheRight(refPoint, targetPoint):
        y1, x1 = refPoint
        y2, x2 = targetPoint

        if x2 >= x1:
            return True
        else:
            return False            


    def matchParent(self, image_data):

        horLines = image_data.horLines
        verLines = image_data.verLines
        horLineMask = image_data.horLineMask
        verLineMask = image_data.verLineMask
        horLineMappingDict = image_data.horLineMappingDict
        verLineMappingDict = image_data.verLineMappingDict

        height, width, dim = horLineMask.shape
        parents = []
        parentsIndexes = {}
        countIndex = 1
        interLines = []
        interLinesIndexes = []


        for horLineIndex, horLineGroup in horLineMappingDict['lineMapping'].items():
            horLineGroup['type'] = None
            overlapArea = np.where((horLineMask[:,:,0] == horLineIndex) & (verLineMask[:,:,1] !=0))
            overlapIndexes = verLineMask[PhyloParser.mapping2Dto3D(overlapArea, 1)]
            intersectArea = np.where((horLineMask[:,:,0] == horLineIndex) & (verLineMask[:,:,0] != 0))
            intersectIndexes = list(verLineMask[PhyloParser.mapping2Dto3D(intersectArea, 0)])

            for overlapIndex in overlapIndexes:
                intersectIndexes += list(verLineMappingDict['overlapMapping'][overlapIndex])

            potentialParents = list(set(intersectIndexes))
            rline = horLineGroup['rline']
            midPoint = horLineGroup['midPoint']
            quarterPt = (rline[1], (rline[0] + 3 * rline[2]) / 4)
            realParent = []
            realParentIndex = []
            isFound = False
            for potParentIndex in potentialParents:
                verMidPoint = verLineMappingDict['lineMapping'][potParentIndex]['midPoint']
                if PhyloParser.isPointOnTheRight(quarterPt, verMidPoint):
                    realParent.append(verLineMappingDict['lineMapping'][potParentIndex]['rline'])
                    realParentIndex.append(potParentIndex)
                    parent = ((horLineGroup['rline'], verLineMappingDict['lineMapping'][potParentIndex]['rline']), 0)
                    parents.append(parent)
                    verLineMappingDict['lineMapping'][potParentIndex]['parent'].append(horLineIndex)
                    isFound = True

                    parentIndex = (horLineIndex, potParentIndex)
                    parentsIndexes[countIndex] = parentIndex
                    countIndex+=1



            if isFound:

                interLines.append(horLineGroup['rline'])

                interLinesIndexes.append(horLineIndex)
                horLineGroup['type'] = 'interLine'
                horLineGroup['parent'] = realParentIndex


            # overlapIndexes = np.where((horLineMask[:,:,0] == horLineIndex) & (verLineMask[:,:,1] !=0))
            # if horLineGroup['rline'] ==  (174, 482, 355, 482, 181):
            #     print 'overlap', overlapIndexes
            # isFound = False
            # if len(overlapIndexes[0]) == 0:
            #     intersectionIndexes = np.where((horLineMask[:,:,0] == horLineIndex) & (verLineMask[:,:,0] != 0))
            #     if horLineGroup['rline'] == (174, 482, 355, 482, 181):
            #         print 'intersect', intersectionIndexes
            #     midPoint = horLineGroup['midPoint']
            #     rightEndIndexes = PhyloParser.getThePointsOnTheRightEnd(intersectionIndexes, midPoint)
                
            #     if len(rightEndIndexes[0])!=0:
            #         indexNumberList = verLineMask[PhyloParser.mapping2Dto3D(rightEndIndexes, 0)]
            #         uniqueIndexes = list(set(indexNumberList))
            #         for index in uniqueIndexes:
            #             parent = ((horLineGroup['rline'], verLineMappingDict['lineMapping'][index]['rline']), 0 )
            #             parents.append(parent)
            #             horLineGroup['parent'].append(index)
            #             verLineMappingDict['lineMapping'][index]['parent'].append(horLineIndex)
            #             isFound = True
            # else:
            #     midPoint = horLineGroup['midPoint']
            #     rightEndIndexes = PhyloParser.getThePointsOnTheRightEnd(overlapIndexes, midPoint)
            #     if horLineGroup['rline'] == (174, 482, 355, 482, 181):
            #         print 'rightEnd', rightEndIndexes
            #         print np.where(verLineMask[:,:,0] == 12)
            #     if len(rightEndIndexes[0])!=0:
            #         indexNumberList = verLineMask[PhyloParser.mapping2Dto3D(rightEndIndexes, 1)]
            #         if horLineGroup['rline'] == (174, 482, 355, 482, 181):
            #             print 'indexNumber', indexNumberList
            #         uniqueIndexes = list(set(indexNumberList))
            #         for overlapIndex in uniqueIndexes:
            #             parentCandidates = verLineMappingDict['overlapMapping'][overlapIndex]
            #             for index in parentCandidates:
            #                 parent = ((horLineGroup['rline'], verLineMappingDict['lineMapping'][index]['rline']), 0 )
            #                 parents.append(parent)
            #                 horLineGroup['parent'].append(index)
            #                 verLineMappingDict['lineMapping'][index]['parent'].append(horLineIndex)
            #                 isFound = True
            # if isFound:
            #     interLines.append(horLineGroup['rline'])
            #     horLineGroup['type'] = 'interLine'

        image_data.parent = parents
        image_data.parentsIndexes = parentsIndexes
        image_data.interLines = interLines
        image_data.interLinesIndexes = interLinesIndexes
        return image_data


            # print coverRange
            # print np.where(verLineMask[coverRange] != 0)
            # print verLineMask[coverRange]
            # intersectionMask[np.where(verLineMask[coverRange] != 0)] = 255

            
        #     verLineGroup['rline']


        # parent = []


    @staticmethod
    def getThePointsOnTheRightEnd(intersecRange,midPoint):
        rightEndY = []
        rightEndX = []
        my, mx = midPoint

        for index, x in enumerate(intersecRange[1]):
            
            if x >= mx:
                rightEndX.append(x)
                rightEndY.append(intersecRange[0][index])
        rightEnd = (np.asarray(rightEndY), np.asarray(rightEndX))

        return rightEnd
    
    
    ## end static method for matchLineGroups ##
    


    @staticmethod     
    def removeText(image_data):
        return image_data
    

    def removeRepeatLinesBasic(self, lineList):
        margin = 5
        i=0 
        while i<len(lineList):
            x1, y1, x2, y2, length= lineList[i]

            for j in xrange(len(lineList)-1, i, -1):
                if x1 - margin < lineList[j][0] and x2 + margin > lineList[j][2]:
                    if y2+margin > lineList[j][3] and y1-margin < lineList[j][1]:
                        del lineList[j]
            i +=1
        return lineList

    def removeRepeatLines(self, lineList):
        margin = 5
        i=0
        while i<len(lineList):
            lines, dist = lineList[i]
            x1, y1, x2, y2, length =lines[0]

            for j in xrange(len(lineList)-1, i, -1):

                if x1 - margin < lineList[j][0][0][0] and x2 + margin > lineList[j][0][0][2]:
                    if y2+margin > lineList[j][0][0][3] and y1-margin < lineList[j][0][0][1]:
                        if lineList[j][0][0][4] <= length+ (margin-2):
                            del lineList[j]

            i +=1

        return lineList


    ## end static method for matchLines ##



    

    @staticmethod
    # padding: enlarge box area
    # margin: height of scan zone after line with no assigned box
    def getSpecies(image_data, padding = 2, margin = 5, debug = False):

        image = image_data.image.copy()
        anchorLines = image_data.anchorLines       
        
        # get non-tree contours #old method using sliding window
#         contours, varianceMask = PhyloParser.findTextContours_old(image_data)
        
        
        nonTreeMask = image_data.nonTreeMask.copy()
        treeMask = image_data.treeMask.copy()
        varianceMask = 255 - (nonTreeMask + treeMask)
        
        mask, contours, hierarchy = PhyloParser.findTextContours(255-nonTreeMask)
        
        # transform contours to bonding boxes
        contourBoxes = []
#         img = image.copy()
        for cnt in contours:
            contourBoxes.append(PhyloParser.getContourInfo(cnt))
#             b = PhyloParser.getContourInfo(cnt)
#             cv.rectangle(img,(b[2],b[0]),(b[3],b[1]),(0,125,0),2)
#         PhyloParser.displayImage(img)
                        
        # sort boxes and anchorlines from top to bot for further matching       
        contourBoxes = sorted(contourBoxes, key = lambda x: (x[0], x[2])) #top, left
        anchorLines = sorted(anchorLines, key = lambda x: (x[3], x[2])) #top, right        
        
        # DEBUG
        if debug:
            print "show variance mask with anchor lines"
            img = varianceMask.copy()
            for b in contourBoxes:
                cv.rectangle(img,(b[2],b[0]),(b[3],b[1]),(0,125,0),2)
            PhyloParser.displayLines(img, anchorLines)
        
        ## filter tall thin boxes
        contourBoxes = PhyloParser.refineContourBox(contourBoxes, threshold_aspect = 10, threshold_height = 20)
        
        ## match line and boxes
        lineIndex2ClusterBoxes, lineIndex2BoxIndex, lineIndex2SubBoxes, lineInTextIndex, noTextLineIndex, orphanBoxFromShareBox, activateBoxIndices, shareBoxIndices, orphanBoxIndices = PhyloParser.matchAnchorLineAndContourBox(image, anchorLines, contourBoxes)
        
        ## collect orphan boxes
        orphanBoxes = PhyloParser.clusterOrphanBox(image, anchorLines, contourBoxes, orphanBoxFromShareBox, orphanBoxIndices)
        
        if debug:
            print "line2box"
            lines = []
            img = image.copy()
            for line_index in lineIndex2ClusterBoxes:
                lines.append(anchorLines[line_index])
                boxes = lineIndex2ClusterBoxes[line_index]
                for b in boxes:
                    cv.rectangle(img,(b[2],b[0]),(b[3],b[1]),(0,125,0),2)
            PhyloParser.displayLines(img, lines)
            
            print "orphane boxes"
            img = image.copy()
            for b in orphanBoxes:
                cv.rectangle(img,(b[2],b[0]),(b[3],b[1]),(0,125,0),2)
            PhyloParser.displayLines(img, lines)
            

        ## get text for anchor lines
        line2Text = PhyloParser.convertLine2Text(image, anchorLines, lineIndex2ClusterBoxes, lineInTextIndex, noTextLineIndex, padding = padding)
        
        ## get text from orphan
        orphanBox2Text = PhyloParser.box2Text(image, orphanBoxes)
        
        if debug:
            for line in line2Text:
                print "line", line, "text", line2Text[line]
            for box in orphanBox2Text:
                print "box", box, "text", orphanBox2Text[box]

        image_data.count_shareBox = len(shareBoxIndices)
        image_data.count_contourBoxes = len(contourBoxes)
        image_data.line2Text = line2Text
        image_data.orphanBox2Text = orphanBox2Text
        image_data.anchorLines = anchorLines
        image_data.speciesNameReady = True 

        return image_data
 
 
    ## static methods for getSpecies ##

    @staticmethod
    # return a mask of the tree, a mask of text and contours
    def findTextContours(var_mask1):

        var_mask1 = 255 - var_mask1

        height, width = var_mask1.shape
        var_mask1 = cv.copyMakeBorder(var_mask1, 1, 1, 1, 1, cv.BORDER_CONSTANT, value = 0)
        _, contours, hierarchy= cv.findContours(var_mask1.copy(), cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
        
        lenghtList = []
        for cnts in contours:
            lenghtList.append(len(cnts))
            for index, points in enumerate(cnts):
                cnts[index] = points - 1 #shift back (because of padding)
           
        hierarchy = hierarchy[0].tolist()
        mask = np.zeros((height, width), dtype = np.uint8)

        textContours = []
        for index in range(0, len(contours)):
#             print contours
#             nonTreeMask = np.zeros((height, width), dtype = np.uint8)
            # draw only contour in level 0
            if hierarchy[index][3] == -1:#####Buggy
                cv.drawContours(mask, contours, index, (255), thickness = -1)
                textContours.append(contours[index])
#             print hierarchy[index][3] == -1
#             PhyloParser.displayImage(nonTreeMask)

#         PhyloParser.displayImage(mask)
        return mask, textContours, hierarchy

    
    @staticmethod
    # padding: enlarge box area
    def convertLine2Text(image, anchorLines, lineIndex2ClusterBoxes, lineInTextIndex, noTextLineIndex, rotate_param = 2, padding = 0):
       
        dim = image.shape
        line2Text = {}
        
        for line_index in lineIndex2ClusterBoxes:
            boxes = lineIndex2ClusterBoxes[line_index]
            
            result = []
            for box in boxes:
                
                aspect_ratio = float((box[1]-box[0])) / (box[3]-box[2]+0.00001)
                subImage = image[max(0,box[0]-padding):min(dim[0],box[1]+padding), max(0,box[2]-padding):min(dim[1],box[3]+padding)]
                rotate = aspect_ratio > rotate_param
                text = PhyloParser.image2text(subImage, rotate = rotate, padding = 0)
                
                if not rotate and text != "":
                    result.append(text)
                            
            line2Text[anchorLines[line_index]] = {"text": result, "status": "from_box"}
                
        for line_index in lineInTextIndex:
            text = PhyloParser.recycleText(image, anchorLines[line_index])
            line2Text[anchorLines[line_index]] = {"text": [text], "status": "in_box"}
            
        for line_index in noTextLineIndex:
            text = PhyloParser.recycleText(image, anchorLines[line_index])
            line2Text[anchorLines[line_index]] = {"text": [text], "status": "no_box"}
        
        return line2Text
    
    
    @staticmethod
    # padding: enlarge box area
    def boxGroup2Text(image, textGroup, padding = 2):
        dim = image.shape
        boxID2Text={}
        box2Text={}
        for i, textBox in enumerate(textGroup):
            for j, box in enumerate(textBox["boxes"]):
                box_id = (i,j)
                aspect_ratio = float((box[1]-box[0])) / (box[3]-box[2]+0.00001)
                subImage = image[max(0,box[0]-padding):min(dim[0],box[1]+padding), max(0,box[2]-padding):min(dim[1],box[3]+padding)]
                rotate = aspect_ratio > 1
                text = PhyloParser.image2text(subImage, rotate = rotate, padding = padding)
                boxID2Text[box_id] = {"text": text, "rotate": rotate}
                box2Text[tuple(box)] = {"text": text, "rotate": rotate}
                
        return boxID2Text, box2Text
    
    @staticmethod
    # padding: enlarge box area
    # apply ocr on box
    def box2Text(image, boxes, rotate_param = 2, padding = 2):
        dim = image.shape
        box2Text={}
        for i, box in enumerate(boxes):
            aspect_ratio = float((box[1]-box[0])) / (box[3]-box[2]+0.00001)
            subImage = image[max(0,box[0]-padding):min(dim[0],box[1]+padding), max(0,box[2]-padding):min(dim[1],box[3]+padding)]
            rotate = aspect_ratio > rotate_param
            
            text = PhyloParser.image2text(subImage, rotate = rotate, padding = padding)
            box2Text[tuple(box)] = {"text": text, "rotate": rotate}
                
        return box2Text
    


    @staticmethod
    # padding: enlarge box area
    def image2text(img, enhance = 3, rotate = False, padding = 2):
        
        #increase size to increase accuracy of ocr

        dim = img.shape
        height = dim[0]
        width = dim[1]
        
        new_height = 36
        new_width = int(float(new_height)/height * width)
        
#         image = cv.resize(img,(new_width, new_height), interpolation = cv.INTER_CUBIC)
        image = cv.resize(img, None, fx=enhance, fy=enhance, interpolation = cv.INTER_CUBIC)        
        image = cv.copyMakeBorder(image, padding, padding, padding, padding, cv.BORDER_CONSTANT, value = 255)
        
        if rotate:
            image = PhyloParser.rotateImage(image)
        cv.imwrite("tmp.tiff", image)
        text = pytesseract.image_to_string(Image.open('tmp.tiff'))

        return text  


    @staticmethod
    # remove tall-thin box
    def refineContourBox(contourBox, threshold_aspect = 10, threshold_height = 20):
        
        newContourBoxes = []
        for box in contourBox:
            box_height = box[1]-box[0]
            aspect_ratio = float((box[1]-box[0])) / (box[3]-box[2]+0.00001)
            if aspect_ratio > threshold_aspect and box_height > threshold_height:
                pass
            else:
                newContourBoxes.append(box)
        return newContourBoxes
        
    @staticmethod
    # coef : coefficient of horizontal distance for sorting lines that corresponds to the same box
    # horizontal_anchor_margin: roll back distance to tolerate overlapping between line and box
    # verticle_anchor_margin: tolerance to y-displacement between line and box
    # threshold_cut: avoid cutting small sharebox with height smaller then this param. 
    #    In solution 1 for split sharebox, drop the small boxes
    #    In solution 2 for split sharebox, keep the size of sharebox
    
    def matchAnchorLineAndContourBox(image, anchorLines, contourBoxes, coef = 5, horizontal_anchor_margin = 5, verticle_anchor_margin = 2, threshold_text_height = 5, debug = False):

        # sort boxes and anchorlines from top to bot for further matching       
        contourBoxes = sorted(contourBoxes, key = lambda x: (x[0], x[2])) #top, left
        anchorLines = sorted(anchorLines, key = lambda x: (x[3], x[2])) #top, right      
        
        index = 0
        boxIndex2LineIndex = {}
        lineIndex2BoxIndex={}
        noTextLineIndex = []
        lineInTextIndex = []
        

        ####################################### 
        # match box and line
        for line_index, line in enumerate(anchorLines):
            y = line[1]
            right = line[2]
            target_box_index = []
            target_boxes = []
            box_found = False
            start_index = index
            
            while index < len(contourBoxes):
                countourBox = contourBoxes[index]

                if PhyloParser.isLineOverLapBox(line, countourBox):
                    if not box_found:
                        start_index = index
   
                    box_found = True
                    
                    # find matching
                    if right - horizontal_anchor_margin <= countourBox[2]:
                        target_boxes.append(countourBox)
                        target_box_index.append(index)
                        #push into boxIndex2LineIndex
                        if index in boxIndex2LineIndex:
                            boxIndex2LineIndex[index].append(line_index)
                        else:
                            boxIndex2LineIndex[index] = [line_index]
                            
                    index += 1
                    
                elif countourBox[0] < y + verticle_anchor_margin:
                    index += 1
                else:
                    index = start_index ## go back to the start_index, since the multiple lines can match with the same box
                    break
                
            if index == len(contourBoxes):
                index = start_index

            if len(target_box_index) > 0:
                lineIndex2BoxIndex[line_index] = target_box_index
            else:
                if box_found:
                    lineInTextIndex.append(line_index)
                else:
                    noTextLineIndex.append(line_index)

        #######################################
      
        # find orphan boxes: boxes that are not matched with any lines
        shareBoxIndices = Set()
        activateBoxIndices = []
        for box_index in boxIndex2LineIndex:
            if len(boxIndex2LineIndex[box_index]) > 1:
                shareBoxIndices.add(box_index)
            else:
                activateBoxIndices.append(box_index)
        
        orphanBoxIndices = Set(range(0, len(contourBoxes)))
        orphanBoxIndices = orphanBoxIndices - shareBoxIndices - Set(activateBoxIndices)
        
            
        if debug:
            print "activateBoxIndices", activateBoxIndices
            img = image.copy()
            for box_index in activateBoxIndices:
                b = contourBoxes[box_index]
                cv.rectangle(img,(b[2],b[0]),(b[3],b[1]),(0,125,0),2)
            PhyloParser.displayLines(img, anchorLines)
            
            print "shareBoxIndices", shareBoxIndices
            img = image.copy()
            for box_index in shareBoxIndices:
                b = contourBoxes[box_index]
                cv.rectangle(img,(b[2],b[0]),(b[3],b[1]),(0,125,0),2)
            PhyloParser.displayLines(img, anchorLines)
            
            print "orphanBoxIndices", orphanBoxIndices
            img = image.copy()
            for box_index in orphanBoxIndices:
                b = contourBoxes[box_index]
                cv.rectangle(img,(b[2],b[0]),(b[3],b[1]),(0,125,0),2)
            PhyloParser.displayLines(img, anchorLines)
        
        # handle multiple lines to the same box
        lineIndex2SubBoxes={}
        orphanBoxFromShareBox = []
        for box_index in shareBoxIndices:
            line_indices = boxIndex2LineIndex[box_index]
            box = contourBoxes[box_index]
            
            # determine using solution 1 or solution 2
            for line_index in line_indices:
                current_box_indices = lineIndex2BoxIndex[line_index]
                found_other_box = False
                for current_box_index in current_box_indices:
                    if current_box_index not in shareBoxIndices:
                        found_other_box = True

            # sorted the lin for solution2
            lines = []
            for line_index in line_indices:
                line = anchorLines[line_index]
                dist = math.sqrt((pow((line[3]-box[0]),2) + coef*pow((line[2]-box[2]),2)))
                lines.append((line, line_index, dist))
            
            lines = sorted(lines, key = lambda x: (x[2]))             
            line_indices = [y for x,y,z in lines]
            lines = [x for x,y,z in lines]
                  
            ceil = box[0]
            new_boxes = []
            for i in range(0, len(line_indices)):
                line_index = line_indices[i]
                line = anchorLines[line_index]
                
                # solution 1, use activated boxes
                current_box_indices = lineIndex2BoxIndex[line_index]
                top = 9999999
                bot = 0 
                found_other_box = False
                for current_box_index in current_box_indices:
                    if current_box_index not in shareBoxIndices:
                        found_other_box = True
                        current_box = contourBoxes[current_box_index]
                        top = min(top, current_box[0])
                        bot = max(bot, current_box[1])
                
                # has other activated box 
                # use sub-box created by solution 1
                if found_other_box:
                    sub_box = [max(top, box[0]), min(bot, box[1]), box[2], box[3]]
                    
                    # determine if the sub box is meaningful (height > threshold)
                    if sub_box[1] - sub_box[0] >= threshold_text_height:      
                        new_boxes.append(sub_box)
                        if line_index in lineIndex2SubBoxes:
                            lineIndex2SubBoxes[line_index].append(sub_box)          
                        else:
                            lineIndex2SubBoxes[line_index] = [sub_box]
                
                    # DEBUG
#                     print "show sharebox"
#                     img = image.copy()
#                     b = box
#                     cv.rectangle(img,(b[2],b[0]),(b[3],b[1]),(0,125,0),2)
#                     PhyloParser.displayLines(img, [line])########## show other box
#                      
#                     print "show remain box", sub_box
#                     img = image.copy()
#                     b = sub_box
#                     cv.rectangle(img,(b[2],b[0]),(b[3],b[1]),(0,125,0),2)
#                     PhyloParser.displayLines(img, [line])########## show other box
                

                # solution 2 use anchorlines to split
                # try to sort line from top to bot, more weight in x axis to make left box in behind
                if i != len(line_indices)-1:
                    next_line_index = line_indices[i+1]
                    next_line = anchorLines[next_line_index]
                      
                    # next line under this line
                    # this line's head < next line' tail
                    if line[1] < next_line[1] and line[0] < next_line[2]:
                        floor = (lines[i+1][1] + lines[i][1])/2
                        next_ceil = floor
                    #next line above this line, this line is the bottom line in this tier
                    else:
                        floor = box[1]
                        next_ceil = box[0]
                #last line            
                else:
                    floor = box[1] 
                 
                # not split the sharebox if it's height <= threshold
                if floor-ceil <= threshold_text_height:
                    new_box = box
                else:
                    new_box = [ceil, floor, box[2], box[3]]
                
                # use sub-box created by solution 2
                if not found_other_box:
                    new_boxes.append(new_box)
                    if line_index in lineIndex2SubBoxes:
                        lineIndex2SubBoxes[line_index].append(new_box)          
                    else:
                        lineIndex2SubBoxes[line_index] = [new_box]
                 
                ceil = next_ceil
                 
            # extract non-matched sub-boxes
            orphanBoxFromShareBox = []
            new_boxes = sorted(new_boxes, key = lambda x: (x[0], x[2])) #top, left
            new_orphan_boxes = []
            split_y = [box[0]]
            for new_box in new_boxes:
                if new_box[0] > split_y[-1]:
                    new_orphan_boxes.append([split_y[-1], new_box[0], box[2], box[3]])
                split_y.append(new_box[1])
            
            if split_y[-1] < box[1]:
                new_orphan_boxes.append([split_y[-1], box[1], box[2], box[3]])
            
            orphanBoxFromShareBox += new_orphan_boxes
            
        # DEBUG
#         print 
#         print "show share box result"
#         img = image.copy()
#         for line_index in lineIndex2SubBoxes: 
#             line = anchorLines[line_index]
#             boxes = lineIndex2SubBoxes[line_index]
#             for b in boxes:
#                 cv.rectangle(img,(b[2],b[0]),(b[3],b[1]),(0,125,0),2)
#         PhyloParser.displayLines(img, [line])
#         
#         print "show orphanBoxFromShareBox"
#         img = image.copy()
#         for b in orphanBoxFromShareBox: 
#             cv.rectangle(img,(b[2],b[0]),(b[3],b[1]),(0,125,0),2)
#         PhyloParser.displayLines(img, [line])


#         print "show sub-box", lineIndex2SubBoxes
#         img = image.copy()
#         for line_index in lineIndex2SubBoxes:
#             boxes = lineIndex2SubBoxes[line_index]
#             for b in boxes:
#                 cv.rectangle(img,(b[2],b[0]),(b[3],b[1]),(0,125,0),2)
#         PhyloParser.displayLines(img, anchorLines)
        
        
        # merge all boxes of anchor lines
        lineIndex2ClusterBoxes= {}
        for line_index in lineIndex2BoxIndex:
            box_group = []
            
            # monopole box
            for box_index in lineIndex2BoxIndex[line_index]:
                if not box_index in shareBoxIndices:
                    box_group.append(contourBoxes[box_index])
            
            # sub box from share box
            if line_index in lineIndex2SubBoxes:
                for box in lineIndex2SubBoxes[line_index]:
                    box_group.append(box)

            # merge box
            boxes = PhyloParser.mergeBoxes(image, box_group)
            lineIndex2ClusterBoxes[line_index] = boxes
            
        return lineIndex2ClusterBoxes, lineIndex2BoxIndex, lineIndex2SubBoxes, lineInTextIndex, noTextLineIndex, orphanBoxFromShareBox, activateBoxIndices, shareBoxIndices, orphanBoxIndices
    
    @staticmethod 
    def clusterOrphanBox(image, anchorLines, contourBoxes, orphanBoxFromShareBox, orphanBoxIndices):
        
        orphanBoxes = orphanBoxFromShareBox
        for box_index in orphanBoxIndices:
            orphanBoxes.append(contourBoxes[box_index])
            
        orphanBoxes = sorted(orphanBoxes, key = lambda x: (x[0], x[2])) #top, left
        
        height_group, box_group, out_boxes = PhyloParser.clusterBoxes(image, orphanBoxes)
                
        new_box_group = []
        for boxes in box_group:
            new_box_group += PhyloParser.mergeBoxes(image, boxes)

        return new_box_group
        
       
    @staticmethod
    def isLineOverLapBox(line, box, verticle_anchor_margin = 5):
        y = line[1]
        box1_top = y-verticle_anchor_margin
        box1_bot = y+verticle_anchor_margin
        
        box2_top = box[0]
        box2_bot = box[1]
        
        return (box1_top <= box2_bot <= box1_bot) or (box1_top <= box2_top <= box1_bot) or (box2_top <= box1_bot <= box2_bot) or (box2_top <= box1_top <= box2_bot)
        
        
        
    @staticmethod
    #coef : coefficient of horizontal distance for sorting lines that corresponds to the same box
    #horizontal_anchor_margin: roll back distance to tolerate overlapping between line and box
    #verticle_anchor_margin: tolerance to y-displacement between line and box
    def matchAnchorLineAndTextBox(image, anchorLines, textGroup, coef = 10, horizontal_anchor_margin = 5, verticle_anchor_margin = 2):

        index = 0
        start_index = index
        boxIndex2LineIndex = {}
        lineIndex2BoxIndex={}
        noTextLineIndex = []
        lineInTextIndex = []
        
        for line_index, line in enumerate(anchorLines):
            y = line[1]
            right = line[2]
            target_box_index = []
            target_boxes = []
            row_found = False
            
#             print "line", line, "startline", index, "/", len(textGroup)
    
            while index < len(textGroup):
                textBox = textGroup[index]
                text_row = textBox["text_row"]
                
#                 print "text_row", text_row, index
    
                if  text_row[0] <= y - verticle_anchor_margin <= text_row[1] \
                    or text_row[0] <= y <= text_row[1] \
                    or text_row[0] <= y + verticle_anchor_margin <= text_row[1]:
                    
#                     print "in, index", index
                    row_found = True
                    start_index = index
#                     print "start_index", index
                    for i, box in enumerate(textBox["boxes"]):
                        
                        # find corresponding box
                        # (give a small margin to tolerate the overlapping between line and box)
                        # will get every box in the right hand side
                        if right - horizontal_anchor_margin <= box[2]:
#                             print "in box", box
                                                
                            target_boxes.append(box)
                            target_box_index.append((index, i))
                            
                            #push into boxIndex2LineIndex
                            if (index, i) in boxIndex2LineIndex:
                                boxIndex2LineIndex[(index, i)].append(line_index)
                            else:
                                boxIndex2LineIndex[(index, i)] = [line_index]
                            
                    index += 1
                elif text_row[0] < y + verticle_anchor_margin:
                    index += 1
                else:
                    index = start_index ## go back to the start_index, since the multiple lines can match with the same box
                    break
                
            if index == len(textGroup):
                index = start_index

            if len(target_box_index) > 0:
                lineIndex2BoxIndex[line_index] = target_box_index
            else:
                if row_found:
                    lineInTextIndex.append(line_index)
                else:
                    noTextLineIndex.append(line_index)
        
#             print "target_box_index", target_box_index
#             print 
#             img = image.copy()
#             PhyloParser.displayLines(img, [line])
            
     
        # handle multiple lines to the same box
        robbedLineIndex = Set()
        lineIndex2ShareBox={}
        
        for key in boxIndex2LineIndex:
            index = key[0]
            box_index = key[1]
            box = textGroup[index]["boxes"][box_index]
            line_indices = boxIndex2LineIndex[key]
            
            # multiple lines take the same text box
            if len(line_indices) > 1:
                
                ## select the line among multiple matched lines
                lines = []
                for line_index in line_indices:
                    line = anchorLines[line_index]
                    dist = math.sqrt((pow((line[3]-box[0]),2) + coef*pow((line[2]-box[2]),2)))
                    lines.append((line, line_index, dist))

                lines = sorted(lines, key = lambda x: (x[2]))       
                         
                line_indices = [y for x,y,z in lines]
                lines = [x for x,y,z in lines]
                
#                 print "multiple", lines
                
                ceil = box[0]
                for i in range(0, len(line_indices)):
                    line_index = line_indices[i]
                    line = anchorLines[line_index]
             
#                     print "line", line, "box", box
                    if i != len(line_indices)-1:
                        next_line_index = line_indices[i+1]
                        next_line = anchorLines[next_line_index]
#                         print "next line", next_line
                        
                        # next line under this line
                        # this line's head < next line' tail
                        if line[1] < next_line[1] and line[0] < next_line[2]:
#                             print "under"
                            floor = (lines[i+1][1] + lines[i][1])/2
                            next_ceil = floor
                        #next line above this line, this line is the bottom line in this tier
                        else:
#                             print "above"
                            floor = box[1]
                            next_ceil = box[0]
                            
#                         print "floor", floor
#                         print "ceil", ceil
                    #last line            
                    else:
#                         print "last line"
                        floor = box[1] 
                                   
                    if line_index in lineIndex2ShareBox:
#                         print "existed"
                        new_box = lineIndex2ShareBox[line_index]
                        new_box = [min(ceil, new_box[0]), max(floor, new_box[1]), min(box[2], new_box[2]), max(box[3], new_box[3])]

                    else:
#                         print "not existed"
                        new_box = [ceil, floor, box[2], box[3]]
                            
#                     print "new box", new_box
                    lineIndex2ShareBox[line_index] = new_box
                    ceil = next_ceil
                   

                ############NOT USE#############
                ## select the line among multiple matched lines
                closest_distance = 999999999
#                 selected_index = 0
                for line_index in line_indices:
                    line = anchorLines[line_index]                   
                    distance = box[2] - (line[2] - horizontal_anchor_margin)    
                    if distance < closest_distance:
                        closest_distance = distance
                        selected_line_index = line_index

#                 boxIndex2LineIndex[key] = [selected_line_index] ## NOT UPDATE
                
                for line_index in line_indices:
                    if line_index != selected_line_index:
                        if line_index in lineIndex2BoxIndex:
#                             del lineIndex2BoxIndex[line_index]  ## NOT DELETE
                            robbedLineIndex.add(line_index)
                ############NOT USE#############
        
        # collect orphan box
        orphanBoxIndex = Set()
        for i, dict in enumerate(textGroup):
            for j, box in enumerate(dict["boxes"]):
                orphanBoxIndex.add((i,j))
        
        mathedBoxIndex = Set()
        for box_index in boxIndex2LineIndex:
            mathedBoxIndex.add(box_index)
        
        orphanBoxIndex -= mathedBoxIndex
        orphanBoxIndex = list(orphanBoxIndex)
        
        return lineIndex2BoxIndex, boxIndex2LineIndex, lineIndex2ShareBox, lineInTextIndex, noTextLineIndex, orphanBoxIndex
            
        
    @staticmethod
    # padding: enlarge box area
    # margin: height of scan zone after line with no assigned box
    def recycleText(image, line, margin = 5):

        dim = image.shape   
        y = line[1]
        right = line[2]
        subImage = image[max(0,y-margin):min(dim[0], y+margin), right:]
        subImage = cv.resize(subImage, None, fx=2, fy=2, interpolation = cv.INTER_CUBIC)      
        cv.imwrite("tmp.tiff", subImage)
        text = pytesseract.image_to_string(Image.open('tmp.tiff'))

        return text

    
    @staticmethod
    # cluster boxes that are entangled
    def clusterBoxes(image, contourBoxes, threshold_aspect = 10, threshold_height = 15): 
        
        contour_group = [] # a list of groups of contour boxes 
        height_group = [] # a list of groups of box height corresponding to contour_group

        textBoxes = [] # a list of outer bonding box corresponding to contour_group
            
        ## group contour boxes
        index = 0
        while index < len(contourBoxes):
            box = contourBoxes[index]
            box_height = box[1] - box[0]
            top = box[0]
            bot = box[1]
            left = box[2]
            right = box[3]
            aspect_ratio = float((box[1]-box[0])) / (box[3]-box[2]+0.00001)
            
            index += 1
            if aspect_ratio > threshold_aspect and box_height > threshold_height:
                pass
            else: 
                tmp = [box] # temporary list of box
                tmp_h = [box_height] # temporary list of box height
                break
            
        while index < len(contourBoxes):
            box =  contourBoxes[index]
            aspect_ratio = float((box[1]-box[0])) / (box[3]-box[2]+0.00001)
            box_height = box[1] - box[0]
            
#             ##DUB
#             print "box", index, "/", len(contourBoxes), box
#             img = image.copy()
#             cv.rectangle(img,(box[2],box[0]),(box[3],box[1]),(0,125,0),2)
#             PhyloParser.displayImage(img)
            
            if aspect_ratio > threshold_aspect and box_height > threshold_height:
                index += 1
                pass
            else:
                # current overlaps the previous box or entirely locates in the previous box
                if box[0] <= bot:
                    tmp.append(box)
                    tmp_h.append(box[1] - box[0])
                    top = min(top, box[0])
                    bot = max(bot, box[1])
                    left = min(left, box[2])
                    right = max(right, box[3])

                else:
                    contour_group.append(tmp)
                    height_group.append(tmp_h)
                    textBoxes.append([top, bot, left, right])
                    box_height = box[1] - box[0]
                    top = box[0]
                    bot = box[1]
                    left = box[2]
                    right = box[3]
                    tmp = [box]
                    tmp_h = [box_height]
                
                index += 1
                    
            # append the last group
            if index == len(contourBoxes):
                contour_group.append(tmp)
                height_group.append(tmp_h)
                textBoxes.append([top, bot, left, right]) 
                     
        return height_group, contour_group, textBoxes
    
    @staticmethod
    def mergeBoxes(image, boxes, split_threshold = 10):
 
        boxes = sorted(boxes, key = lambda x: (x[2], x[0])) #left, top
        
        length = len(boxes)
        matrix_graph = np.zeros((length, length), dtype = np.uint8)
        matrix_overlap = np.zeros((length, length), dtype = np.uint8)
        
        img = image.copy() ############
        # make matrix
        for j in range(0, length):
            box_1 = boxes[j]
            cv.rectangle(img,(box_1[2],box_1[0]),(box_1[3],box_1[1]),(0,125,0),2)
            
            for k in range(j, length):
                box_2 = boxes[k]
                
                is_overlap = PhyloParser.isOverLap(box_1, box_2)
                
                if j == k:
                    is_attached = True
                if j < k: # j <-> k
                    is_attached = box_2[2] - box_1[3] <= split_threshold
                else:    # k <-> j
                    is_attached = box_1[2] - box_2[3] <= split_threshold
                    
                if is_overlap and is_attached:
                    matrix_graph[j, k] = 1
                    matrix_graph[k, j] = 1
                elif is_overlap:
                    matrix_overlap[j, k] = 1
                    matrix_overlap[k, j] = 1
                
                if j == k: #handle width = 0 or height = 0 case
                    matrix_graph[j, k] = 1
                    matrix_overlap[k, j] = 1
        
#         print "matrix_graph"
#         print matrix_graph
#         print "matrix_overlap"
#         print matrix_overlap
#         print "matrix_relative"
        matrix_relative = matrix_graph + matrix_overlap
#         print matrix_relative
        matrix_sum = np.sum(matrix_graph, axis = 0)
        
#         print matrix_sum
#         PhyloParser.displayImage(img) #############################################
        
        # merging boxes that are attached
        # from left to right
        new_boxes = boxes[:]
        for j in range(0, length):
            if matrix_sum[j] <= 20:
                targets = np.argwhere(matrix_graph[j,:] == 1)
                box = None
                for t in targets:
                    index = t[0]                    
                    if box is None:
                        box = new_boxes[index]
                    else:
                        box = PhyloParser.mergeBox(box, new_boxes[index])
                
                # update box
                new_boxes[j] = box #update current box
                for t in targets:
                    index = t[0]
                    new_boxes[index] = box  #update target box
        
        #from right to left
        for j in range(length-1, -1, -1):
            if matrix_sum[j] <= 20:
                targets = np.argwhere(matrix_graph[j,:] == 1)
                box = None
                for t in targets:
                    index = t[0]
                    if box is None:
                        box = new_boxes[index]
                    else:
                        box = PhyloParser.mergeBox(box, new_boxes[index])
                
                # update box
                new_boxes[j] = box #update current box
                for t in targets:
                    index = t[0]
                    new_boxes[index] = box  #update target box
        
        # remove duplicate box
        tmp = Set()
        for box in new_boxes:
            tmp.add(tuple(box))
        
        new_boxes = list(tmp)        
        new_boxes = sorted(new_boxes, key = lambda x: (x[2], x[0])) #left, top
            
        return new_boxes
                    
    @staticmethod
    def mergeBox(box_1, box_2):
        box = [min(box_1[0], box_2[0]), max(box_1[1], box_2[1]), min(box_1[2], box_2[2]), max(box_1[3], box_2[3])]
        return box
    
    @staticmethod
    def isOverLap(box1, box2):
        box1_top = box1[0]
        box1_bot = box1[1]
        
        box2_top = box2[0]
        box2_bot = box2[1]
        
        return (box1_top < box2_bot <= box1_bot) or (box1_top <= box2_top < box1_bot) or (box2_top < box1_bot <= box2_bot) or (box2_top <= box1_top < box2_bot)
    
    @staticmethod
    def isNeighbor(box1, box2, threshold = 10):
        return abs(box1[2]-box2[3]) <= 10 or abs(box2[2]-box1[3]) <= 10
        


    @staticmethod
    # return the top, bottom, left, right value of the contour
    def getContourInfo(contour):
        max_location = np.max(np.max(contour, axis=1), axis = 0)
        min_location = np.min(np.min(contour, axis=1), axis = 0)
        
        box = [min_location[1], max_location[1], min_location[0], max_location[0]]
        return box
 
    ## end static method for getSpecies ## 
           
    
    
    
    
    def constructTree(self, image_data, tracing = False, debug = False):
        
        if image_data.lineDetected and image_data.lineMatched:
            
            # Pair h-v branch (parent) and v-h branch (children)
            image_data = PhyloParser.createNodesFromLineGroups(image_data, tracing)
            if debug:
                print "Display Nodes"
                image_data.displayNodes()

            # Create Tree
            image_data = self.createRootList(image_data)
            if debug:
                print "display Tree"
                image_data.displayTrees('regular')
                
            ## found no node
            if len(image_data.rootList) == 0:
                image_data.treeHead = None
                image_data.treeStructure = ""
                return image_data
            
# ------------------------------------------------------------------------ #

#             if tracing:
#                 print "tracing"
#                 image_data = self.connectRootByTracing(image_data)            
#                 image_data = PhyloParser.getBreakSpot(image_data)

# ------------------------------------------------------------------------ #

            ## verified leaves
            image_data.rightVerticalLineX = PhyloParser.getRightVerticalLineX(image_data.image, image_data.rootList)
            image_data.avg_anchorLine_x =  PhyloParser.getAvgRightEndPointOfLines(image_data.anchorLines)
            image_data = self.labelSuspiciousAnchorLine(image_data, useClf=True)
            
            if debug:
                for i, node in enumerate(image_data.rootList):
                    print i, node.branch
                    print "verifiedAnchorLines", len(node.verifiedAnchorLines)
    #                 PhyloParser.displayLines(image_data.image, node.verifiedAnchorLines)
                    print "suspiciousAnchorLines", len(node.suspiciousAnchorLines)
    #                 PhyloParser.displayLines(image_data.image, node.suspiciousAnchorLines)
                    print "unsureAnchorLines", len(node.unsureAnchorLines)
    #                 PhyloParser.displayLines(image_data.image, node.unsureAnchorLines)
                    print ""

            print "refineAnchorLine"
            image_data = PhyloParser.refineAnchorLine(image_data)


# ------------------------------------------------------------------------ #

            ###########################################
            ### Recover missing components ############
            
            ## directly connect right sub-trees of broken point
            if not self.isTreeReady(image_data):#######
                ## Fix false-positive sub-trees and connect sub-trees
                image_data = self.fixTrees(image_data)

#                 print image_data.rootList
                if debug:
                    print "display Fixed Tree"
                    image_data.displayTrees('regular')
                    

            if tracing:
                print "tracing"
                image_data = self.connectRootByTracing(image_data)            
                image_data = PhyloParser.getBreakSpot(image_data)


            ## use orphane box to recover line
            # sort again to ensure the first root is the largest
            image_data.rootList = sorted(image_data.rootList, key = lambda x: -x.numNodes)
            if len(image_data.rootList[0].breakSpot) > 0 and image_data.speciesNameReady:
                print "recoverInterLeaveFromOrphanBox"
                image_data = PhyloParser.recoverInterLeaveFromOrphanBox(image_data) ## not test yet
                if debug:
                    print "recoverInterLeaveFromOrphanBox result"
                    image_data.displayTrees('regular')

# ------------------------------------------------------------------------ #

            # select largest sub-tree as the final tree
            image_data.defineTreeHead()

            # merge tree structure and species text
            useText = False
            if image_data.speciesNameReady:
#                 print "mergeTreeAndText"
                self.mergeTreeAndText(image_data)
#                 print "end mergeTreeAndText"
                useText = True
                
            image_data.treeStructure = self.getTreeString(image_data, useText=useText)
            if debug:
                print image_data.treeStructure
                image_data.displayTrees('final')                
        else:
            print "Error! Tree components are not found. Please do detectLine and matchLine before this method"
        
        return image_data



    ### For evaluation
    def constructTree_eval(self, image_data, fixTree = False, tracing = False, orphanHint = False, debug = False):
        
        if image_data.lineDetected and image_data.lineMatched:
            
            # Pair h-v branch (parent) and v-h branch (children)
            image_data = PhyloParser.createNodesFromLineGroups(image_data, tracing)
            if debug:
                print "Display Nodes"
                image_data.displayNodes()

            # Create Tree
            image_data = self.createRootList(image_data)
            if debug:
                print "display Tree"
                image_data.displayTrees('regular')
                
            ## found no node
            if len(image_data.rootList) == 0:
                image_data.treeHead = None
                image_data.treeStructure = ""
                return image_data

# ------------------------------------------------------------------------ #

            # if tracing:
            #     print "tracing"
            #     image_data = PhyloParser.getBreakSpot(image_data)
            #     image_data = self.connectRootByTracing(image_data)            
            #     image_data = PhyloParser.getBreakSpot(image_data)
                  
 
# ------------------------------------------------------------------------ #

            ## verified leaves
            image_data.rightVerticalLineX = PhyloParser.getRightVerticalLineX(image_data.image, image_data.rootList)
            image_data.avg_anchorLine_x =  PhyloParser.getAvgRightEndPointOfLines(image_data.anchorLines)
            image_data = self.labelSuspiciousAnchorLine(image_data, useClf=True)
            
            if debug:
                for i, node in enumerate(image_data.rootList):
                    print i, node.branch
                    print "verifiedAnchorLines", len(node.verifiedAnchorLines)
    #                 PhyloParser.displayLines(image_data.image, node.verifiedAnchorLines)
                    print "suspiciousAnchorLines", len(node.suspiciousAnchorLines)
    #                 PhyloParser.displayLines(image_data.image, node.suspiciousAnchorLines)
                    print "unsureAnchorLines", len(node.unsureAnchorLines)
    #                 PhyloParser.displayLines(image_data.image, node.unsureAnchorLines)
                    print ""

            image_data = PhyloParser.refineAnchorLine(image_data)


# ------------------------------------------------------------------------ #

            ###########################################
            ### Recover missing components ############
            
            ## directly connect right sub-trees of broken point
            if not self.isTreeReady(image_data) and fixTree:#######
                ## Fix false-positive sub-trees and mandatorily connect sub-trees
                print "fixTrees"    
                image_data = self.fixTrees(image_data)

#                 print image_data.rootList
                if debug:
                    print "display Fixed Tree"
                    image_data.displayTrees('regular')
            

# ------------------------------------------------------------------------ #

            if tracing:
                print "tracing"
                image_data = PhyloParser.getBreakSpot(image_data)
                image_data = self.connectRootByTracing(image_data)            
                image_data = PhyloParser.getBreakSpot(image_data)
                
 # ------------------------------------------------------------------------ #           
            ## use orphane box to recover line
            # sort again to ensure the first root is the largest
            image_data.rootList = sorted(image_data.rootList, key = lambda x: -x.numNodes)
            if len(image_data.rootList[0].breakSpot) > 0 and image_data.speciesNameReady and orphanHint:
#                 print "recoverInterLeaveFromOrphanBox"    
                # image_data = self.recoverInterLeaveFromOrphanBox(image_data) ## not test yet
                if debug:
                    print "recoverInterLeaveFromOrphanBox result"
                    image_data.displayTrees('regular')

# ------------------------------------------------------------------------ #

            # select largest sub-tree as the final tree
            image_data.defineTreeHead()

            # merge tree structure and species text
            useText = False
            if image_data.speciesNameReady:
                self.mergeTreeAndText(image_data)
                useText = True
                
            image_data.treeStructure = self.getTreeString(image_data, useText=useText)
            if debug:
                image_data.displayTrees('final')                
        else:
            print "Error! Tree components are not found. Please do detectLine and matchLine before this method"
        
        return image_data


    # find the root from nodeList
    def createRootList(self, image_data):
        nodeList = image_data.nodeList
        anchorLines = image_data.anchorLines
        seen = []

        rootList = []
        count = 0

        for node in nodeList:
            if node not in seen:

                stack = []
                count +=1
                # print count
                rootNode = None
                stack.append(node)
                foundRoot = False
                while stack:

                    subnode = stack.pop()

                    if subnode in seen:
                        break               
                    else:
                        seen.append(subnode)
                        if subnode.whereFrom:                     
                            stack.append(subnode.whereFrom)
                        else:
                            foundRoot = True
                            subnode.isRoot = True
                            rootNode = subnode
                            rootList.append(subnode)


                if foundRoot:
                    (seen, loop) = self.groupNodes(rootNode, seen, image_data)

                    # if loop[0]:
                    #     rootList.remove(rootNode)


        # rootList = sorted(rootList, key = lambda x: -x.numNodes)

        for rootNode in rootList:
            rootNode.breakSpot = list(set(rootNode.breakSpot))


        image_data.rootList = rootList
        return image_data 
    
    
    # group nodes representing the same nodes
    def groupNodes(self, rootNode, seen, image_data):

        anchorLines = image_data.anchorLines
        stack = []
        visit = []
        lineList = []
        visit.append(rootNode)
        lineList.append(rootNode.branch)
        rootNode.origin = rootNode
        isComplete = True


        if rootNode.to[0]:
            if rootNode.branch != rootNode.to[0].branch:
                stack.append(rootNode.to[0])
            else:
                tmpNode = list(rootNode.to)
                tmpNode[0] = None
                rootNode.to = tuple(tmpNode)            
        else:
            isAnchorLine = self.checkError(rootNode, 'upper', image_data)
            if isAnchorLine:
                lineList.append(rootNode.upperLeave)
            else:
                if rootNode.to[0]:
                    stack.append(rootNode.to[0])
                else:
                    isComplete = False
                    lineList.append(rootNode.branch)
        if rootNode.to[1]:
            if rootNode.branch != rootNode.to[1].branch:
                stack.append(rootNode.to[1])
            else:
                tmpNode = list(rootNode.to)
                tmpNode[1] = None
                rootNode.to = tuple(tmpNode)
        else:
            isAnchorLine = self.checkError(rootNode, 'lower', image_data)
            if isAnchorLine:
                lineList.append(rootNode.lowerLeave)
            else:
                if rootNode.to[1]:
                    stack.append(rootNode.to[1])
                else:
                    isComplete = False
                    lineList.append(rootNode.branch)
        if not rootNode.isBinary:

            for index, to in enumerate(rootNode.otherTo):
                if to:
                    if rootNode.branch != to.branch:
                        stack.append(to)
                    else:
                        rootNode.otherTo[index] = None
                else:
                    isAnchorLine = self.checkError(rootNode, 'inter%s' %str(index), image_data)
                    if isAnchorLine:
                        anchorLines.append(rootNode.interLeave[index])
                        lineList.append(rootNode.interLeave[index])
                    else:
                        if rootNode.otherTo[index]:
                            stack.append(rootNode.otherTo[index])
                        else:
                            if isComplete:
                                isComplete = False
                                lineList.append(rootNode.branch)
        numNodes = 1



        while stack:

            numNodes +=1
            node = stack.pop()
            # node.getNodeInfo()
            # self.drawNode(whatever, node)
            # plt.imshow(whatever)
            # plt.show()
            visit.append(node)
            node.origin = rootNode
            if node.to[0] :


                if node.to[0] not in seen:
                    seen.append(node.to[0])
                if node.to[0] not in visit and node.branch != node.to[0].branch:
                    stack.append(node.to[0])
                else:
                    loop = [True, node]


                    tmpNode = list(node.to)
                    tmpNode[0] = None
                    node.to = tuple(tmpNode)
                    # print 'haaaa'
                    # return (seen, loop)
            else:
                if not self.checkError(node,'upper',image_data):
                    isComplete = False
                    if node.to[0]:
                        stack.append(node.to[0])
                    else:
                        lineList.append(node.branch)
                else:
                    lineList.append(node.upperLeave)
                

            if node.to[1]:

                if node.to[1] not in seen:
                    seen.append(node.to[1])
                if node.to[1] not in visit and node.branch != node.to[1].branch:

                    stack.append(node.to[1])
                else:


                    loop = [True, node]
                    tmpNode = list(node.to)
                    tmpNode[1] = None
                    node.to = tuple(tmpNode)

                    # return (seen, loop)
            else:

                if not self.checkError(node, 'lower', image_data):
                    isComplete = False
                    if node.to[1]:
                        stack.append(node.to[1])
                    else:
                        lineList.append(node.branch)
                else:
                    lineList.append(node.lowerLeave)

            if not node.isBinary:

                for index, to in enumerate(node.otherTo):

                    if to:

                        if to not in seen:
                            seen.append(to)
                        if to not in visit and node.branch != to.branch:
                            stack.append(to)
                        else:
                            node.otherTo[index] = None
                    else:

                        isAnchorLine = self.checkError(node, 'inter%s' %str(index), image_data)
                        if isAnchorLine:
                            anchorLines.append(node.interLeave[index])
                            lineList.append(node.interLeave[index])
                        else:
                            if node.otherTo[index]:
                                stack.append(node.otherTo[index])
                            else:
                                if isComplete:
                                    isComplete = False
                                    lineList.append(node.branch)




        rootNode.numNodes = numNodes


        area = self.countArea(lineList, image_data)

        rootNode.area = area

        rootNode.nodesIncluded = visit
        if isComplete:
            rootNode.isComplete = True
        loop = False, None
        return (seen, loop)    


    @staticmethod
    def countArea(lineList,image_data):

        lineList = sorted(lineList, key = image_data.sortByLeftTop)

        leftTop = lineList[0]
        x1 = leftTop[0]
        y1 = leftTop[1]
        lineList = sorted(lineList, key = image_data.sortByBtmRight)
        btmRight = lineList[0]
        x2 = btmRight[2]
        y2 = btmRight[3]
        area = abs(y2-y1) *abs(x2-x1) 
        if area==0:
            x1 = leftTop[2]
            y1 = leftTop[3]
            x2 = btmRight[2]
            y2 = btmRight[3]
            area = abs(y2-y1) *abs(x2-x1) 


        return area

    def checkError(self, node, mode , image_data):
        
        # node.getNodeInfo()
        # print mode
        anchorLines = image_data.anchorLines
        parent = image_data.parent
        if mode == 'upper':
            if node.upperLeave:
                if node.isUpperAnchor:
                    return True
                else:
                    # PhyloParser.recoverMissingSmallTrees(node, image_data)
                    for package in parent:
                        lines, dist = package
                        if self.isSameLine(lines[0], node.upperLeave):
                            newNode = Node(node.upperLeave, lines[1])
                            nodeTo = list(node.to)
                            nodeTo[0] = newNode
                            node.to = tuple(nodeTo)
                            if node.isRoot:
                                node.breakSpot.append(newNode)
                            else:
                                rootNode = node.origin
                                rootNode.breakSpot.append(node)
                            if PhyloParser.isNodeComplete(node):
                                if node in node.origin.breakSpot:
                                    node.origin.breakSpot.remove(node)
                            return False
            else:

                if node.isRoot:

                    node.breakSpot.append(node)
                else:
                    rootNode = node.origin
                    rootNode.breakSpot.append(node)
                return False

        elif mode == 'lower':
            if node.lowerLeave:
                if node.isLowerAnchor:
                    return True
                else:
                    # PhyloParser.recoverMissingSmallTrees(node, image_data)
                    # return False
                    for package in parent:
                        lines, dist = package
                        if self.isSameLine(lines[0], node.lowerLeave):
                            newNode = Node(node.lowerLeave, lines[1])
                            nodeTo = list(node.to)
                            nodeTo[1] = newNode
                            node.to = tuple(nodeTo)
                            if node.isRoot:
                                node.breakSpot.append(newNode)
                            else:
                                rootNode = node.origin
                                rootNode.breakSpot.append(node)                            
                            if PhyloParser.isNodeComplete(node):
                                if node in node.origin.breakSpot:
                                    node.origin.breakSpot.remove(node)
                            return False
            else:
                if node.isRoot:
                    node.breakSpot.append(node)
                else:
                    rootNode = node.origin
                    rootNode.breakSpot.append(node)
                return False
        elif mode[:5] == 'inter':

            index = int(mode[5:])
            if index < len(node.interLeave) and node.interLeave[index]:

                if node.isInterAnchor[index]:
                    return True
                else:
                    # PhyloParser.recoverMissingSmallTrees(node, image_data)
                    # return False
                    for package in parent:
                        lines, dist = package
                        if self.isSameLine(lines[0], node.interLeave[index]):
                            newNode = Node(node.interLeave[index], lines[1])
                            node.otherTo[index] = newNode
                            if node.isRoot:
                                node.breakSpot.append(newNode)
                            else:
                                rootNode = node.origin
                                rootNode.breakSpot.append(node)
                            if PhyloParser.isNodeComplete(node):
                                if node in node.origin.breakSpot:
                                    node.origin.breakSpot.remove(node)
                            return False
            else:
                if node.isRoot:
                    node.breakSpot.append(node)
                else:
                    rootNode = node.origin
                    rootNode.breakSpot.append(node)
                return False


    @staticmethod
    def isNodeComplete(node):
        if not (node.branch and node.upperLeave and node.lowerLeave):
            return False
        else:
            return True
        

    @staticmethod
    def isDotWithinLine(dot, line, mode = None):
        if not line:
            return False
        margin = 5
        x, y = dot
        x1, y1, x2, y2, length = line
        # print 'dot', x, x, y, y
        # print 'lin', x1 - margin, x2 + margin, y1-margin, y2+margin
        if mode == 'short':
            if x >= x1 and x< x2 + margin and y1 - margin < y and y < y2 + margin:
                return True
            else:
                return False

        else:
            if x1-margin < x and x < x2+margin and y1 - margin < y and y < y2 + margin:
                return True
            else:
                return False
    @staticmethod
    def isLefter(branch, ref):
        x1 = branch[0]
        x2 = ref[0]
        margin = 5
        if x2 + margin < x1:
            return True
        return False

    @staticmethod
    def sortNodeByLeftEnd(item):
        return item.branch[0]

    @staticmethod
    def getNodeBranchOnTheRight(breakNode, nodeList, mode):
        margin = 2
        if mode == 'upper':
            x = breakNode.branch[0]
            y = breakNode.branch[1]
        elif mode == 'lower':
            x = breakNode.branch[0]
            y = breakNode.branch[3]
        else:
            x = mode[0]
            y = mode[1]
        potentialNodes = []

#         print mode

        for node in nodeList:
            x1, y1, x2, y2, length = node.branch

#             print y1, y, y2, x, x1
            if y1 < y and y2 > y and x1>x - margin:
                potentialNodes.append(node)
            if mode == 'lower':
                if breakNode.lowerLeave and node.root:
                    if PhyloParser.isSameLine(breakNode.lowerLeave, node.root):
                        potentialNodes.append(node)
            elif mode == 'upper':
                if breakNode.upperLeave and node.root:
                    if PhyloParser.isSameLine(breakNode.upperLeave, node.root):
                        potentialNodes.append(node)
            else:
                if node.root:
                    if PhyloParser.isSameLine(mode, node.root):
                        potentialNodes.append(node)

#         print potentialNodes

        if len(potentialNodes) == 0:
            # for node in nodeList:
            #     x1, y1, x2, y2, length = node.branch
            #     if y1 < y and y2 > y and x1>x - margin:
            #         potentialNodes.append(node)
            #     if mode == 'lower':
            #         if breakNode.lowerLeave and node.root:
            #             if isSameLine(breakNode.lowerLeave, node.root):
            #                 potentialNodes.append(node)
            #     elif mode == 'upper':
            #         if breakNode.upperLeave and node.root:
            #             if isSameLine(breakNode.upperLeave, node.root):
            #                 potentialNode.append(node)
            
            return False
        else:
            potentialNodes = sorted(potentialNodes, key =PhyloParser.sortNodeByLeftEnd)
            return potentialNodes[0]



    @staticmethod
    # return orphan boxes that are at the right of the given vertical line
    # returned orphan boxes are sorted from top to bot in y 
    def matchVerticalBranchWithOrphanBox(verLine, orphanBox2Text, x_margin = 15, box_size_threshold = 3):
        
        x_left, y_top, x_right, y_bot, length = verLine

        matchBoxes= []
#         img = image.copy()
        for b, text in orphanBox2Text.items():

            b_top, b_bot, b_left, b_right = b
            
            ## determine if the box is meaningful                
            if b_right - b_left > box_size_threshold and b_bot - b_top > box_size_threshold and text['text'] is not None and text['text'] != "":
                
                ## determine if the box is matched 
                if ((b_top <= y_top and b_bot >= y_top) or (b_top <= y_bot and b_bot >= y_bot) or (b_top >= y_top and b_bot <= y_bot)) \
                and b_left >= x_right and b_left <= x_right + x_margin:
                    matchBoxes.append([b, text])
    #                 cv.rectangle(img,(b[2],b[0]),(b[3],b[1]),(0,125,0),2)
        
#         PhyloParser.displayLines(img, [verLine])
        
        matchBoxes = sorted(matchBoxes, key = lambda x: x[0][0])
        
        # remove duplicate (bugs from getSpecies_v3)
        i = 0
        while i < len(matchBoxes) - 1:
            current_b = matchBoxes[i][0]
            next_b = matchBoxes[i+1][0]
            
            if current_b[1] > next_b[0] and \
                ((next_b[2] < current_b[3] and next_b[3] >= current_b[3]) or \
                (next_b[2] <= current_b[2] and next_b[3] > current_b[2])):
                # find duplicate
                del matchBoxes[i+1]
            else:
                i += 1

        return matchBoxes
    
    @staticmethod
    # return boxes corresponding to the given vertex
    # returned boxes are sorted from left to right in x
    def getBoxCorrespond2Vertex(vertex, boxes, margin = 2):
        
        result = []
        for box in boxes:
            key = box[0]
            y = vertex[1]
            right = vertex[0]
                        
            if ((y-margin >= key[0] and y-margin <= key[1]) \
                or (y-margin <= key[1] and y+margin >= key[1]) \
                or (y+margin >= key[0] and y-margin <= key[1])) \
                and right <= key[2]:
                
                    result.append(box)        
        
        result = sorted(result, key = lambda x: x[0][2])
        return result
        

    @staticmethod
    # search right area 
    # return the leftest matched orphan box
    def matchComponentWithOrphanBox(component, orphanBox2Text, margin = 2):
        
        result = []
        for key in orphanBox2Text:
            
            if len(component) == 5: # input line
                y = component[1];
                right = component[2]
            else: # input point
                y = component[1]
                right = component[0]
    
            
            if ((y-margin >= key[0] and y-margin <= key[1]) \
            or (y-margin <= key[1] and y+margin >= key[1]) \
            or (y+margin >= key[0] and y-margin <= key[1])) \
            and right <= key[2]:
            
                result.append((orphanBox2Text[key], key[2]-right))
        
        if len(result) > 0:  
            result = sorted(result, key = lambda x: x[1])
            return result[0][0]["text"]
        else:
            return None
    
    @staticmethod
    def replaceEscapeChar(string):
        try:
            if string is not None:
                string = string.encode(encoding='UTF-8')
                string = string.replace("(", "[")
                string = string.replace(")", "]")
                string = string.replace(",", "|")
                string = string.replace(";", "|")
                string = string.replace(":", "|")
                if string == "":
                    return None
            return string
        except:
            #decode issue
            return None
        
    @staticmethod
    def mergeTreeAndText(image_data):
        
        node_list = [image_data.treeHead];
        
        while len(node_list) > 0:
            
            node = node_list.pop();
            
#             print "branch", node.branch
            
            if node.to[0] is None:
                name = None
                if node.upperLeave is not None: # leave found
                    if image_data.line2Text.has_key(node.upperLeave) and len(image_data.line2Text[node.upperLeave]['text']) > 0:
#                         print "node.upperLeave", node.upperLeave

                        name = image_data.line2Text[node.upperLeave]['text'][0]
#                         print "upperLeave:", name
#                     elif node.upperLabel is not None: #handle
#                         name = node.upperLabel
                    else:
                        name = PhyloParser.matchComponentWithOrphanBox(node.upperLeave, image_data.orphanBox2Text)
#                         print "upperLeave: search orphan", name
                        
                else:# leave not found
                    name = PhyloParser.matchComponentWithOrphanBox(node.branch[0:2], image_data.orphanBox2Text)
#                     print "branch top:", name
                    
                node.upperLabel = PhyloParser.replaceEscapeChar(name)
            else:
                node_list.append(node.to[0])
                

            if node.to[1] is None:
                name = None
                if node.lowerLeave is not None: # leave found
                    if image_data.line2Text.has_key(node.lowerLeave) and len(image_data.line2Text[node.lowerLeave]['text']) > 0:
#                         print "node.lowerLeave", node.lowerLeave
                        name = image_data.line2Text[node.lowerLeave]['text'][0]
#                         print "lowerLeave:", name
                    else:
                        name = PhyloParser.matchComponentWithOrphanBox(node.lowerLeave, image_data.orphanBox2Text)
#                         print "lowerLeave: search orphan", name
                else:# leave not found
                    name = PhyloParser.matchComponentWithOrphanBox(node.branch[2:4], image_data.orphanBox2Text)
#                     print "branch bot:", name

                node.lowerLabel = PhyloParser.replaceEscapeChar(name)
            else:
                node_list.append(node.to[1])
                
            
            if len(node.otherTo) > 0:
                for i, children in enumerate(node.otherTo):
                    name = None ####
                    if children is None:
                        if node.interLeave[i] is not None:# leave found (must)
                            if image_data.line2Text.has_key(node.interLeave[i]) and len(image_data.line2Text[node.interLeave[i]]['text']) > 0:
                                name = image_data.line2Text[node.interLeave[i]]['text'][0]
#                                 print "interLeave:", name
                            else:
                                name = PhyloParser.matchComponentWithOrphanBox(node.interLeave[i], image_data.orphanBox2Text)
#                                 print "interLeave: search orphan", name
                        else:# leave not found (not possible)
                            pass
                        
                    else:
                        node_list.append(children)
        
                    node.interLabel[i] = PhyloParser.replaceEscapeChar(name) ####
            
#         image_data.displayTrees('regular')
        
        
    @staticmethod
    # verify tree structure by text box
    # if tree structure with no matched box then this tree structure is probably false positive
    def checkRootWithTextBox(image_data):
        return image_data
        
        
    @staticmethod
    # convert tree structure to tree string
    def getTreeString(image_data, useText = False):
        return image_data.treeHead.getTreeString(useText = useText)



    @staticmethod
    # return a list describing the x position of the rightest vertical lines in each y
    def getRightVerticalLineX(img, rootList):
#         print "getRightVerticalLineX"
        rightVerticalLineX = np.zeros(img.shape[0])
          
        for root in rootList:
            descendants = [root]
            
            # traverse descedants
            while len(descendants) > 0:
                node = descendants.pop()

                # add children into descendants
                if node.to[0] is not None and node.to[0] != node: #prevent infinite loop
                    descendants.append(node.to[0])
#                     print "up node", node.to[0].branch
                if node.to[1] is not None and node.to[1] != node:
                    descendants.append(node.to[1])
#                     print "low node", node.to[1].branch
                if len(node.otherTo) > 0:
#                     print "num of other", len(node.otherTo)
                    for n in node.otherTo:
                        if n is not None and n != node:
                            descendants.append(n)
#                             print "mid node", n.branch
                        
                # update rightVerticalLineX
                y_top = node.branch[1]
                y_bot = node.branch[3]
                temp = np.zeros(img.shape[0])
                temp[y_top:y_bot+1] = node.branch[0]          
                rightVerticalLineX = np.maximum(temp, rightVerticalLineX)
         
#                 if len(descendants) == 0:
#                     break
                
        return rightVerticalLineX
    
    
    @staticmethod
    #return a patch for review line feature
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
    
    @staticmethod 
    #average x position of all given lines
    def getAvgRightEndPointOfLines(lines):
        x = 0
        if len(lines) > 0:
            for line in lines:
                x += line[2]
            
            avg_x = x/float(len(lines))
        else:
            avg_x = 0
        return avg_x
    
    @staticmethod
    # return line feature
    # use local region on the right of the line
    # use local verical region around the right end point of the line
    # use relative distance to the average x position of all anchor lines
    def getLineFeature(image, line, avg_line_x, hpatch_size = (3,15), vpatch_size = (9,3), x_margin = 7):
         
        border = 50
        extend_image = cv.copyMakeBorder(image, border, border, border, border, cv.BORDER_CONSTANT, value = 255)
       
        right_end = (line[3], line[2])
        
        hh = hpatch_size[0] / 2
        hpatch = extend_image[right_end[0]-hh + border:right_end[0]+hh+1 + border, right_end[1] + x_margin + border: border + right_end[1] + x_margin + hpatch_size[1]]
        
        hh = vpatch_size[0] / 2
        vpatch = extend_image[border+right_end[0]-hh:border+right_end[0]+hh+1, border+right_end[1]-vpatch_size[1]: border+right_end[1]]
        
        distanceToAvgX = (right_end[1] - avg_line_x)/image.shape[1]
        
        feature = np.hstack((hpatch.flatten(), vpatch.T.flatten())) / float(255)
        feature = np.hstack((distanceToAvgX, feature))
        
#         avgline = [int(avg_line_x), 0, int(avg_line_x), image.shape[0], 0]
#         print avgline
#         PhyloParser.displayLines(image, [avgline])

    #     viewPatch = getViewPatch(image, line)
        
        return feature

      
    # return 1: true from given classifier
    # return 2: false from given classifier
    def isAnchor(self, image_data, line, useClf = True):
        disagreements = np.array([0,0])
        if useClf and self.classifier is not None:
            label, prob = self.identifyAnchorLineByClassifier(image_data, line)
            return label[0], disagreements
        else:
            isSus, disagreements = self.isSuspicious(image_data, line)
            return 1-abs(isSus), disagreements
        
        
#     @staticmethod
    # given line, textboxes, and right vertical xs, heuristically determine suspicious line      
    # return 1: highly suspicious, there is a vertical line or subtree on the right of line
    # return -1: mediums suspicious, there is no matched text box
    # return 0: netiehr the above cases.  
    def isSuspicious(self, image_data, line):
        y = line[1]
        x_right = line[2]
        

        ### Use classifier
        if self.classifier is not None:
            label, prob = self.identifyAnchorLineByClassifier(image_data, line)
        
        
        ### Heuristic determination
        # 1. there is a vertical branch on it's right
        # 2. it does not have paired text
        isSus = 0
        if image_data.rightVerticalLineX[y] >= x_right:
            isSus = 1
        elif not image_data.line2Text.has_key(line):
            if PhyloParser.matchComponentWithOrphanBox(line, image_data.orphanBox2Text, margin = 2) is None:
                isSus = 1
            else:
                isSus = 0#-1
        else:
            isSus =  0
        
        
        
        ### Compare to classifier is available
        isAgreeByClassifier = 1
        disagreements = np.array([0, 0])
        if self.classifier is not None and abs(isSus) != abs(1- label[0]):
            isAgreeByClassifier = 0
            if label[0] == 1:
                #classifier says it's an anchorline
                disagreements[0] = 1
            else:
                #classifier says it's not an anchorline
                disagreements[1] = 1
        
        ##### DEBUG
#         if self.classifier is not None:
#             print "isSus:", isSus, "agree? ",  isAgreeByClassifier
#             print "classification:", label, prob
        ##### DEBUG
        
        return isSus, disagreements
    
    def identifyAnchorLineByClassifier(self, image_data, line):
        X = self.getLineFeature(image_data.image_preproc, line, image_data.avg_anchorLine_x)
        label = self.classifier.predict(X)
        predict_proba = self.classifier.predict_proba(X)
        
        return label, predict_proba
        
#     @staticmethod
    # return a list describing the x position of the rightest vertical lines in each y
    def labelSuspiciousAnchorLine(self, image_data, useClf = True):  
        
        rootList = image_data.rootList
        
        leave_count = 0
        disagreements = np.array([0, 0])
#         print "labelSuspiciousAnchorLine"
        for root in rootList:
            descendants = [root]
            anchorLines = []
            suspiciousAnchorLines = []
            verifiedAnchorLines = []
            unsureAnchorLines = []
            
            # traverse descedants
            while True:
                node = descendants.pop()
                node.biAnchorVerification = [7,7] # initialized as children node

                #### uppder leave
                # add children into descendants
                if node.to[0] is not None and node.to[0] != node: #prevent infinite loop:
                    descendants.append(node.to[0])
                    
                elif node.upperLeave is not None:
                    #this is anchorline candidates
#                     anchorLines.append(node.upperLeave)
                    
                    isTrueAnchor, d = self.isAnchor(image_data, node.upperLeave, useClf= useClf)
                    disagreements += d
                    leave_count += 1

#                     PhyloParser.displayLines(image_data.image, [node.upperLeave])
                    node.biAnchorVerification[0] = isTrueAnchor
                    if isTrueAnchor == 1: 
                        verifiedAnchorLines.append(node.upperLeave)
                    else:
                        suspiciousAnchorLines.append(node.upperLeave)
                
                    
                    
                ##### lower leave
                if node.to[1] is not None and node.to[1] != node: #prevent infinite loop:
                    descendants.append(node.to[1])
                    
                elif node.lowerLeave is not None:
                    #this is anchorline candidates
                    #anchorLines.append(node.lowerLeave)
                    
                    
                    isTrueAnchor, d = self.isAnchor(image_data, node.lowerLeave,  useClf= useClf)
                    disagreements += d
                    leave_count += 1

#                     PhyloParser.displayLines(image_data.image, [node.lowerLeave])
                    node.biAnchorVerification[1] = isTrueAnchor
                    if isTrueAnchor == 1: 
                        verifiedAnchorLines.append(node.lowerLeave)
                    else:
                        suspiciousAnchorLines.append(node.lowerLeave)
                        
                
                # get interleaves that does not connect to a node
                interLeave = node.interLeave[:]
                
                if len(node.otherTo) > 0:
                    node.interAnchorVerification = [7] *  len(node.otherTo) # initialized as children node
                    for i in range(0, len(node.otherTo)):
                        n = node.otherTo[i]
                        interleaf = interLeave[i]
                        if n is None:
                            #anchorLines.append(interleaf)

                            isTrueAnchor, d = self.isAnchor(image_data, interleaf, useClf= useClf)
                            disagreements += d
                            leave_count += 1

#                             PhyloParser.displayLines(image_data.image, [interleaf])
                            node.interAnchorVerification[i] = isTrueAnchor
                            if isTrueAnchor == 1: 
                                verifiedAnchorLines.append(interleaf)
                            else:
                                suspiciousAnchorLines.append(interleaf)
                        
                        else:
                            if n != node: #prevent infinite loop
                                descendants.append(n)
                            

                if len(descendants) == 0:
                    break
            
#             PhyloParser.displayATree(root)
#             for line in anchorLines:
#                 
#                 isSus = PhyloParser.isSuspicoius(line, rightVerticalLineX, line2Text, orphanBox2Text)
#                 
#                 if isSus == 0: 
#                     verifiedAnchorLines.append(line)
#                 elif isSus == -1:
#                     unsureAnchorLines.append(line)
#                 else:
#                     suspiciousAnchorLines.append(line)
                    
                    
                # 1. there is a vertical branch on it's right
                # 2. it does not have paired text
#                 if rightVerticalLineX[y] >= x_right:
#                     suspiciousAnchorLines.append(line)
#                 elif not line2Text.has_key(line):
#                     unsureAnchorLines.append(line)
#                 else:
#                     verifiedAnchorLines.append(line)
                    
            
            root.verifiedAnchorLines = verifiedAnchorLines
            root.unsureAnchorLines = unsureAnchorLines
            root.suspiciousAnchorLines = suspiciousAnchorLines
          
#         print "leave count", leave_count
#         print "disagreements", disagreements  
        image_data.rootList = rootList
        
#         print "end labelSuspiciousAnchorLine"
#         print ""
        return image_data
                
        
    @staticmethod
    # remove anchor lines that are determined false
    # remove inter-anchor line only
    # update line2text, orphanBox, node.interLeave. node.otherTo, node.interAnchorVerification, node.isInterAnchor
    def refineAnchorLine(image_data, useClf = True):  
        
        rootList = image_data.rootList

        for root in rootList:
            descendants = [root]
            
            # traverse descedants
            while True:
                node = descendants.pop()

                #### uppder leave
                # add children into descendants
                if node.to[0] is not None and node.to[0] != node: #prevent infinite loop:
                    descendants.append(node.to[0])
                    
                elif node.upperLeave is not None:
                    pass
#                     isTrueAnchor = node.biAnchorVerification[0]
#                     if isTrueAnchor == 0: # upper leave is false anchorline
#                         # find next inter-node that is not false anchorline to be the upperleave
#                         # it cound be an anchorline or a subtree
#                         if len(node.otherTo) > 0: 
#                             j = 0
#                             while j < len(node.otherTo[0]):
#                                 new_node = node.otherTo[0]
#                                 isTrueAnchor_new_node = new_node.interAnchorVerification[j]
#                                 if isTrueAnchor_new_node != 0: # this inter-node is not false leave, then rotate to be the top leave
#                                     node.to[0] = new_node
#                                     del node.interLeave[j]
#                                     del node.otherTo[j]
#                                     del node.interAnchorVerification[j]
#                                     break
#                                 
#                                 j += 1
#                     else:    
#                         pass
                    
                
                ##### lower leave
                if node.to[1] is not None and node.to[1] != node: #prevent infinite loop:
                    descendants.append(node.to[1])
                    
                elif node.lowerLeave is not None:
                    pass
                        
                # get interleaves that does not connect to a node                
                remove_index = []
                if len(node.otherTo) > 0:
                    for i in range(0, len(node.otherTo)):
                        isTrueAnchor = node.interAnchorVerification[i]
                        if isTrueAnchor == 0:
                            remove_index.append(i)
                    
                    remove_index.sort(reverse=True)
                    for index in remove_index:
                       
                        ## update line2box and orphan box 
                        x_left, y_top, x_right, y_bot, length = node.interLeave[index]
                        new_orphan_box = (y_top-3, y_top+3, x_right+1, x_right+7)

                        if image_data.line2Text.has_key(node.interLeave[index]):
                            text = image_data.line2Text[node.interLeave[index]]['text']
                            if text is not None and len(text) > 0:
                                new_orphan_text = text[0]
                                # add the textbox to orphan_box
                                image_data.orphanBox2Text[new_orphan_box] = {"text": new_orphan_text, "rotate":False}
                            
                            del image_data.line2Text[node.interLeave[index]] # remove the anchorline from line2text

                        del node.interLeave[index]
                        del node.otherTo[index]
                        del node.interAnchorVerification[index]
                        del node.isInterAnchor[index]
                        del node.interLabel[index]
                
                if len(node.otherTo) > 0:
                    
                    for i in range(0, len(node.otherTo)):
                        n = node.otherTo[i]
                        interleaf = node.interLeave[i]
                        if n is None:
                            pass
                        else:
                            if n != node: #prevent infinite loop
                                descendants.append(n)
                            

                if len(descendants) == 0:
                    break
            
        image_data.rootList = rootList
        return image_data
    
           
        
    @staticmethod
    def displayATree(image, rootNode):
        if len(image.shape) ==2:
            whatever = image.copy()
            whatever = cv.cvtColor(whatever, cv.COLOR_GRAY2RGB)
        else:
            whatever = image.copy()
        count = 0
        stack = []
        stack.append(rootNode)
        color = PhyloParser.getColor(count)
        while stack:
            node = stack.pop()
            if node.to[0]:
                stack.append(node.to[0])
            if node.to[1]:
                stack.append(node.to[1])
            if not node.isBinary:
                for to in node.otherTo:
                    if to:
                        stack.append(to)
            if node.root:
                x1, y1, x2, y2, length = node.root
                cv.rectangle(whatever , (x1, y1), (x2, y2), color=color, thickness=2)
            if node.branch:
                x1, y1, x2, y2, length = node.branch
                cv.rectangle(whatever , (x1, y1), (x2, y2), color=color, thickness=2)
            if node.upperLeave:
                x1, y1, x2, y2, length = node.upperLeave
                cv.rectangle(whatever , (x1, y1), (x2, y2), color=color, thickness=2)
            if node.lowerLeave:
                x1, y1, x2, y2, length = node.lowerLeave
                cv.rectangle(whatever , (x1, y1), (x2, y2), color=color, thickness=2)
            if not node.isBinary:
                for line in node.interLeave:
                    x1, y1, x2, y2, length = line
                    cv.rectangle(whatever , (x1, y1), (x2, y2), color=color, thickness=2)

        plt.imshow(whatever)
        plt.show()
        
    @staticmethod
    def checkAnchorLines(image_data):
        
        anchorLines = image_data.anchorLines
        branchArray = image_data.branchArray
        image = image_data.image
        rootList = image_data.rootList
        rootList = sorted(rootList, key = lambda x: x.branch[0])
        root = rootList[0]
        newAnchors = []
        wrongAnchors = []
        for line in anchorLines:
            x1, y1, x2, y2, length = line

            if not (x1+x2)/2 > max(branchArray[y1]):
                wrongAnchors.append(line)
            else:
                newAnchors.append(line)

        if len(wrongAnchors) > 0:
            stack = []
            seen = []

            for rootNode in rootList:
                stack.append(rootNode)
                while stack:

                    node = stack.pop()
                    seen.append(node)
                    for line in wrongAnchors:
                        if node.upperLeave and PhyloParser.isSameLine(node.upperLeave, line):
                            node.isUpperAnchor = False
                            connectNode = PhyloParser.getNodeBranchOnTheRight(node, rootList, mode = 'upper')
                            if connectNode:
                                rootNode.numNodes += connectNode.numNodes
                                rootNode.nodesIncluded += connectNode.nodesIncluded
                                tmpTo = list(node.to)
                                tmpTo[0] = connectNode
                                node.to = tuple(tmpTo)
                        elif node.to[0] and node.to[0] not in seen:
                            stack.append(node.to[0])

                        if node.lowerLeave and PhyloParser.isSameLine(node.lowerLeave, line):
                            node.isLowerAnchor = False
                            connectNode = PhyloParser.getNodeBranchOnTheRight(node, rootList, mode = 'lower')
                            if connectNode:
                                rootNode.numNodes += connectNode.numNodes
                                rootNode.nodesIncluded += connectNode.nodesIncluded
                                tmpTo = list(node.to)
                                tmpTo[1] = connectNode
                                node.to = tuple(tmpTo)
                        elif node.to[1] and node.to[1] not in seen:
                            stack.append(node.to[1])

                        if not node.isBinary:
                            for lineIndex, interLine in enumerate(node.interLeave):
                                if PhyloParser.isSameLine(interLine, line):
                                    node.isInterAnchor[lineIndex] = False
                                    connectNode = PhyloParser.getNodeBranchOnTheRight(node, rootList, mode = interLine)
                                    if connectNode:
                                        rootNode.numNodes += connectNode.numNodes
                                        rootNode.nodesIncluded += connectNode.nodesIncluded
                                        node.otherTo[lineIndex] = connectNode
                                elif node.otherTo[lineIndex] and node.otherTo[lineIndex] not in seen:
                                    stack.append(node.otherTo[lineIndex])

        image_data.anchorLines = newAnchors


        return image_data

    #Fix false-positive sub-trees and connect sub-trees
    def fixTrees(self,image_data):
        rootList = image_data.rootList
        parent = image_data.parent
        children = image_data.children
        horLines = image_data.horLines
        verLines = image_data.verLines
        breakNodeList = []
        tmpList = rootList[:]
        rootList = sorted(rootList, key = lambda x: -x.branch[0])
        for node in rootList:
            # image_data.displayNode(node)
            # print node.numNodes,
            # image_data.displayATree(node)
            if node in tmpList:


                if not node.isComplete:
                    for breakNode in node.breakSpot[:]:

                        isFixed = False
                        isUpper = True
                        if not ((breakNode.to[0] or breakNode.upperLeave) and (breakNode.to[1] or breakNode.lowerLeave)):
                            x1, y1, x2, y2, length = breakNode.branch
                            result = self.getNodeBranchOnTheRight(breakNode, rootList, mode = 'lower')

                            if result and len(result.nodesIncluded + node.nodesIncluded) == len(list(set(result.nodesIncluded + node.nodesIncluded))): 
                                
                                node.nodesIncluded += result.nodesIncluded
                                to = list(breakNode.to)
                                to[1] = result
                                breakNode.to = tuple(to)
                                result.whereFrom = breakNode
                                result.origin = node
                                node.numNodes +=result.numNodes
                                if result.isComplete:
                                    # print "remove"
                                    if result in tmpList:
                                        tmpList.remove(result) 
                                if PhyloParser.isNodeComplete(breakNode):                               
                                    node.breakSpot.remove(breakNode)
                                result.whereFrom = breakNode
                                
                            else:
                                isUpper = False

                            x1, y1, x2, y2, length = breakNode.branch
                            result = self.getNodeBranchOnTheRight(breakNode, rootList, mode = 'upper')

                            # # print result
                            if result and len(result.nodesIncluded + node.nodesIncluded) == len(list(set(result.nodesIncluded + node.nodesIncluded))): 
                                node.nodesIncluded += result.nodesIncluded
                                to = list(breakNode.to)
                                to[0] = result
                                breakNode.to = tuple(to)
                                if PhyloParser.isNodeComplete(breakNode):                               
                                    node.breakSpot.remove(breakNode)
                                result.origin = node
                                node.numNodes+=result.numNodes
                                if result.isComplete:
                                    # print "remove"
                                    if result in tmpList:
                                        tmpList.remove(result)
                                result.whereFrom = breakNode
                                

                        elif (breakNode.to[0] or breakNode.upperLeave) and not (breakNode.to[1] or breakNode.lowerLeave):
                            # print "lower"breakSpot
                            x1, y1, x2, y2, length = breakNode.branch
                            result = self.getNodeBranchOnTheRight(breakNode, rootList, mode = 'lower')
                            if result and len(result.nodesIncluded + node.nodesIncluded) == len(list(set(result.nodesIncluded + node.nodesIncluded))):                                 
                                node.nodesIncluded += result.nodesIncluded
                                to = list(breakNode.to)
                                to[1] = result
                                breakNode.to = tuple(to)
                                result.whereFrom = breakNode
                                result.origin = node
                                node.numNodes +=result.numNodes
                                if result.isComplete:
                                    # print "remove"
                                    if result in tmpList:
                                        tmpList.remove(result) 
                                if PhyloParser.isNodeComplete(breakNode):                               
                                    node.breakSpot.remove(breakNode)
                                result.whereFrom = breakNode
                                isFixed = True
                            else:
                                isUpper = False


                        elif not (breakNode.to[0] or breakNode.upperLeave) and (breakNode.to[1] or breakNode.lowerLeave):
                            x1, y1, x2, y2, length = breakNode.branch
                            result = self.getNodeBranchOnTheRight(breakNode, rootList, mode = 'upper')

                            # print result
                            if result and len(result.nodesIncluded + node.nodesIncluded) == len(list(set(result.nodesIncluded + node.nodesIncluded))): 
                                node.nodesIncluded += result.nodesIncluded
                                to = list(breakNode.to)
                                to[0] = result
                                breakNode.to = tuple(to)
                                if PhyloParser.isNodeComplete(breakNode):                               
                                    node.breakSpot.remove(breakNode)
                                result.origin = node
                                node.numNodes+=result.numNodes
                                if result.isComplete:
                                    # print "remove"
                                    if result in tmpList:
                                        tmpList.remove(result)
                                result.whereFrom = breakNode
                                isFixed = True
                        if isUpper:
                            breakSpot = 'upper'
                        else:
                            breakSpot = 'lower'

                else:
                    pass
            # print node.numNodes
            # image_data.displayATree(node)


        rootList = tmpList[:]
        for node in rootList:
            if node in tmpList:
                if len(node.breakSpot) == 0 and node.whereFrom != None:
                    tmpList.remove(node)
        rootList = tmpList[:]

        rootList = sorted(rootList, key = lambda x: -x.numNodes)
        if len(rootList) == 1:
            rootList[0].isComplete = True

        image_data.rootList = rootList
        image_data.parent = parent
        image_data.children = children
        return image_data


    @staticmethod 
    # find breakspot
    def getBreakSpot(image_data, debug = False):
        breakSpot = Set([])
        node_list = [image_data.rootList[0]];

        while len(node_list) > 0:
            
            node = node_list.pop();
            
            if node.to[0] is None and node.upperLeave is None:
                breakSpot.add(node)
            elif node.to[0] is not None:
                node_list.append(node.to[0])
                
                
            if node.to[1] is None and node.lowerLeave is None:
                breakSpot.add(node)
            elif node.to[1] is not None:
                node_list.append(node.to[1])

            if len(node.otherTo) > 0:
                for children in node.otherTo:
                    if children is not None:
                        node_list.append(children)
                        
        image_data.rootList[0].breakSpot = list(breakSpot)
        
        return image_data



    @staticmethod
    # for each node with breakspot, find matched orphan box
    # effective to a node with either only top leave or bot leave found or both 
    # it does not work for the multi-leave node that has both top and bot leave found. In this case, it does not do anything for the node.
    def recoverInterLeaveFromOrphanBox(image_data, debug = False):

        rootNode = image_data.rootList[0]
        breakSpot = rootNode.breakSpot

        for node in breakSpot:
#             print 
#             print "node.branch", node.branch
#             print "top leave", node.upperLeave
#             print "bot leave", node.lowerLeave

            x_left, y_top, x_right, y_bot, length = node.branch
      
            matchBoxes = PhyloParser.matchVerticalBranchWithOrphanBox(node.branch, image_data.orphanBox2Text)
#             print "matchBoxes", matchBoxes
#             print 
            if node.upperLeave is None:
                boxes = PhyloParser.getBoxCorrespond2Vertex(node.branch[0:2], matchBoxes)
                if len(boxes) > 0:           
                    # update the upper label
                    box = boxes[0]
                    name = box[1]['text']                    
                    node.upperLeave = (x_left, y_top, x_right + 5, y_top, 5)##fake anchorilne
                    node.upperLabel = PhyloParser.replaceEscapeChar(name)
                    image_data.line2Text[node.upperLeave] = {'status': 'recovered_from_orphan', 'text': [PhyloParser.replaceEscapeChar(name)]}
                    
#                     print "node.upperLeave", node.upperLeave
#                     print " node.upperLabel",  node.upperLabel
                    # remove the box from matchBoxes
                    for b in boxes:
                        matchBoxes.remove(b)
                else:
                    # mandatory assign?
                    pass
                
            if node.lowerLeave is None:
                boxes = PhyloParser.getBoxCorrespond2Vertex(node.branch[2:4], matchBoxes)
                if len(boxes) > 0:           
                    # update the upper label
                    box = boxes[0]
                    name = box[1]['text']                    
                    node.lowerLeave = (x_left, y_bot, x_right + 5, y_bot, 5)##fake anchorilne
                    node.lowerLabel = PhyloParser.replaceEscapeChar(name)
                    image_data.line2Text[node.lowerLabel] = {'status': 'recovered_from_orphan', 'text': [PhyloParser.replaceEscapeChar(name)]}
                    
#                     print "node.lowerLeave", node.lowerLeave
#                     print " node.lowerLabel",  node.lowerLabel
                    
                    # remove the box from matchBoxes
                    for b in boxes:
                        matchBoxes.remove(b)
                else:
                    # mandatory assign?
                    pass
                        
            if len(matchBoxes) > 0:
#                 print "matchBoxes for interLeaves", matchBoxes
#                 print "current interleaves", node.interLeave
                node.isBinary = False
                for mb in matchBoxes:
                    b = mb[0]
                    name = mb[1]['text']
                    new_y = (b[0] + b[1])/2
                    fakeLeave =  (x_right, new_y, x_right + 5, new_y, 5)##fake anchorilne
                    
                    # update line2text
                    image_data.line2Text[fakeLeave] = {'status': 'recovered_from_orphan', 'text': [PhyloParser.replaceEscapeChar(name)]}
                    
                    
                    # update node data
                    node.interLeave.append(fakeLeave)
                    node.isInterAnchor.append(True)
                    node.interLabel.append(None) # will be update in mergeTreeAndText
                    node.otherTo.append(None)
                    node.interAnchorVerification.append(False)
 
                                       
                    # sort from top to bottom in case there are nodes already in the list
                        
                    tmp = zip(node.interLeave, node.isInterAnchor, node.interLabel, node.otherTo, node.interAnchorVerification)
                    tmp.sort(key = lambda x: x[0][1])
                    
                    node.interLeave = map(lambda x: x[0], tmp)
                    node.isInterAnchor = map(lambda x: x[1], tmp)
                    node.interLabel = map(lambda x: x[2], tmp)
                    node.otherTo = map(lambda x: x[3], tmp)
                    node.interAnchorVerification = map(lambda x: x[4], tmp)
            if debug:
                image_data.displayNode(node)
            
        return image_data
        


    @staticmethod
    # not use
    def recoverLineFromText(image_data):
        print "recoverLineFromText"
        image_height = image_data.image_height
        image_width = image_data.image_width
        rootNode = image_data.rootList[0]
        breakSpot = rootNode.breakSpot
        nodeList = image_data.nodeList
        line2Text = image_data.line2Text
        orphanBox2Text = image_data.orphanBox2Text
        anchorLines = image_data.anchorLines
        refBreakSpot = breakSpot[:]
        
        for node in refBreakSpot:
            isMatched = False
            branch = node.branch
            branch_x1, branch_y1, branch_x2, branch_y2, branch_len = branch

            for textBox, textInfo in orphanBox2Text.items():
                textBox_y1, textBox_y2, textBox_x1, textBox_x2 = textBox
                if textBox_x1 > branch_x1 and textBox_y2 > branch_y1 and textBox_y1 < branch_y2:
                    score1 = 200/(textBox_x1 - branch_x1)
                    score2 = (textBox_x2 - textBox_x1 + 0.0 ) / image_width

                    if score1 >= 10 and score2 > 0.07:

                        if branch_y1 >= textBox_y1 and branch_y1 <= textBox_y2: ## will be handled later
                            newAnchor = (branch_x1, branch_y1, branch_x1 + 5, branch_y1, 5) ####
                            node.upperLeave = newAnchor
                            node.isUpperAnchor = True
                        elif branch_y2 >= textBox_y1 and branch_y2 <= textBox_y2: ## will be handled later
                            newAnchor = (branch_x1, branch_y2, branch_x1 + 5, branch_y2, 5) ####
                            node.lowerLeave = newAnchor
                            node.isLowerAnchor = True
                        else:
                            textBox_y = int((textBox_y1 + textBox_y2)/2)
                            newAnchor = (branch_x1, textBox_y, branch_x1 + 5, textBox_y, 5)
                            node.interLeave.append(newAnchor)
                            node.isInterAnchor.append(True)
                            node.otherTo.append(None)
                            node.interLabel.append(textInfo['text'])
                            if node.isBinary:
                                node.isBinary = False

#                         image_data.displayNode(node)
                        tmpDict = {}
                        tmpDict['status'] = 'from_box'
                        tmpDict['text'] = textInfo['text']
                        line2Text[newAnchor] = tmpDict
                        anchorLines.append(newAnchor)


            if isMatched:
                breakSpot.remove(node)

        rootNode.breakSpot = breakSpot
        image_data.rootList[0] = rootNode
        image_data.line2Text = line2Text
        image_data.orphanBox2Text = image_data.orphanBox2Text
        image_data.anchorLines = anchorLines

        print "recoverLineFromText end"
        return image_data


    # Return True if the tree is completed; otherwise return false
    def isTreeReady(self, image_data):
        rootList = image_data.rootList
        isDone = True
        rootList = sorted(rootList, key = lambda x: -x.numNodes)
        rootNode = rootList[0]

        #
        if not rootNode.isComplete:
            isDone =  False

        #
        if rootNode.root:
            x1, y1, x2, y2, length = rootNode.root
            for node in rootList:
                if node != rootNode:
                    if self.isDotWithinLine((x1, y1), node.branch) or self.isLefter(rootNode.branch, node.branch):
                        isDone = False
        #
        else:
            for node in rootList:
                if node != rootNode:
                    if self.isLefter(rootNode.branch, node.branch):
                        isDone = False

        return isDone
 

    # Not use
    def recoverMissingSmallTrees(self, node, image_data):
        branch = node.branch
        result = self.traceTree_v2_1_1(image_data, None, branch)
        if result:
            trunkList, isMulti = result
            nodeList = [node]
            if isMulti:
                origin = trunkList[0]
                # newNode = Node(node.root, node.branch, None, None)
                node = PhyloParser.matchNodeAndTrunkBasic(node, origin)

                for index, trunk in enumerate(trunkList):
                    if index!=0:
                        newNode = Node(None, None, None, None)
                        newNode = PhyloParser.matchNodeAndTrunkBasic(newNode, trunk)
                        # brokenNodes.append(newNode)
                        nodeList.append(newNode)
                refNodeList = nodeList[:]
                # print refNodeList
                for subNode in nodeList:
                    for refNode in refNodeList:
                        if subNode!=refNode:
                            if subNode.upperLeave == refNode.root:
                                tmp = list(subNode.to)
                                tmp[0] = refNode
                                subNode.to = tuple(tmp)
                            elif subNode.lowerLeave == refNode.root:
                                tmp = list(subNode.to)
                                tmp[1] = refNode
                                subNode.to = tuple(tmp)
                            if not subNode.isBinary:
                                for index, line in enumerate(subNode.interLeave):
                                    if line == refNode.root:
                                        subNode.otherTo[index] = refNode
                                        subNode.isInterAnchor[index] = False


        # for node in brokenNodes:
        #     image_data.displayNode(node)
        return 

    # used by recoverMissingSmallTrees(not use)
    @staticmethod
    def matchNodeAndTrunkBasic(node, trunk):

        ########################haven't implemented nonbinary case ##################
        if not node.upperLeave and trunk.upperLine:
            node.upperLeave = trunk.upperLine
            x1, y1, x2, y2, length = trunk.upperLine
        else:
            if trunk.upperLine:
                if not PhyloParser.isSameLine(node.upperLeave, trunk.upperLine):
                    node.upperLeave = trunk.upperLine
                    x1, y1, x2, y2, length = trunk.upperLine

        if not node.lowerLeave and trunk.lowerLine:
            node.lowerLeave = trunk.lowerLine
            x1, y1, x2, y2, length = trunk.lowerLine
        else:
            if trunk.lowerLine:
                if not PhyloParser.isSameLine(node.lowerLeave, trunk.lowerLine):
                    node.lowerLeave = trunk.lowerLine
                    x1, y1, x2, y2, length = trunk.lowerLine

        if trunk.rootLine:
            node.root = trunk.rootLine
        if len(trunk.nonBinaryLines)!=0:
            for line in trunk.nonBinaryLines:
                node.interLeave.append(line)
                node.otherTo.append(None)
                node.isInterAnchor.append(True)
                node.interLabel.append(None)
                node.isBinary = False

        if not node.branch:
            if trunk.trunkLine:
                node.branch = trunk.trunkLine
                x1, y1, x2, y2, length = trunk.trunkLine

        return node  





    @staticmethod
    def createNodesFromLineGroups(image_data, tracing = False):

        image = image_data.image
        height, width = image.shape
        nodesCoveredMask = np.zeros((height, width), dtype = np.uint8)
        # parents = image_data.parentsIndexes
        # children = image_data.childrenIndexes
        horLineMask = image_data.horLineMask
        horLineMappingDict = image_data.horLineMappingDict
        verLineMask = image_data.verLineMask
        verLineMappingDict = image_data.verLineMappingDict
        nodeIndex = 1
        nodes = {}
        notCompleteNode = []
        oldversion_nodeList = []
        brokenNodes = []
        for verLineIndex, verLineGroup in verLineMappingDict['lineMapping'].items():
            if len(verLineGroup['children'])!=0:
                lineChildren = verLineGroup['children'][:]
                isFound = False
                for parent in verLineGroup['parent']:
                    newversion_node = [parent, verLineIndex, lineChildren]
                    nodes[nodeIndex] = [parent, verLineIndex, lineChildren]
                    PhyloParser.drawNodesCoveredMask(newversion_node, nodesCoveredMask, [horLineMappingDict, verLineMappingDict])
                    oldversion_node = PhyloParser.old2newConverter(newversion_node, [horLineMappingDict, verLineMappingDict])
                    oldversion_nodeList.append(oldversion_node)
                    if PhyloParser.isNodeComplete(oldversion_node):
                        notCompleteNode.append(nodeIndex)
                        brokenNodes.append(oldversion_node)
                    nodeIndex+=1
                    isFound = True
                if not isFound:
                    newversion_node = [None, verLineIndex, lineChildren]
                    nodes[nodeIndex] = [None, verLineIndex, lineChildren]
                    PhyloParser.drawNodesCoveredMask(newversion_node, nodesCoveredMask, [horLineMappingDict, verLineMappingDict])
                    oldversion_node = PhyloParser.old2newConverter(newversion_node, [horLineMappingDict, verLineMappingDict])
                    oldversion_nodeList.append(oldversion_node)
                    notCompleteNode.append(nodeIndex)
                    nodeIndex+=1

        # if tracing:
        #     print 'Before first Tracing'
        #     # PhyloParser.displayImage(lineCoveredMask)

        #     nodesCoveredMask, brokenNodes, oldversion_nodeList = PhyloParser.findMissingLines(brokenNodes, oldversion_nodeList, image_data, mask = nodesCoveredMask)
        #     print 'After first Tracing'
        
        image_data.nodeList = oldversion_nodeList


        image_data = PhyloParser.connectNodes_old(image_data)

        image_data.nodesCoveredMask = nodesCoveredMask

        return image_data
        # for childIndex, child in children.items():
        #     c_branchIndex, c_leavesIndexes = child
        #     isFound = False    
        #     for parentIndex, parent in parents.items():
        #         p_rootInex, p_branchIndex = parent
        #         if c_branchIndex == p_branchIndex:
        #             isFound = True

    @staticmethod
    def connectNodes_old(image_data):
        nodeList = image_data.nodeList
        image = image_data.image
        height, width = image.shape
        nodeList = sorted(nodeList, key = lambda x: -x.branch[0])
        branchArray = [None] * height

        for node in nodeList:

            if not node.isConnected:
                potentialUpperLeaves = []
                potentialLowerLeaves = []
                potentialInterLeaves = []
                if not node.isBinary:
                    for line in node.interLeave:
                        potentialInterLeaves.append([])
                if not (node.isUpperAnchor and node.isLowerAnchor and node.isBinary):
                    for subNode in nodeList:
                        if subNode!=node and subNode.branch != node.branch:
                            if node.upperLeave and not node.isUpperAnchor:
                                lineEndx = node.upperLeave[2]
                                lineEndy = node.upperLeave[3]
                                if (subNode.root and PhyloParser.isSameLine(subNode.root, node.upperLeave)) or PhyloParser.isDotWithinLine((lineEndx, lineEndy), subNode.branch):
                                    score = PhyloParser.evaluateNode(subNode)
                                    if abs(subNode.branch[0] - lineEndx) == 0:
                                        distScore = 10
                                    else:
                                        distScore = (10+0.0)/abs(subNode.branch[0] - lineEndx)
                                    potentialUpperLeaves.append((subNode, score + distScore))
                            if node.lowerLeave and not node.isLowerAnchor:
                                lineEndx = node.lowerLeave[2]
                                lineEndy = node.lowerLeave[3]

                                if (subNode.root and PhyloParser.isSameLine(subNode.root, node.lowerLeave)) or PhyloParser.isDotWithinLine((lineEndx, lineEndy), subNode.branch):
                                    score = PhyloParser.evaluateNode(subNode)
                                    if abs(subNode.branch[0] - lineEndx) == 0:
                                        distScore = 10
                                    else:
                                        distScore = (10+0.0)/abs(subNode.branch[0] - lineEndx)
                                    potentialLowerLeaves.append((subNode, score+distScore))
                            if not node.isBinary:
                                for index, interLine in enumerate(node.interLeave):
                                    lineEndx = interLine[2]
                                    lineEndy = interLine[3]
                                    if (subNode.root and PhyloParser.isSameLine(interLine, subNode.root)) or PhyloParser.isDotWithinLine((lineEndx, lineEndy), subNode.branch):
                                        score = PhyloParser.evaluateNode(subNode)
                                        if abs(subNode.branch[0] - lineEndx) == 0:
                                            distScore = 10
                                        else:
                                            distScore = (10+0.0)/abs(subNode.branch[0] - lineEndx)
                                        potentialInterLeaves[index].append((subNode, score+distScore))


            potentialUpperLeaves = sorted(potentialUpperLeaves, key = lambda x: -x[1])
            potentialLowerLeaves = sorted(potentialLowerLeaves, key = lambda x: -x[1])
            
            if not node.isBinary:
                for index, interLeave in enumerate(potentialInterLeaves):
                    potentialInterLeaves[index] = sorted(interLeave, key = lambda x: -x[1])
            isConnected = False
            tmpTo = list(node.to)
            if len(potentialUpperLeaves) != 0:
                isConnected = True
                targetNode = potentialUpperLeaves[0][0]
                if not targetNode.isConnected:
                    branchArray = PhyloParser.markBranchArray(targetNode, branchArray)
                    tmpTo[0] = targetNode
                    targetNode.isConnected = True
                    targetNode.whereFrom = node
                else:
                    refNode = targetNode.whereFrom
                    if PhyloParser.betterNode(node, refNode):
                        tmpTo[0] = targetNode
                        targetNode.whereFrom = node

            if len(potentialLowerLeaves) !=0:
                targetNode = potentialLowerLeaves[0][0]
                isConnected = True
                if not targetNode.isConnected:
                    tmpTo[1] = targetNode
                    branchArray = PhyloParser.markBranchArray(targetNode, branchArray)
                    targetNode.isConnected = True
                    targetNode.whereFrom = node
                else:
                    refNode = targetNode.whereFrom
                    if PhyloParser.betterNode(node, refNode):
                        tmpTo[1] = targetNode
                        targetNode.whereFrom = node
            node.to = tuple(tmpTo)
            if not node.isBinary:               
                for index, inter in enumerate(potentialInterLeaves):
                    if len(inter)!=0:
                        # print interLeave
                        isConnected = True
                        tmpList = sorted(inter, key = lambda x: -x[1])
                        targetNode = tmpList[0][0]
                        if not targetNode.isConnected:
                            node.otherTo[index] = targetNode
                            branchArray = PhyloParser.markBranchArray(targetNode, branchArray)
                            targetNode.isConnected = True
                            targetNode.whereFrom = node
                        else:
                            refNode = targetNode.whereFrom
                            if PhyloParser.betterNode(node, refNode):
                                if index<len(node.otherTo):
                                    node.otherTo[index] = targetNode
                                    targetNode.whereFrom = node

            if isConnected:
                branchArray = PhyloParser.markBranchArray(node, branchArray)
        
        for array in branchArray:
            if array !=None:
                array.sort()

        # for node in nodeList:
        #     node.getNodeInfo()

        image_data.nodeList = nodeList
        image_data.branchArray = branchArray

        return image_data
    
    
    @staticmethod
    def isSameDot(dot1, dot2, margin = 5):
        y1, x1 = dot1
        y2, x2 = dot2

        if x2 < x1 + margin and x2 > x1 - margin and y2 < y1 + margin and y2 > y1 - margin:
            return True
        else:
            return False




    @staticmethod
    def old2newConverter(nodeInfo, mappingDict):
        root, branch, leaves = nodeInfo
        horLineMappingDict, verLineMappingDict = mappingDict
        if root !=None:
            rootLine = horLineMappingDict['lineMapping'][root]['rline']
        else:
            rootLine = None
        branchLine = verLineMappingDict['lineMapping'][branch]['rline']

        upperLeave = None
        lowerLeave = None
        isUpperAnchor = False
        isLowerAnchor = False
        interLeave = []
        isInterAnchor = []
        otherTo = []
        interLabel = []
        leaves = sorted(leaves, key = lambda x: horLineMappingDict['lineMapping'][x]['rline'][1])
        x1, y1, x2, y2, length = branchLine

        for leaf in leaves:
            lx1, ly1, lx2, ly2, llength = horLineMappingDict['lineMapping'][leaf]['rline']

            if not upperLeave and PhyloParser.isSameDot((ly1, lx1), (y1, x1)):
                upperLeave = horLineMappingDict['lineMapping'][leaf]['rline']
                if horLineMappingDict['lineMapping'][leaf]['type'] == 'anchorLine':
                    isUpperAnchor = True
            elif not lowerLeave and PhyloParser.isSameDot((ly1, lx1), (y2, x2)):
                lowerLeave = horLineMappingDict['lineMapping'][leaf]['rline']
                if horLineMappingDict['lineMapping'][leaf]['type'] == 'anchorLine':
                    isLowerAnchor = True
            else:
                interLeave.append(horLineMappingDict['lineMapping'][leaf]['rline'])
                if horLineMappingDict['lineMapping'][leaf]['type'] == 'anchorLine':
                    isInterAnchor.append(True)
                else:
                    isInterAnchor.append(False)
                otherTo.append(None)
                interLabel.append(None)

        a = Node(rootLine, branchLine, upperLeave, lowerLeave)

        a.isUpperAnchor = isUpperAnchor
        a.isLowerAnchor = isLowerAnchor

        if len(interLeave) == 0:
            a.isBinary = True
        else:
            a.isBinary = False
            a.interLeave = interLeave
            a.otherTo = otherTo
            a.isInterAnchor = isInterAnchor
            a.interLabel = interLabel

        # if len(leaves) == 2:

        #     leaves = sorted(leaves, key = lambda x: horLineMappingDict['lineMapping'][x]['rline'][1])

        #     upperLeave = horLineMappingDict['lineMapping'][leaves[0]]['rline']
        #     lowerLeave = horLineMappingDict['lineMapping'][leaves[1]]['rline']
        #     a = Node(rootLine, branchLine, upperLeave, lowerLeave)
        #     a.isBinary = True
        #     if horLineMappingDict['lineMapping'][leaves[0]]['type'] == 'anchorLine':
        #         a.isUpperAnchor = True
        #     if horLineMappingDict['lineMapping'][leaves[1]]['type'] == 'anchorLine':
        #         a.isLowerAnchor = True
        # elif len(leaves) >2:
        #     leaves = sorted(leaves, key = lambda x: horLineMappingDict['lineMapping'][x]['rline'][1])
        #     upperLeave = horLineMappingDict['lineMapping'][leaves[0]]['rline']
        #     lowerLeave = horLineMappingDict['lineMapping'][leaves[-1]]['rline']   
        #     a = Node(rootLine, branchLine, upperLeave, lowerLeave)

        #     if horLineMappingDict['lineMapping'][leaves[0]]['type'] == 'anchorLine':
        #         a.isUpperAnchor = True
        #     if horLineMappingDict['lineMapping'][leaves[-1]]['type'] == 'anchorLine':
        #         a.isLowerAnchor = True
        #     for leaveIndex in range(1, len(leaves)-1):
        #         interLeaveLine = horLineMappingDict['lineMapping'][leaves[leaveIndex]]['rline']
        #         a.interLeave.append(interLeaveLine)
        #         a.otherTo.append(None)
        #         a.isInterAnchor.append(horLineMappingDict['lineMapping'][leaves[leaveIndex]]['type'])
        #         a.interLabel.append(None)
        # elif len(leaves) == 1:
        #     branchMidPt = ((branchLine[1] + branchLine[3])/2, branchLine[0])
        #     leaveLine = horLineMappingDict['lineMapping'][leaves[0]]['rline']
        #     if leaveLine[1] > branchMidPt[0]:
        #         a = Node(rootLine, branchLine, None, leaveLine)
        #         if horLineMappingDict['lineMapping'][leaves[0]]['type'] == 'anchorLine':
        #             a.isUpperAnchor = True

        #     else:
        #         a = Node(rootLine, branchLine, leaveLine, None)
        #         if horLineMappingDict['lineMapping'][leaves[0]]['rline'] == 'anchorLine':
        #             a.isLowerAnchor = True
        return a



    @staticmethod
    def markBranchArray(node, branchArray):
        x1, y1, x2, y2, length = node.branch

        for index, array in enumerate(branchArray[y1:y2+1]):
            if array == None:
                branchArray[y1+index] = []
            if x1 not in branchArray[y1+index]:
                branchArray[y1+index].append(x1)


        return branchArray


 
    def findMissingLines(self, brokenNodes, nodeList, image_data, mask = None):
        refinedLines = []

        if mask == None:
            image = image_data.image
            image_height, image_width = image.shape
            mask = np.zeros((image_height,image_width), dtype=np.uint8)
            for node in nodeList:
                tmpLines = [node.root, node.upperLeave, node.lowerLeave, node.branch]
                if not node.isBinary:
                    for line in node.interLeave:
                        tmpLines.append(line)
                PhyloParser.doLineMask(mask, tmpLines)


        for node in brokenNodes:

            branch = node.branch
            result = self.traceTree_v2_1_1(image_data, mask, branch)

            # print result
            # image_data.displayNode(node)
            if result:
                trunkList, isMulti = result
                # for trunk in trunkList:
                #     PhyloParser.displayTrunk(image_data.image, trunk)
                if not isMulti:
                    trunk = trunkList[0]

                    if not (len(trunk.leaves) == 0 and len(trunk.interLines) == 0):
                        # PhyloParser.displayTrunk(image_data.image, trunk)
                        node, mask, refinedLines = PhyloParser.matchNodeAndTrunk(node, trunk, mask, refinedLines)
                        # trunk.getTrunkInfo()
                        # node.getNodeInfo()
                        # image_data.displayNode(node)

                else:
                    trunk = trunkList[0]
                    if not (len(trunk.leaves) == 0 and len(trunk.interLines) == 0):
                        node, mask, refinedLines = PhyloParser.matchNodeAndTrunk(node, trunk, mask, refinedLines)
                    for index, trunk in enumerate(trunkList):
                        if index!=0 and len(trunk.leaves) == 0 and not len(trunk.interLines) == 0:
                            # PhyloParser.displayTrunk(image_data.image, trunk)
                            newNode = Node(None, None, None, None)
                            newNode, mask, refinedLines = PhyloParser.matchNodeAndTrunk(newNode, trunk, mask, refinedLines, mode = 'new')
                            # brokenNodes.append(newNode)
                            # trunk.getTrunkInfo()
                            # newNode.getNodeInfo()
                            # image_data.displayNode(newNode)
                            nodeList.append(newNode)
        # margin = 2
        # for node in brokenNodes:
        #     if not node.root:
        #         for line in refinedLines:
        #             x1, y1, x2, y2, length = line                    
        #             bx1, by1, bx2, by2, blength = node.branch
        #             if y1-y2 == 0:
        #                 if x2 > bx1 - margin and x2 < bx2 + margin and y2 > by1 - margin and y2 < by2 + margin:
        #                     node.root = line                            
        #                     break

        # for node in brokenNodes:
        #     image_data.displayNode(node)

        return mask, brokenNodes, nodeList


    @staticmethod
    def matchNodeAndTrunk(node, trunk, mask, refinedLines, mode = 'notNew'):

        ####################### haven't implemented non binary case ########################

        if not node.upperLeave and trunk.upperLine:
            node.upperLeave = trunk.upperLine
            x1, y1, x2, y2, length = trunk.upperLine
            mask[y1, x1:x2] = 255
            refinedLines.append(trunk.upperLine)
        else:
            if trunk.upperLine:
                if not PhyloParser.isSameLine(node.upperLeave, trunk.upperLine):
                    node.upperLeave = trunk.upperLine
                    x1, y1, x2, y2, length = trunk.upperLine
                    mask[y1, x1:x2] = 255
        if not node.lowerLeave and trunk.lowerLine:
            node.lowerLeave = trunk.lowerLine
            x1, y1, x2, y2, length = trunk.lowerLine
            mask[y1, x1:x2] = 255
            refinedLines.append(trunk.lowerLine)
        else:
            if trunk.lowerLine:
                if not PhyloParser.isSameLine(node.lowerLeave, trunk.lowerLine):
                    node.lowerLeave = trunk.lowerLine
                    x1, y1, x2, y2, length = trunk.lowerLine
                    mask[y1, x1:x2] = 255
        # if not node.isBinary:

        if len(trunk.nonBinaryLines)!=0:
            for line in trunk.nonBinaryLines:
                node.interLeave.append(line)
                node.otherTo.append(None)
                node.isInterAnchor.append(None)
                node.interLabel.append(None)
                node.isBinary = False

        if not node.branch:
            if trunk.trunkLine:
                node.branch = trunk.trunkLine

                x1, y1, x2, y2, length = trunk.trunkLine

                mask[y1:y2, x1] = 255

                refinedLines.append(trunk.trunkLine)

        if trunk.rootLine:
            node.root = trunk.rootLine



        return node, mask, refinedLines




    @staticmethod
    def doLineMask(mask, lineList):
        for line in lineList:
            if line:
                x1, y1, x2, y2, length = line
                isHor = False
                if x2 - x1 != 0:
                    isHor = True
                if isHor:
                    mask[y1,x1:x2] = 255
                else:
                    mask[y1:y2, x1] = 255
        return mask





    @staticmethod
    def betterNode(node1, node2):
        score1 = 0
        score2 = 0
        if node1.score:
            score1 = node1.score
        else:
            score1 = PhyloParser.evaluateNode(node1)
        if node2.score:
            score2 = node2.score
        else:
            score1 = PhyloParser.evaluateNode(node2)
        score1 += math.sqrt(int(PhyloParser.countNodeArea(node1)))
        score2 += math.sqrt(int(PhyloParser.countNodeArea(node2)))

        if score1> score2:
            return True
        else:
            return False


    @staticmethod
    def countNodeArea(node):
        if node.branch:
            if node.lowerLeave:
                return abs((node.lowerLeave[2] - node.branch[0]) * (node.branch[1] - node.lowerLeave[1]))
            if node.upperLeave:
                return abs((node.branch[0] - node.upperLeave[2]) * (node.branch[4]))
            return 0
        else:
            return 0

    @staticmethod
    def evaluateNode(node):
        score = 0
        if node.upperLeave:
            score +=1
            if node.to[0]:
                score +=1
                if node.to[0].score:
                    score += node.to[0].score
            elif node.isUpperAnchor:
                score +=1
        if node.lowerLeave:
            score +=1
            if node.to[1]:
                if node.to[1].score:
                    score+= node.to[1].score
            elif node.isLowerAnchor:
                score +=1
        if not node.isBinary:
            for index, line in enumerate(node.interLeave):
                if line:
                    score+=1
                    if node.otherTo[index]:
                        score+=1
                        if node.otherTo[index].score:
                            score+=node.otherTo[index].score
                    elif node.isInterAnchor[index]:
                        score +=1
        node.score = score
        return score
    
    
    


### Tracing methods
###################################################################################################################################################################################################
    
    @staticmethod
    def isLineAndNodeConnected(line, node):
        x1, y1, x2, y2, length = line
        bx1, by1, bx2, by2, blength = node.branch

        if (node.root and PhyloParser.isSameLine(line, node.root)) or PhyloParser.isDotWithinLine((x2, y2), node.branch) or (y1>by1 and y1<by2 and bx1>x1):
            # print line, node.branch
            # print node.root, (node.root and PhyloParser.isSameLine(line, node.root))
            # print (x2, y2), node.branch, PhyloParser.isDotWithinLine((x2, y2), node.branch)
            # print (y1>by1 and y1<by2 and bx1>x1)
            return True
        else:
            return False

    @staticmethod
    def connectNodesInAList(nodeList):

        refList = nodeList[:]
        isConnected = []
        nodeList = sorted(nodeList, key = lambda x: x.branch[0])

        for nodeIndex, node in enumerate(nodeList):

            isConnected.append(node)
            for refNodeIndex, refNode in enumerate(refList):
                if refNodeIndex!=nodeIndex:
                    if node.upperLeave:
                        if PhyloParser.isLineAndNodeConnected(node.upperLeave, refNode) and refNode not in isConnected:
                            if not node.to[0]:

                                tmp = list(node.to)
                                tmp[0] = refNode
                                node.to = tuple(tmp)
                                node.numNodes += refNode.numNodes
                                refNode.whereFrom = node
                                refNode.origin = node.origin
                                isConnected.append(refNode)
                    if node.lowerLeave:
                        if PhyloParser.isLineAndNodeConnected(node.lowerLeave, refNode) and refNode not in isConnected:
                            if not node.to[1]:

                                tmp = list(node.to)
                                tmp[1] = refNode
                                node.to = tuple(tmp)
                                node.numNodes += refNode.numNodes
                                refNode.whereFrom = node
                                refNode.origin = node.origin
                                isConnected.append(refNode)
                    if not node.isBinary:
                        for index, interLine in enumerate(node.interLeave):
                            if PhyloParser.isLineAndNodeConnected(interLine, refNode)and refNode not in isConnected:
                                if not node.otherTo[index]:
                                    node.otherTo[index] = refNode
                                    refNode.whereFrom = node
                                    refNode.origin = node.origin
                                    node.numNodes += refNode.numNodes
                                    isConnected.append(refNode)
                                        
    
    ### Use tracing technique to construct a tree
    def constructTreeByTracing(self, image_data, debug = False, tracing = True):
        
        if image_data.lineDetected and image_data.lineMatched:
            
            # Pair h-v branch (parent) and v-h branch (children)
            image_data = PhyloParser.createNodesFromLineGroups(image_data)
            if debug:
                print "Display Nodes"
                image_data.displayNodes()

            # Create Tree
            image_data = self.createRootList(image_data)
            if debug:
                print "display Tree"
                image_data.displayTrees('regular')
                
            ## found no node
            if len(image_data.rootList) == 0:
                image_data.treeHead = None
                image_data.treeStructure = ""
                return image_data
            
# ------------------------------------------------------------------------ #

            image_data = PhyloParser.initializeImageDataForTracing(image_data)


            if tracing:
                print "tracing"
                image_data = PhyloParser.makeTreeByTracing(image_data)            


# ------------------------------------------------------------------------ #

            ## verified leaves
            image_data.rightVerticalLineX = PhyloParser.getRightVerticalLineX(image_data.image, image_data.rootList)
            image_data.avg_anchorLine_x =  PhyloParser.getAvgRightEndPointOfLines(image_data.anchorLines)
            image_data = self.labelSuspiciousAnchorLine(image_data, useClf=True)
            
            if debug:
                for i, node in enumerate(image_data.rootList):
                    print i, node.branch
                    print "verifiedAnchorLines", len(node.verifiedAnchorLines)
    #                 PhyloParser.displayLines(image_data.image, node.verifiedAnchorLines)
                    print "suspiciousAnchorLines", len(node.suspiciousAnchorLines)
    #                 PhyloParser.displayLines(image_data.image, node.suspiciousAnchorLines)
                    print "unsureAnchorLines", len(node.unsureAnchorLines)
    #                 PhyloParser.displayLines(image_data.image, node.unsureAnchorLines)
                    print ""

#             print "refineAnchorLine"
            image_data = PhyloParser.refineAnchorLine(image_data)


# ------------------------------------------------------------------------ #

            ###########################################
            ### Recover missing components ############
            
            ## directly connect right sub-trees of broken point
            if not self.isTreeReady(image_data):#######
#                 ## Fix false-positive sub-trees and mandatorily connect sub-trees
                image_data = self.fixTrees(image_data)
# 
# #                 print image_data.rootList
#                 if debug:
#                     print "display Fixed Tree"
#                     image_data.displayTrees('regular')
            
            ## use orphane box to recover line
            # sort again to ensure the first root is the largest
            image_data.rootList = sorted(image_data.rootList, key = lambda x: -x.numNodes)
            if len(image_data.rootList[0].breakSpot) > 0 and image_data.speciesNameReady:
                print "recoverInterLeaveFromOrphanBox"    
                image_data = self.recoverInterLeaveFromOrphanBox(image_data) ## not test yet
                if debug:
                    print "recoverInterLeaveFromOrphanBox result"
                    image_data.displayTrees('regular')

# ------------------------------------------------------------------------ #

            # select largest sub-tree as the final tree
            image_data.defineTreeHead()

            # merge tree structure and species text
            useText = False
            if image_data.speciesNameReady:
                print "mergeTreeAndText"
                self.mergeTreeAndText(image_data)
                print "end mergeTreeAndText"
                useText = True
                
            image_data.treeStructure = self.getTreeString(image_data, useText=useText)
            if debug:
                print image_data.treeStructure
                image_data.displayTrees('final')                
        else:
            print "Error! Tree components are not found. Please do detectLine and matchLine before this method"
        
        return image_data


    @staticmethod
    def initializeImageDataForTracing(image_data):

        rootList = image_data.rootList[:]

        rootList = sorted(rootList, key = lambda x: x.branch[0])
        newRoot = Node(branch = rootList[0].branch)
        newRoot.breakSpot.append(newRoot)
        image_data.startNodeForTracing = newRoot
        # image_data.displayNode(newRoot)
        image_data.nodeList = []
        image_data.rootList = [newRoot]


        return image_data



    def findMissingNodesByTracing(self, brokenNode, image_data, mask = None):

        if mask == None:
            image = image_data.image
            image_height, image_width = image.shape
            mask = np.zeros((image_height,image_width), dtype=np.uint8)
            # for node in nodeList:
            #     tmpLines = [node.root, node.upperLeave, node.lowerLeave, node.branch]
            #     if not node.isBinary:
            #         for line in node.interLeave:
            #             tmpLines.append(line)
            #     PhyloParser.doLineMask(mask, tmpLines)


        refinedLines = []

        branch = brokenNode.branch
        result = self.traceTree_v2_1_1(image_data, mask, branch)
        newNodeList = []
            # print result

        if result:
            trunkList, isMulti = result
            # for trunk in trunkList:
            #     PhyloParser.displayTrunk(image_data.image, trunk)

            if not isMulti:
                trunk = trunkList[0]

                if not (len(trunk.leaves) == 0 and len(trunk.interLines) == 0):
                    # PhyloParser.displayTrunk(image_data.image, trunk)
                    brokenNode, mask, refinedLines = PhyloParser.matchNodeAndTrunk(brokenNode, trunk, mask, refinedLines)
                    # trunk.getTrunkInfo()
                    # node.getNodeInfo()

                    newNodeList.append(brokenNode)
            else:
                trunk = trunkList[0]

                brokenNode, mask, refinedLines = PhyloParser.matchNodeAndTrunk(brokenNode, trunk, mask, refinedLines, mode = 'new')
                newNodeList.append(brokenNode)

                for index, trunk in enumerate(trunkList):

                    # print len(trunk.leaves), len(trunk.interLines)
                    # PhyloParser.displayTrunk(image_data.image, trunk)
                    if index!=0 and (len(trunk.leaves) != 0 or len(trunk.interLines) != 0):
                        # PhyloParser.displayTrunk(image_data.image, trunk)

                        newNode = Node(None, None, None, None)

                        newNode, mask, refinedLines = PhyloParser.matchNodeAndTrunk(newNode, trunk, mask, refinedLines, mode = 'new')
                        newNodeList.append(newNode)
                        # brokenNodes.append(newNode)
                        # trunk.getTrunkInfo()
                        # newNode.getNodeInfo()
                        # image_data.displayNode(newNode)
                        # nodeList.append(newNode)

                PhyloParser.connectNodesInAList(newNodeList)


        # margin = 2
        # for node in brokenNodes:
        #     if not node.root:
        #         for line in refinedLines:
        #             x1, y1, x2, y2, length = line                    
        #             bx1, by1, bx2, by2, blength = node.branch
        #             if y1-y2 == 0:
        #                 if x2 > bx1 - margin and x2 < bx2 + margin and y2 > by1 - margin and y2 < by2 + margin:
        #                     node.root = line                            
        #                     break

        # for node in brokenNodes:
        #     image_data.displayNode(node)

        return mask, newNodeList        

    # connect broken point using tracing algorithm
    def connectRootByTracing(self, image_data, debug = False):

        rootList = image_data.rootList

        rootList = sorted(rootList, key = lambda x: -x.branch[0])


        connectedRootNodes = []


        for topRootNode in rootList:
            # print rootList

            # for root in rootList:
            #     image_data.displayATree(root)
            for brokenNode in topRootNode.breakSpot:
                # print 'firstRoot'
                # print topRootNode.breakSpot
                # image_data.nodeList = topRootNode.breakSpot
                # image_data.displayNodes()
                # print 'brokenNode', brokenNode
                # image_data.displayNode(brokenNode)
                mask, newNodeList = self.findMissingNodesByTracing(brokenNode, image_data, mask = image_data.nodesCoveredMask)
                # mask, newNodeList = PhyloParser.findMissingNodesByTracing(brokenNode, image_data)
                # image_data.nodeList = newNodeList
                # image_data.displayNodes()
                # for node in newNodeList:
                #     node.getNodeInfo()
                
                # print 'recoveredNodes'
                # for node in newNodeList:
                #     print node
                #     image_data.displayNode(node)
                if len(newNodeList)>1:
                    image_data.nodeList += newNodeList[1:]



                stack = []
                if len(newNodeList)!=0:

                    stack.append(newNodeList[0])
                    seen = []
                    seen.append(newNodeList[0])
                    while stack:
                        node = stack.pop()
                        if node.to[0]:
                            stack.append(node.to[0])
                        else:
                            if node.upperLeave:
                                for potNode in rootList:
                                    if topRootNode != potNode and potNode not in connectedRootNodes:
                                        if PhyloParser.isLineAndNodeConnected(node.upperLeave, potNode):
                                            if len(topRootNode.nodesIncluded + potNode.nodesIncluded) == len(list(set(topRootNode.nodesIncluded + potNode.nodesIncluded))):
                                                topRootNode.nodesIncluded += potNode.nodesIncluded
                                                connectedRootNodes.append(potNode)
                                                tmp = list(node.to)
                                                tmp[0] = potNode
                                                node.to = tuple(tmp)
                                                topRootNode.numNodes += potNode.numNodes
                                                potNode.whereFrom = node
                                                potNode.origin = node.origin

                        if node.to[1]:
                            stack.append(node.to[1])
                        else:
                            if node.lowerLeave:
                                for potNode in rootList:
                                    if topRootNode != potNode and potNode not in connectedRootNodes:
                                        if PhyloParser.isLineAndNodeConnected(node.lowerLeave, potNode):
                                            if len(topRootNode.nodesIncluded + potNode.nodesIncluded) == len(list(set(topRootNode.nodesIncluded + potNode.nodesIncluded))):
                                                topRootNode.nodesIncluded += potNode.nodesIncluded
                                                connectedRootNodes.append(potNode)
                                                tmp = list(node.to)
                                                tmp[1] = potNode
                                                node.to = tuple(tmp)
                                                topRootNode.numNodes += potNode.numNodes
                                                potNode.whereFrom = node
                                                potNode.origin = node.origin
                        if not node.isBinary:
                            for index, toNode in enumerate(node.otherTo):
                                if toNode:
                                    stack.append(toNode)
                                else:
                                    if node.interLeave[index]:
                                        for potNode in rootList:
                                            if topRootNode != potNode and potNode not in connectedRootNodes:
                                                if PhyloParser.isLineAndNodeConnected(node.interLeave[index], potNode):
                                                    if len(topRootNode.nodesIncluded + potNode.nodesIncluded) == len(list(set(topRootNode.nodesIncluded + potNode.nodesIncluded))):
                                                        topRootNode.nodesIncluded += potNode.nodesIncluded
                                                        connectedRootNodes.append(potNode)
                                                        node.otherTo[index] = potNode
                                                        topRootNode.numNodes += potNode.numNodes
                                                        potNode.whereFrom = node
                                                        potNode.origin = node.origin
        # #         seen = []
        #         refNewNodeList = newNodeList[:]
        #         for breakSpotNode in topRootNode.breakSpot:
        #             for newNode in refNewNodeList:
        #                 if PhyloParser.isSameLine(breakSpotNode.branch, newNode.branch):
        #                     if newNode in newNodeList:
        #                         newNodeList.remove(newNode)
        #         # print 'breakNodes'
        #         # image_data.nodeList = breakNodes
        #         # image_data.displayNodes()
        #         # print 'newNodes'
        #         # image_data.nodeList = newNodeList
        #         # image_data.displayNodes()
        #         for node in newNodeList:
        #             if not node.to[0] and not node.isUpperAnchor:
        #                 if node.upperLeave:
        #                     for newNode in newNodeList:
        #                         if newNode!=node and newNode.root and PhyloParser.isSameLine(node.upperLeave, newNode.root) and newNode not in seen:
        #                             if not node.to[0] or newNode.numNodes >=node.to[0].numNodes:
        #                                 tmp = list(node.to)
        #                                 tmp[0] = newNode
        #                                 node.to = tuple(tmp)
        #                                 newNode.whereFrom = node
        #                                 node.numNodes+=newNode.numNodes
        #                                 if node not in node.nodesIncluded:
        #                                     node.nodesIncluded.append(node)
        #                                 node.nodesIncluded.append(newNode)
        #                                 newNode.origin = node


        #                     foundRoot =None
        #                     for rootNode in rootList:
        #                         if rootNode!=topRootNode and rootNode not in connectedRootNodes:
        #                             rootBranch = rootNode.branch
        #                             endPt = (node.upperLeave[2], node.upperLeave[3])
        #                             if PhyloParser.isDotWithinLine(endPt, rootBranch):
        #                                 if not node.to[0] or rootNode.numNodes >=node.to[0].numNodes:
        #                                     tmp = list(node.to)
        #                                     tmp[0] = rootNode
        #                                     node.to = tuple(tmp)
        #                                     rootNode.whereFrom = node
        #                                     foundRoot = rootNode
        #                                     node.origin = node
        #                                     node.nodesIncluded.append(node)
        #                                     node.origin.numNodes += rootNode.numNodes
        #                                     node.origin.nodesIncluded += rootNode.nodesIncluded
        #                                     connectedRootNodes.append(rootNode)
        #                                     break

        #             if not node.to[1] and not node.isLowerAnchor:
        #                 if node.lowerLeave:
        #                     for newNode in newNodeList:
        #                         if newNode!=node and newNode.root and PhyloParser.isSameLine(node.lowerLeave, newNode.root):
        #                             if not node.to[1] or newNode.numNodes >=node.to[1].numNodes:
        #                                 tmp = list(node.to)
        #                                 tmp[1] = newNode  
        #                                 node.to = tuple(tmp)
        #                                 newNode.whereFrom = node
        #                                 node.numNodes+=newNode.numNodes
        #                                 if node not in node.nodesIncluded:
        #                                     node.nodesIncluded.append(node)
        #                                 node.nodesIncluded.append(newNode)
        #                                 newNode.origin = node
        #                     foundRoot =None
        #                     for rootNode in rootList:
        #                         if rootNode!=topRootNode and rootNode not in connectedRootNodes:
        #                             rootBranch = rootNode.branch
        #                             endPt = (node.lowerLeave[2], node.lowerLeave[3])
        #                             if PhyloParser.isDotWithinLine(endPt, rootBranch):
        #                                 # image_data.displayNode(rootNode)
        #                                 # image_data.displayNode(node)
        #                                 if not node.to[1] or rootNode.numNodes >=node.to[1].numNodes:
        #                                     tmp = list(node.to)
        #                                     tmp[1] = rootNode
        #                                     node.to = tuple(tmp)
        #                                     rootNode.whereFrom = node
        #                                     foundRoot = rootNode
        #                                     node.origin = node
        #                                     node.nodesIncluded.append(node)
        #                                     node.origin.numNodes += rootNode.numNodes
        #                                     node.origin.nodesIncluded += rootNode.nodesIncluded
        #                                     connectedRootNodes.append(rootNode)
        #                                     break


        #             if not node.isBinary:
        #                 for index, line in enumerate(node.interLeave):
        #                     if not node.otherTo[index] and not node.isInterAnchor[index]:
        #                         if line:
        #                             for newNode in newNodeList:
        #                                 if newNode!=node and newNode.root and PhyloParser.isSameLine(line, newNode.root):
        #                                     if not node.otherTo[index] or newNode.numNodes >=node.otherTo[index].numNodes:
        #                                         node.otherTo[index] = newNode
        #                                         newNode.whereFrom = node
        #                                         node.numNodes+=newNode.numNodes
        #                                         if node not in node.nodesIncluded:
        #                                             node.nodesIncluded.append(node)
        #                                         node.nodesIncluded.append(newNode)
        #                                         newNode.origin = node
        #                             foundRoot = None
        #                             for rootNode in rootList:
        #                                 if rootNode!=topRootNode and rootNode not in connectedRootNodes:
        #                                     rootBranch = rootNode.branch
        #                                     endPt = (line[2], line[3])
        #                                     if PhyloParser.isDotWithinLine(endPt, rootBranch):
        #                                         if not node.otherTo[index] or newNode.numNodes >=node.otherTo[index].numNodes:
        #                                             node.otherTo[index] = rootNode
        #                                             rootNode.whereFrom = node
        #                                             foundRoot = rootNode
        #                                             node.origin = node
        #                                             node.nodesIncluded.append(node)
        #                                             node.origin.numNodes += rootNode.numNodes
        #                                             node.origin.nodesIncluded += rootNode.nodesIncluded
        #                                             connectedRootNodes.append(rootNode)
        #                                             break


        #         for node in breakNodes:

        #             if not node.to[0] and not node.isUpperAnchor:
        #                 if node.upperLeave:
        #                     for newNode in newNodeList:
        #                         if newNode.root and PhyloParser.isSameLine(node.upperLeave, newNode.root):
        #                             if not node.to[0] or newNode.numNodes >=node.to[0].numNodes:
        #                                 tmp = list(node.to)
        #                                 tmp[0] = newNode
        #                                 node.to = tuple(tmp)
        #                                 newNode.whereFrom = node
        #                     foundRoot =None
        #                     for rootNode in rootList:
        #                         if rootNode!=topRootNode and rootNode not in connectedRootNodes:
        #                             rootBranch = rootNode.branch
        #                             endPt = (node.upperLeave[2], node.upperLeave[3])

        #                             if PhyloParser.isDotWithinLine(endPt, rootBranch):

        #                                 if not node.to[0] or rootNode.numNodes >=node.to[0].numNodes:
        #                                     tmp = list(node.to)
        #                                     tmp[0] = rootNode
        #                                     node.to = tuple(tmp)
        #                                     rootNode.whereFrom = node
        #                                     foundRoot = rootNode
        #                                     node.origin.numNodes += rootNode.numNodes
        #                                     node.origin.nodesIncluded += rootNode.nodesIncluded
        #                                     connectedRootNodes.append(rootNode)
        #                                     break

        #             if not node.to[1] and not node.isLowerAnchor:

        #                 if node.lowerLeave:
        #                     for newNode in newNodeList:
        #                         if newNode.root and PhyloParser.isSameLine(node.lowerLeave, newNode.root):
        #                             if not node.to[1] or newNode.numNodes >=node.to[1].numNodes:

        #                                 tmp = list(node.to)
        #                                 tmp[1] = newNode
        #                                 node.to = tuple(tmp)
        #                                 newNode.whereFrom = node
        #                     foundRoot =None
        #                     for rootNode in rootList:
        #                         rootBranch = rootNode.branch
        #                         # print rootNode, topRootNode
        #                         # print rootNode, connectedRootNodes
        #                         if rootNode!=topRootNode and rootNode not in connectedRootNodes:
        #                             rootBranch = rootNode.branch
        #                             endPt = (node.lowerLeave[2], node.lowerLeave[3])
        #                             # print endPt, rootBranch
        #                             if PhyloParser.isDotWithinLine(endPt, rootBranch):
        #                                 if not node.to[1] or rootNode.numNodes>=node.to[1].numNodes:
        #                                     # print 'connect to root node'
        #                                     # image_data.displayNode(rootNode)
                                            
        #                                     tmp = list(node.to)
        #                                     tmp[1] = rootNode
        #                                     node.to = tuple(tmp)
        #                                     rootNode.whereFrom = node
        #                                     foundRoot = rootNode
        #                                     node.origin.numNodes += rootNode.numNodes
        #                                     node.origin.nodesIncluded += rootNode.nodesIncluded
        #                                     connectedRootNodes.append(rootNode)

        #                                     break

        #             if not node.isBinary:
        #                 for index, line in enumerate(node.interLeave):
        #                     if not node.otherTo[index] and not node.isInterAnchor[index]:
        #                         if line:
        #                             for newNode in newNodeList:
        #                                 if newNode.root and  PhyloParser.isSameLine(line, newNode.root):
        #                                     if not node.otherTo[index] or newNode.numNodes >node.otherTo[index].numNodes:
        #                                         node.otherTo[index] = newNode
        #                                         newNode.whereFrom = node
        #                             foundRoot = None
        #                             for rootNode in rootList:
        #                                 if rootNode!=topRootNode and rootNode not in connectedRootNodes:
        #                                     rootBranch = rootNode.branch
        #                                     endPt = (line[2], line[3])
        #                                     if PhyloParser.isDotWithinLine(endPt, rootBranch):
        #                                         if not node.otherTo[index] or rootNode.numNodes >=node.otherTo[index].numNodes:   
        #                                             node.otherTo[index] = rootNode
        #                                             rootNode.whereFrom = node
        #                                             foundRoot = rootNode
        #                                             node.origin.numNodes += rootNode.numNodes
        #                                             node.origin.nodesIncluded += rootNode.nodesIncluded
        #                                             connectedRootNodes.append(rootNode)
        #                                             break


        # for node in connectedRootNodes:
        #     if node in rootList:
        #         rootList.remove(node)

        rootList = sorted(rootList, key = lambda x: -x.numNodes)

        image_data.rootList = rootList

        return image_data

    @staticmethod
    def makeTreeByTracing(image_data, debug = False):

        rootList = image_data.rootList

        rootList = sorted(rootList, key = lambda x: -x.branch[0])


        connectedRootNodes = []


        for topRootNode in rootList:
            # print rootList

            # for root in rootList:
            #     image_data.displayATree(root)
            for brokenNode in topRootNode.breakSpot:
                # print 'firstRoot'
                # print topRootNode.breakSpot
                # image_data.nodeList = topRootNode.breakSpot
                # image_data.displayNodes()
                # print 'brokenNode', brokenNode
                # image_data.displayNode(brokenNode)
                mask, newNodeList = PhyloParser.findMissingNodesByTracing(brokenNode, image_data, mask = None)
                # mask, newNodeList = PhyloParser.findMissingNodesByTracing(brokenNode, image_data)
                # image_data.nodeList = newNodeList
                # image_data.displayNodes()
                # for node in newNodeList:
                #     node.getNodeInfo()
                
                # print 'recoveredNodes'
                # for node in newNodeList:
                #     print node
                #     image_data.displayNode(node)
                if len(newNodeList)>1:
                    image_data.nodeList += newNodeList[1:]



                stack = []
                if len(newNodeList)!=0:

                    stack.append(newNodeList[0])
                    seen = []
                    seen.append(newNodeList[0])
                    while stack:
                        node = stack.pop()
                        if node.to[0]:
                            stack.append(node.to[0])
                        else:
                            if node.upperLeave:
                                for potNode in rootList:
                                    if topRootNode != potNode and potNode not in connectedRootNodes:
                                        if PhyloParser.isLineAndNodeConnected(node.upperLeave, potNode):
                                            if len(topRootNode.nodesIncluded + potNode.nodesIncluded) == len(list(set(topRootNode.nodesIncluded + potNode.nodesIncluded))):
                                                topRootNode.nodesIncluded += potNode.nodesIncluded
                                                connectedRootNodes.append(potNode)
                                                tmp = list(node.to)
                                                tmp[0] = potNode
                                                node.to = tuple(tmp)
                                                topRootNode.numNodes += potNode.numNodes
                                                potNode.whereFrom = node
                                                potNode.origin = node.origin

                        if node.to[1]:
                            stack.append(node.to[1])
                        else:
                            if node.lowerLeave:
                                for potNode in rootList:
                                    if topRootNode != potNode and potNode not in connectedRootNodes:
                                        if PhyloParser.isLineAndNodeConnected(node.lowerLeave, potNode):
                                            if len(topRootNode.nodesIncluded + potNode.nodesIncluded) == len(list(set(topRootNode.nodesIncluded + potNode.nodesIncluded))):
                                                topRootNode.nodesIncluded += potNode.nodesIncluded
                                                connectedRootNodes.append(potNode)
                                                tmp = list(node.to)
                                                tmp[1] = potNode
                                                node.to = tuple(tmp)
                                                topRootNode.numNodes += potNode.numNodes
                                                potNode.whereFrom = node
                                                potNode.origin = node.origin
                        if not node.isBinary:
                            for index, toNode in enumerate(node.otherTo):
                                if toNode:
                                    stack.append(toNode)
                                else:
                                    if node.interLeave[index]:
                                        for potNode in rootList:
                                            if topRootNode != potNode and potNode not in connectedRootNodes:
                                                if PhyloParser.isLineAndNodeConnected(node.interLeave[index], potNode):
                                                    if len(topRootNode.nodesIncluded + potNode.nodesIncluded) == len(list(set(topRootNode.nodesIncluded + potNode.nodesIncluded))):
                                                        topRootNode.nodesIncluded += potNode.nodesIncluded
                                                        connectedRootNodes.append(potNode)
                                                        node.otherTo[index] = potNode
                                                        topRootNode.numNodes += potNode.numNodes
                                                        potNode.whereFrom = node
                                                        potNode.origin = node.origin
        
        rootList = sorted(rootList, key = lambda x: -x.numNodes)

        image_data.rootList = rootList

        return image_data


    @staticmethod    
    def drawNode(whatever, node):
        color = (255, 0, 0)
        if node.root:
            x1, y1, x2, y2, length = node.root
            cv.rectangle(whatever, (x1, y1), (x2, y2), color=color, thickness=2)
        if node.branch:
            x1, y1, x2, y2, length = node.branch
            cv.rectangle(whatever, (x1, y1), (x2, y2), color=color, thickness=2)

        if node.upperLeave:
            x1, y1, x2, y2, length = node.upperLeave
            cv.rectangle(whatever, (x1, y1), (x2, y2), color=color, thickness=2)
        if node.lowerLeave:
            x1, y1, x2, y2, length = node.lowerLeave
            cv.rectangle(whatever, (x1, y1), (x2, y2), color=color, thickness=2)
        if not node.isBinary:
            for line in node.interLeave:
                x1, y1, x2, y2, length = line
                cv.rectangle(whatever, (x1, y1), (x2, y2), color=color, thickness=2)
        return whatever






    @staticmethod
    def drawNodesCoveredMask(node, nodesCoveredMask, mappingDictList):
        horLineMappingDict, verLineMappingDict = mappingDictList
        parent, verLineIndex, lineChildren = node
        # if parent:
        #     for line in horLineMappingDict['lineMapping'][parent]['lineGroup']:
        #         x1, y1, x2, y2, length = line
        #         nodesCoveredMask[y1:y2+1, x1:x2] = 255

        for line in verLineMappingDict['lineMapping'][verLineIndex]['lineGroup']:
            x1, y1, x2, y2, length = line
            nodesCoveredMask[y1:y2, x1-2:x2+3] = 255


        # for child in lineChildren:
        #     for line in horLineMappingDict['lineMapping'][child]['lineGroup']:
        #         x1, y1, x2, y2, length = line
        #         nodesCoveredMask[y1:y2+1, x1:x2] = 255






        
        
        
 
    
    

    
