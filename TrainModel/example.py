import cv2 as cv
from GroundTruthConverter import *
from PhyloParser import *

## Classifier Path
clfPath = "model/RF.pkl"
phyloParser = PhyloParser(clfPath = clfPath)
    
# Read image file
filePath = "/Users/sephon/Desktop/Research/VizioMetrics/Corpus/Phylogenetic/CNN_corpus/high_quality_tree/PMC2233639_1471-2199-8-95-2.jpg"
image = cv.imread(filePath,0)

# Normalize image dimension based on linewidth
image, w = PhyloParser.resizeImageByLineWidth(image)
# PhyloParser.displayImage(image)

image_data = ImageData(image)
# Preprocess
image_data = phyloParser.preprocces(image_data, debug= False)
# Line Detection
image_data = phyloParser.detectLines(image_data, debug = False)
image_data = phyloParser.getCorners(image_data, debug = False)   
image_data = phyloParser.makeLinesFromCorner(image_data, debug = False)
image_data = phyloParser.includeLinesFromCorners(image_data)
image_data = phyloParser.postProcessLines(image_data)
image_data = phyloParser.groupLines(image_data)
image_data = phyloParser.matchLineGroups(image_data, debug = False)

# Read Species Name
image_data = phyloParser.getSpecies(image_data, debug = False)

# Reconstruct the tree structure
image_data = phyloParser.constructTree(image_data, tracing = False , debug = False)

# Result in string format
print image_data.treeStructure
