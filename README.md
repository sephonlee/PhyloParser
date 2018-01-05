# PhyloParser: A Hybrid Algorithm for Extracting Phylogenies from Dendrograms

## Abstract

We consider a new approach to extracting information from dendograms in the biological literature representing phylogenetic trees. Existing algorithmic approaches to extract these relationships rely on tracing tree contours and are very sensitive to image quality issues, but manual approaches require significant human effort and cannot be used at scale. We introduce PhyloParser, a fully automated, end-to-end system for automatically extracting species relationships from phylogenetic tree diagrams using a multi-modal approach to digest diverse tree styles. Our approach automatically identifies phylogenetic tree figures in the scientific literature, extracts the key components of tree structure, reconstructs the tree, and recovers the species relationships. We use multiple methods to extract tree components with high recall, then filter false positives by applying topological heuristics about how these components fit together. We present an evaluation on a real-world dataset to quantitatively and qualitatively demonstrate the efficacy of our approach. Our classifier achieves 89% recall and 99% precision, with a low average error rate relative to previous approaches. We aim to use PhyloParser to build a linked, open, comprehensive database of phylogenetic information that covers the historical literature as well as current data, and then use this resource to identify areas of disagreement and poor coverage in the biological literature.

### VizioMetrics Project Website
http://viziometrics.org/about/

### ICDAR2017 Paper
https://staff.washington.edu/sephon/publication/LeeICDAR2017.pdf

### Bibtex
```sh
Bibtex:
	@article{lee2017phyloparser,
                  author = {Lee, Poshen and Yang, T. Sean and West, Jevin and Howe, Bill},
                  title = {PhyloParser: A Hybrid Algorithm for Extracting Phylogenies from Dendrograms},
                  booktitle = {Document Analysis and Recognition (ICDAR), 2017 14th International Conference on},
                  year = {2017}
                }
```

## Requirements
```sh
opencv 3.0.0
numpy 1.11.1
matplotlib 1.4.3
ete3 3.0.0b35
peakutils 1.0.3
sklearn 0.18.1 (higher version can cause errors)
scipy 0.17.1
skimage 0.12.3
pytesseract 0.1.6
tesseract 3.04.01
```

## Example Code

```sh
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
```

