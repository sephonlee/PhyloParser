class TrunkNode():
    
    startPoint = None
    nextStartPoint = []
    parentStartPorint = None
    buds = []
    parent = None
    top = None
    bot = None
    interLines = []
    leaves = []


    upperLine = None
    lowerLine = None
    trunkLine = None
    rootLine = None    
    nonBinaryLines = None
    upperGo = None
    lowerGo = None
    interGo = None
    
    def __init__(self, startPoint):
        self.startPoint = startPoint
        self.buds = []
        if self.nonBinaryLines == None:
            self.nonBinaryLines = []
        if self.interGo == None:
            self.interGo = []
        
    def __str__(self):
#         return "startPoint: (" + str(self.startPoint[0]) + "," + str(self.startPoint[1]) + ")"
        return "startPoint: " + str(self.startPoint)
#         return "startPoint: " + "(" + ",".join(self.startPoint) + ")\n" \
#         +  ", buds: " + str(self.buds) \
#         + ", top: ", self.top.toString() + ", bot: ", self.bot.toString() \
#         + ", interLines: ", str(self.interLines), ", leaves: " + str(self.leaves)

    def getTrunkInfo(self):
        print '------trunk Information-------'
        print 'rootLine', self.rootLine
        print 'branch: ', self.trunkLine
        print 'upperLeave: ', self.upperLine
        print 'lowerLeave: ', self.lowerLine
        print 'interLines:', self.interLines
        print 'leaves:', self.leaves
        print 'nonBinaryLines', self.nonBinaryLines
        print 'upperGo:', self.upperGo
        print 'lowerGo:', self.lowerGo
        print 'interGo:', self.interGo
