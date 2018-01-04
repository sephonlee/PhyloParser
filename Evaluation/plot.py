import csv
import numpy as np
import matplotlib.pyplot as plt


# ground_truth_path  = "/Users/sephon/Desktop/Research/VizioMetrics/Corpus/Phylogenetic/CNN_corpus/high_quality_tree_result_line_corner_v2_20170228_best.csv"

ground_truth_path = "/Users/sephon/Desktop/Research/VizioMetrics/Corpus/Phylogenetic/CNN_corpus/hqtree_base_fixTree_orphanHint_20170307.csv"

data = []
x = []
y = []
with open(ground_truth_path ,'rb') as incsv:
    reader = csv.reader(incsv, dialect='excel')
    reader.next()

    for row in reader:
        print row
        data.append([row[0], int(row[1]), int(row[2]), float(row[3])])
        x.append(float(row[2]))
        y.append(float(row[3]))
        
print data[0:5]
    
print zip(x,y)

plt.scatter(x, y)
plt.ylabel('Error Rate')
plt.xlabel('Number of Nodes')
plt.show()

    
# tick = ["0", "0-0.1", "0.1-0.2", "0.2-0.5", "0.5-0.8", "0.8+"]
cat = np.array([0, 0.1, 0.2, 0.5, 0.8])
tick = []
for i, c in enumerate(cat):
    if i == 0:
        label = str(c)
    else:
        label = "%.1f - %.1f" %(cat[i-1], c)
    tick.append(label)

tick.append("%.1f+" %cat[-1]) 
print tick

counts = [0] * (len(cat)+1)
avgs = [0] * (len(cat)+1)

for row in data:
    tmp = row[3] - cat
#     print row[3], tmp
    tmp = np.where(tmp>=0)[0].tolist()

    if row[3] == 0:
        counts[0] += 1
        avgs[0] += row[2]
    else:
        counts[tmp[-1] + 1] += 1
        avgs[tmp[-1] + 1] += row[2]
        
        if tmp[-1] + 1 ==5:
            print "num node", row[2]
        
#     print "index", tmp[-1]
  
print "average node: ", sum(avgs)/sum(counts)  
print avgs
for i, c in enumerate(counts):
    avgs[i] = avgs[i]/(float(c)+0.0000001)
    
    
print counts
print avgs
    

    
fig, ax1 = plt.subplots()

# Example data

y_pos = np.arange(len(tick))
performance = 3 + 10 * np.random.rand(len(tick))

x_avg_offset = 200
x_avg = range(0,240,40)
x_avg_tick = x_avg[:]
x_avg_tick.sort(reverse = True)





fontsize = 16
ax1.barh(y_pos, counts, align='center',
        color='lightblue')

ax2 = ax1.twiny()
ax2.barh(y_pos, avgs, height = 0.2, left = (x_avg_offset - np.array(avgs)), align='center',
        color='lightcoral')
ax2.set_xlim(0,x_avg_offset)


for i, c in enumerate(counts):
    per = c / float(sum(counts)) * 100
    ax1.text(1, y_pos[i]+0.1, "%.1f%%"%(per), fontsize = fontsize)



ax1.set_yticks(y_pos)
ax1.set_yticklabels(tick, fontsize = fontsize-2)
ax1.set_xticklabels(range(0,70,10), fontsize = fontsize-2)

ax2.set_xticks(x_avg)
ax2.set_xticklabels(x_avg_tick, fontsize = fontsize-2)

ax1.invert_yaxis()  # labels read top-to-bottom
ax1.set_xlabel('Count', fontsize = fontsize)
ax2.set_xlabel('Average Num. of Nodes', fontsize = fontsize)
ax1.set_ylabel('Error Rate', fontsize = fontsize)
ax1.set_title('Population of Sample (Binned by Error Rate)', y=1.08, fontsize = fontsize)

plt.show()

    