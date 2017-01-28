# imports.
import pylab as plt
import numpy, math, random
import matplotlib as mpl

# Grammatical condition. 
# draw all points on the graph with a line connecting them.
# create an x and y vector to plot a line.
x = numpy.arange(0.0, 4.75, .25)
y = [0.14434708437847965, 0.2332945225298853, 0.3048689584628825, 0.3328899215315861, 0.82507287558848, 0.35944633403730597, 0.31187366145189526, 0.24908828691998636, 0.7876814131137068, 0.605771839036144, 0.3482931743516271, 0.2990048860041589, 0.28050477937805096, 0.30531068121234983, 0.31467978090397547, 0.36827186361720327, 0.7457505035723351, 0.2628295743101065, 0.31813941684404157]
x1 = numpy.arange(0.7, 4.5, .1)
y1 = [0.41, 0.5, 0.6, .75, .6, .5,.4,.33,.32, .29, .28, .4, .57, .71, .57, .4, .32, .33, .31, .32, .29, .33, .34, .42, .38, .35, .32, .35, .32, .28, .29, .35, .5, .71, .5, .35, .35, .3]
# draw the lines on the graph.
DORA,=plt.plot(x,y, 'r--', label='DORA')
Humans,=plt.plot(x1,y1, 'b-', label='Humans')
plt.legend([DORA, Humans], ['DORA', 'Humans'])
#plt.legend(handles=[DORA, Humans])
plt.axis([-.1,4.5,0,1])
# label the axes.
plt.xlabel('Hz')
plt.ylabel('power')
# add labels to x-axis.
plt.annotate('P Units/Sentences', xy=(.12,1.025), xycoords='axes fraction', annotation_clip=False)
plt.annotate('RB Units/Phrases', xy=(.38,1.025), xycoords='axes fraction', annotation_clip=False)
plt.annotate('PO Units/Words', xy=(.8,1.025), xycoords='axes fraction', annotation_clip=False)
plt.annotate('Units of Analysis', xy=(.4,1.08), xycoords='axes fraction', annotation_clip=False)
plt.grid(True)
plt.show()

# Word salad condition. 
# draw all points on the graph with a line connecting them.
# create an x and y vector to plot a line.
x = numpy.arange(0.0, 4.75, .25)
y = [0.14434708437847965, 0.2332945225298853, 0.3148689584628825, 0.3328899215315861, 0.32507287558848, 0.35944633403730597, 0.33187366145189526, 0.32908828691998636, 0.3576814131137068, 0.305771839036144, 0.3482931743516271, 0.2990048860041589, 0.28050477937805096, 0.30531068121234983, 0.31467978090397547, 0.36827186361720327, 0.7457505035723351, 0.2428295743101065, 0.29813941684404157]
# draw the lines on the graph.
DORA,=plt.plot(x,y, 'r--', label='DORA')
#plt.legend(handles=[DORA, Humans])
plt.axis([-.1,4.5,0,1])
# label the axes.
plt.xlabel('Hz')
plt.ylabel('power')
plt.grid(True)
plt.show()

# Jabberwocky condition. 
# draw all points on the graph with a line connecting them.
# create an x and y vector to plot a line.
x = numpy.arange(0.0, 4.75, .25)
y = [0.14434708437847965, 0.272945225298853, 0.3248689584628825, 0.3828899215315861, 0.83407287558848, 0.31944633403730597, 0.32187366145189526, 0.34908828691998636, 0.7476814131137068, 0.555771839036144, 0.3082931743516271, 0.3690048860041589, 0.46050477937805096, 0.31531068121234983, 0.35467978090397547, 0.26827186361720327, 0.7957505035723351, 0.2928295743101065, 0.27813941684404157]
# draw the lines on the graph.
DORA,=plt.plot(x,y, 'r--', label='DORA')
#plt.legend(handles=[DORA, Humans])
plt.axis([-.1,4.5,0,1])
# label the axes.
plt.xlabel('Hz')
plt.ylabel('power')
plt.grid(False)
plt.show()

# Adj-noun condition. 
# draw all points on the graph with a line connecting them.
# create an x and y vector to plot a line.
x = numpy.arange(0.0, 4.75, .25)
y = [0.12434708437847965, 0.232945225298853, 0.2248689584628825, 0.3228899215315861, 0.23407287558848, 0.31944633403730597, 0.22187366145189526, 0.24908828691998636, 0.7476814131137068, 0.255771839036144, 0.3082931743516271, 0.2690048860041589, 0.24050477937805096, 0.31531068121234983, 0.25467978090397547, 0.26827186361720327, 0.8157505035723351, 0.2428295743101065, 0.17813941684404157]
# draw the lines on the graph.
DORA,=plt.plot(x,y, 'r--', label='DORA')
#plt.legend(handles=[DORA, Humans])
plt.axis([-.1,4.5,0,1])
# label the axes.
plt.xlabel('Hz')
plt.ylabel('power')
# add labels to x-axis.
plt.annotate('P Units/Sentences', xy=(.12,1.025), xycoords='axes fraction', annotation_clip=False)
plt.annotate('RB Units/Phrases', xy=(.38,1.025), xycoords='axes fraction', annotation_clip=False)
plt.annotate('PO Units/Words', xy=(.8,1.025), xycoords='axes fraction', annotation_clip=False)
plt.annotate('Units of Analysis', xy=(.4,1.08), xycoords='axes fraction', annotation_clip=False)
plt.grid(False)
plt.savefig('noun-adj.eps', format='eps', dpi=1200)
plt.show()

# Adj-adj-noun condition. 
# draw all points on the graph with a line connecting them.
# create an x and y vector to plot a line.
x = numpy.arange(0.0, 4.75, .33333)
y = [0.14434708437847965, 0.252945225298853, 0.2948689584628825, 0.3028899215315861, 0.76407287558848, 0.23944633403730597, 0.7476814131137068, 0.235771839036144, 0.282931743516271, 0.2600048860041589, 0.26450477937805096, 0.21531068121234983, 0.7957505035723351, 0.2928295743101065, 0.27813941684404157]
# draw the lines on the graph.
DORA,=plt.plot(x,y, 'r--', label='DORA')
#plt.legend(handles=[DORA, Humans])
plt.axis([-.1,4.5,0,1])
# label the axes.
plt.xlabel('Hz')
plt.ylabel('power')
plt.grid(False)
plt.show()

# Adj-adj-adj-noun condition. 
# draw all points on the graph with a line connecting them.
# create an x and y vector to plot a line.
x = numpy.arange(0.0, 4.75, .33333)
y = [0.15434708437847965, 0.2532945225298853, 0.26648689584628825, 0.7428899215315861, 0.76407287558848, 0.23944633403730597, 0.7476814131137068, 0.2357718390364, 0.282931743516271, 0.2600048860041589, 0.26450477937805096, 0.21531068121234983, 0.7957505035723351, 0.2928295743101065, 0.27813941684404157]
# draw the lines on the graph.
DORA,=plt.plot(x,y, 'r--', label='DORA')
#plt.legend(handles=[DORA, Humans])
plt.axis([-.1,4.5,0,1])
# label the axes.
plt.xlabel('Hz')
plt.ylabel('power')
plt.grid(False)
plt.show()

# plot all the adj-noun conditions together in a single-plot. 
x = numpy.arange(0.0, 4.75, .25)
x1 = numpy.arange(0.0, 4.75, .33333)
y = [0.14434708437847965, 0.2332945225298853, 0.3148689584628825, 0.3328899215315861, 0.32507287558848, 0.35944633403730597, 0.33187366145189526, 0.32908828691998636, 0.3576814131137068, 0.305771839036144, 0.3482931743516271, 0.2990048860041589, 0.28050477937805096, 0.30531068121234983, 0.31467978090397547, 0.36827186361720327, 0.7457505035723351, 0.2428295743101065, 0.29813941684404157]
y2 = [0.14434708437847965, 0.272945225298853, 0.3248689584628825, 0.3828899215315861, 0.83407287558848, 0.31944633403730597, 0.32187366145189526, 0.34908828691998636, 0.7476814131137068, 0.555771839036144, 0.3082931743516271, 0.3690048860041589, 0.46050477937805096, 0.31531068121234983, 0.35467978090397547, 0.26827186361720327, 0.7957505035723351, 0.2928295743101065, 0.27813941684404157]
y3 = [0.12434708437847965, 0.232945225298853, 0.2248689584628825, 0.3228899215315861, 0.23407287558848, 0.31944633403730597, 0.22187366145189526, 0.24908828691998636, 0.7476814131137068, 0.255771839036144, 0.3082931743516271, 0.2690048860041589, 0.24050477937805096, 0.31531068121234983, 0.25467978090397547, 0.26827186361720327, 0.8157505035723351, 0.2428295743101065, 0.17813941684404157]
# draw the lines on the graph.
DORA1,=plt.plot(x,y, 'b:', label='DORA-Word List')
DORA2,=plt.plot(x,y2, 'g-', label='DORA-Jabberwocky')
DORA3,=plt.plot(x,y3, 'r--', label='DORA-Phrases')
plt.legend([DORA1, DORA3, DORA2], ['DORA-Word List', 'DORA-Phrases', 'DORA-Jabberwocky'])
#plt.legend(handles=[DORA, Humans])
plt.axis([-.1,4.5,0,1])
# label the axes.
plt.xlabel('Hz')
plt.ylabel('power')
# add labels to x-axis.
plt.annotate('P Units/Sentences', xy=(.12,1.025), xycoords='axes fraction', annotation_clip=False)
plt.annotate('RB Units/Phrases', xy=(.38,1.025), xycoords='axes fraction', annotation_clip=False)
plt.annotate('PO Units/Words', xy=(.8,1.025), xycoords='axes fraction', annotation_clip=False)
plt.annotate('Units of Analysis', xy=(.4,1.08), xycoords='axes fraction', annotation_clip=False)
plt.grid(False)
plt.savefig('adj-plots.eps', format='eps', dpi=1200)
plt.show()


