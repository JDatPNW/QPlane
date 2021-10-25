import re
import os
import numpy as np
import matplotlib.pyplot as plt

selector = "pitch"  # Canbe "pitch", "roll", or "rewards"

plotList = []
dir = "./"
if(selector == "pitch"):
    regex = re.compile("pitch_ep*")

elif(selector == "roll"):
    regex = re.compile("roll_ep*")
elif(selector == "rewards"):
    regex = re.compile("rewards_ep*")

for root, dirs, files in os.walk(dir):
    for file in files:
        if regex.match(file):
            plotList.append([np.load(file, allow_pickle=True), file])

for i in range(len(plotList)):
    plt.plot(plotList[i][0], label=plotList[i][1])


plt.title(selector.capitalize() + " per Step over Episode")
plt.xlabel("steps")
plt.ylabel(selector)
plt.legend(loc=4)
plt.show()
