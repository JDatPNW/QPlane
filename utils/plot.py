import numpy as np
import matplotlib.pyplot as plt

plotAverage = True
plotAverageQ = True
plotMaximum = True
plotMinimum = True
plotEpsilon = True

results = np.load("results.npy", allow_pickle=True)

if(plotAverage):
    average = 2. * (np.array(results.item().get('average')) - np.min(np.array(
        results.item().get('average')))) / np.ptp(np.array(results.item().get('average'))) - 1
if(plotAverageQ):
    averageQ = 2. * (np.array(results.item().get('averageQ')) - np.min(np.array(
        results.item().get('averageQ')))) / np.ptp(np.array(results.item().get('averageQ'))) - 1
if(plotMaximum):
    maximum = 2. * (np.array(results.item().get('maximum')) - np.min(np.array(
        results.item().get('maximum')))) / np.ptp(np.array(results.item().get('maximum'))) - 1
if(plotMinimum):
    minimum = 2. * (np.array(results.item().get('minimum')) - np.min(np.array(
        results.item().get('minimum')))) / np.ptp(np.array(results.item().get('minimum'))) - 1
if(plotEpsilon):
    epsilon = 2. * (np.array(results.item().get('epsilon')) - np.min(np.array(
        results.item().get('epsilon')))) / np.ptp(np.array(results.item().get('epsilon'))) - 1

if(plotAverage):
    plt.plot(results.item().get('epoch'), average, label="average rewards")
if(plotAverageQ):
    plt.plot(results.item().get('epoch'), averageQ, label="average Qs")
if(plotMaximum):
    plt.plot(results.item().get('epoch'), maximum, label="max rewards")
if(plotMinimum):
    plt.plot(results.item().get('epoch'), minimum, label="min rewards")
if(plotEpsilon):
    plt.plot(results.item().get('epoch'), epsilon, label="epsilon")
plt.title("Normalized Results")
plt.legend(loc=4)
plt.show()

if(plotAverage):
    plt.plot(results.item().get('epoch'), results.item().get(
        'average'), label="average rewards")
if(plotAverageQ):
    plt.plot(results.item().get('epoch'), results.item().get(
        'averageQ'), label="average Qs")
if(plotMaximum):
    plt.plot(results.item().get('epoch'), results.item().get(
        'maximum'), label="max rewards")
if(plotMinimum):
    plt.plot(results.item().get('epoch'), results.item().get(
        'minimum'), label="min rewards")
if(plotEpsilon):
    plt.plot(results.item().get('epoch'),
             results.item().get('epsilon'), label="epsilon")
plt.title("Results")
plt.legend(loc=4)
plt.show()
