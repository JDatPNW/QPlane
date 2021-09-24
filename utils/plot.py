import numpy as np
import matplotlib.pyplot as plt

plotAverage = True
plotAverageQ = True
plotMaximum = True
plotMinimum = True
plotEpsilon = True

plotNormalized = False

results = np.load("results.npy", allow_pickle=True)

if(plotNormalized):
    if(plotAverage):
        average = (np.array(results.item().get('average')) - np.min(np.array(
            results.item().get('average')))) / np.ptp(np.array(results.item().get('average')))
    if(plotAverageQ):
        averageQ = (np.array(results.item().get('averageQ')) - np.min(np.array(
            results.item().get('averageQ')))) / np.ptp(np.array(results.item().get('averageQ')))
    if(plotMaximum):
        maximum = (np.array(results.item().get('maximum')) - np.min(np.array(
            results.item().get('maximum')))) / np.ptp(np.array(results.item().get('maximum')))
    if(plotMinimum):
        minimum = (np.array(results.item().get('minimum')) - np.min(np.array(
            results.item().get('minimum')))) / np.ptp(np.array(results.item().get('minimum')))
    if(plotEpsilon):
        epsilon = (np.array(results.item().get('epsilon')) - np.min(np.array(
            results.item().get('epsilon')))) / np.ptp(np.array(results.item().get('epsilon')))

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
    plt.xlabel("episodes")
    plt.ylabel("reward")
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
plt.xlabel("episodes")
plt.ylabel("reward")
plt.legend(loc=4)
plt.show()
