import numpy as np
import matplotlib.pyplot as plt

results = np.load("results.npy", allow_pickle=True)

plt.plot(results.item().get('epoch'), results.item().get('average'), label="average rewards")
plt.plot(results.item().get('epoch'), results.item().get('averageQ'), label="average Qs")
plt.plot(results.item().get('epoch'), results.item().get('maximum'), label="max rewards")
plt.plot(results.item().get('epoch'), results.item().get('minimum'), label="min rewards")
plt.legend(loc=4)
plt.show()
