import numpy as np
import pandas as pd
pd.set_option("display.max_rows", None, "display.max_columns", None)

table = np.load("model.npy")
table = pd.DataFrame(table)
table = table[(table.T != 0).any()]


print(table)
