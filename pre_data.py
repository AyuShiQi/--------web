import numpy as np
import pandas as pd

df = np.asarray(pd.read_csv('./train.csv'))
df = pd.DataFrame(df)
df.to_csv('./train.csv')
