import pandas as pd

# Load a dataset
data = {'Age': [25, 30, 40, 35],
        'Income': [50000, 60000, 70000, 65000]}
df = pd.DataFrame(data)

# Descriptive statistics
print(df.describe())

# Hypothesis testing (t-test)
from scipy import stats

t_statistic, p_value = stats.ttest_ind(df['Age'], df['Income'])
print("T-statistic:", t_statistic)
print("P-value:", p_value)
