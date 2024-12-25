import io
import numpy as np
import pandas as pd
import keras
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import seaborn as sns

chicago_taxi_dataset = pd.read_csv('https://raw.githubusercontent.com/plotly/datasets/master/2014_chicago_crime.csv')

training_df = chicago_taxi_dataset[['TRTP_MILES', 'TRTP_SECONDS', 'FARE', 'COMPANY', 'PAYMENT_TYPE', 'PURPOSE', 'TIP_RATE']]

print('Read dataset completed successfully')
print('Total number of rows:{0}'.format(len(training_df.index)))
training_df.head(200)

print('Total number of rows:{0}\n'.format(len(training_df.index)))
training_df.describe(include='all')

# 最高票价是多少？
max_fare = training_df['FARE'].max()  # 最高票价
print("What is the maximum fare? \t\t\t\tAnswer: ${fare:.2f}".format(fare = max_fare)) # 输出最高票价

# 所有行程的平均距离是多少？
mean_distance = training_df['TRIP_MILES'].mean() # 平均距离
print("What is the mean distance across all trips? \t\tAnswer: {mean:.4f} miles".format(mean = mean_distance)) # 输出平均距离

# 数据集中有多少出租车公司？
num_unique_companies =  training_df['COMPANY'].nunique() # 出租车公司数量，nunique指的是不重复的数量
print("How many cab companies are in the dataset? \t\tAnswer: {number}".format(number = num_unique_companies)) # 输出出租车公司数量

# 最常见的付款方式是什么？
most_freq_payment_type = training_df['PAYMENT_TYPE'].value_counts().idxmax() # 最常见的付款方式，idxmax()返回最大值所在的索引
print("What is the most frequent payment type? \t\tAnswer: {type}".format(type = most_freq_payment_type)) # 输出最常见的付款方式

# 是否有任何特性缺少数据？
missing_values = training_df.isnull().sum().sum() # 缺失值数量，sum()求和
print("Are any features missing data? \t\t\t\tAnswer:", "No" if missing_values == 0 else "Yes") # 输出是否有缺失值
"""
输出：
What is the maximum fare? 						Answer: $159.25
What is the mean distance across all trips? 	Answer:  8.2895 miles
How many cab companies are in the dataset? 	Answer: 31
What is the most frequent payment type? 		Answer: CASH
Are any features missing data? 					Answer: No
"""

training_df.corr(numeric_only=True) # 相关性分析
# Which feature correlates most strongly to the label FARE?
# ---------------------------------------------------------
answer = '''
The feature with the strongest correlation to the FARE is TRIP_MILES.
As you might expect, TRIP_MILES looks like a good feature to start with to train
the model. Also, notice that the feature TRIP_SECONDS has a strong correlation
with fare too.
'''
print(answer)


# Which feature correlates least strongly to the label FARE?
# -----------------------------------------------------------
answer = '''The feature with the weakest correlation to the FARE is TIP_RATE.'''
print(answer)

sns.pairplot(training_df, x_vars=["FARE", "TRIP_MILES", "TRIP_SECONDS"], y_vars=["FARE", "TRIP_MILES", "TRIP_SECONDS"]) # 散点图

# How did raising the learning rate impact your ability to train the model?
# -----------------------------------------------------------------------------
answer = """
When the learning rate is too high, the loss curve bounces around and does not
appear to be moving towards convergence with each iteration. Also, notice that
the predicted model does not fit the data very well. With a learning rate that
is too high, it is unlikely that you will be able to train a model with good
results.
"""
print(answer)

# How did lowering the learning rate impact your ability to train the model?
# -----------------------------------------------------------------------------
answer = '''
When the learning rate is too small, it may take longer for the loss curve to
converge. With a small learning rate the loss curve decreases slowly, but does
not show a dramatic drop or leveling off. With a small learning rate you could
increase the number of epochs so that your model will eventually converge, but
it will take longer.
'''
print(answer)

# Did changing the batch size effect your training results?
# -----------------------------------------------------------------------------
answer = '''
Increasing the batch size makes each epoch run faster, but as with the smaller
learning rate, the model does not converge with just 20 epochs. If you have
time, try increasing the number of epochs and eventually you should see the
model converge.
'''
print(answer)