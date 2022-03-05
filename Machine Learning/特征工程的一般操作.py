"""
特征工程：简言之，对原始数据进行精细化加工，使其更便于或更有利于解决实际问题。

在机器学习领域，一个好的模型往往基于三点：【1】足量的优质数据。【2】准备好的特征工程。【3】模型的选择。 其中准备好的特征工程可以具体概括为：基于样本（观测/个案）和属性（变量）所构成的数据集
我们需要从众多属性或变量中挑选或提取出有益于后续建模分析的那一些，这些“优质”的属性或变量即称为特征。例如回归分析中我们需要事先筛选出有价值的预测变量。

在进行特征工程之前，我们需要对原始数据集作大致的探索，主要关注点如下：
【1】样本量是否较充足？是否存在样本不均衡问题？（定类型数据）
【2】数据结构：整型（连续型 or 离散型），浮点型 or 字符串（哑变量处理）？
【3】是否作标准化？是否有缺失值？是否有异常值？
【4】统计分布情况：散点图（前后趋势），直方图（分布情况），箱线图（极值，均值，中位数）

下面我们介绍特征工程的一般流程以及对所选取特征的评价：
"""
#（1）数据清洗与变量转换

#（1.1）缺失值处理
"""
缺失值处理包括删除和填补两种：

删除：简单地，针对数据集中存在的缺失值，可以行删除（删除样本）；也可以列删除（删除变量）

填补：常用填补方法如：随机数填补，固定值填补，变量均值、中位数、众数填补，插值法填补，其他复杂的如机器学习算法填补
"""
import pandas as pd
import random

data1=pd.DataFrame(pd.read_csv("data/1.csv",header=0,encoding="utf-8"))

# 删除
data1.drop(columns="age",axis=1,inplace=True) # 删除含有缺失值的变量

data1.drop(2,axis=0,inplace=True) # 删除含有缺失值的第3行，删除多行可用[row1,row2,……]

data1.dropna(axis=0) # 删除缺失值的所有空行，axis=1则为删除空列

# 填补
data1["age"]=data1["age"].fillna(random.randint(10,15)) # 随机数填补（范围为10-15的整数）

data1["age"]=data1["age"].fillna(0) # 固定值填补

data1["age"]=data1["age"].fillna(data1["age"].mean()) # 均值填补

data1["age"]=data1["age"].fillna(data1["age"].median()) # 中位数填补

data1["age"]=data1["age"].interpolate() # 插值法填补

# （1.2）异常值处理
"""
我们需要判断是否存在异常值点：【1】单变量或两变量异常值监测：散点图观察法 【2】多变量异常值监测：聚类法（拟定聚类数，少部分自成一类的样本点可视作异常值点）

在挑出异常值点后，处理方法可以是直接删除，也可以对其重新填补
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data2=pd.DataFrame(pd.read_csv("data/2.csv",header=0,encoding="utf-8"))
X=np.array(range(len(data2["value"])))

fig=plt.figure()
plt.scatter(X,data2["value"])
plt.show() # 观察可知第4个样本点value值异常

data2["value"]=data2["value"].drop(3,axis=0,inplace=True) # 删除该异常值点

#（1.3）数据格式转换
import pandas as pd

data2=pd.DataFrame(pd.read_csv("data/2.csv",header=0,encoding="utf-8"))

data2["value"]=data2["value"].astype("string") # float为浮点型，int为整型，string为字符串型


#（1.4）数据过采样或欠采样
"""
针对于数据集中某一定类二值型变量，若类别0数量远大于类别1，则;
过采样：通过采样，扩充类别1的数量以达到类别0，1的目标比例（默认为1：1）
欠采样：减少类别0的样本数量以达到类别0，1的目标比例
"""
import pandas as pd
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler

data3=pd.DataFrame(pd.read_csv("data/3.csv",header=0,encoding='utf-8'))

X=data3.drop(columns="type").values # 只取数值，不取变量标签
y=data3["type"].values

ros=RandomOverSampler(random_state=0) 
X_sample, y_sample = ros.fit_resample(X,y) # 过采样
for i in range(len(X),len(X_sample)): # 拼接数据
    new_row=[y_sample[i]]
    for j in range(len(X_sample[i])):
        new_row.append(X_sample[i][j])
    data3.loc[i+1]=new_row    

rus=RandomUnderSampler(random_state=0)
X_sample, y_sample = rus.fit_resample(X,y) # 欠采样


#（1.5）变量转换
import pandas as pd

data2=pd.DataFrame(pd.read_csv("data/2.csv",header=0,encoding="utf-8"))

data2["value"]=data2[["value"]].apply(lambda x : (x-x.mean())/x.std()) # 标准化处理
data2["value"]=data2[["value"]].apply(lambda x: x/(x**2).sum()) # 正则化处理（单位向量化）
data2["value"]=data2[["value"]].apply(lambda x: (x-x.min())/(x.max()-x.min())) # 归一化处理（区间缩放）

for i in range(len(data2)): # 二值化处理（同理可以多值化处理）
    if data2.loc[i,"value"]>=12:
        data2.loc[i,"value"]=0
    else:
        data2.loc[i,"value"]=1

for i in range(len(data2)): # 哑变量（字符串）编码
    if data2.loc[i,"id"]=="A":
        data2.loc[i,"id"]=1
    elif data2.loc[i,"id"]=="B":
        data2.loc[i,"id"]=2
    else:
        data2.loc[i,"id"]=3

#-------------------------------------------------#

#（2）从原有变量中选择特征
"""
传统的有监督式机器学习建模包含分类和回归两大类，基于目标变量（因变量）与其他变量（自变量）之间的相关性，我们可通过求相关系数，卡方值（信息增益）等方法来选取特征。
这是选取特征最基本的方法

更进一步，还可以利用机器学习，深度学习算法来作更精细，准确的特征选择，此处不作更多介绍
"""
import pandas as pd
from scipy.stats import chi2_contingency

data4=pd.DataFrame(pd.read_csv("data/4.csv",header=0,encoding="utf-8"))
for i in range(len(data4)): # 二值化处理,添加新的一列
    if data4.loc[i,"weight"]>=40:
        data4.loc[i,"weight_level"]=1
    else:
        data4.loc[i,"weight_level"]=0
    
#（2.1）目标变量为数值型，定序型
data4["age"].corr(data4["weight"]) # Pearson相关系数（自变量为数值型，因变量为数值型）
data4["age"].corr(data4["level"],"spearman") # Spearman相关系数（自变量为数值型，因变量为定序型）
data4["age"].corr(data4["level"],"kendall") # Kendall相关系数（自变量和因变量均为定序型）

#（2.2）目标变量为定类型（自变量一般也为定类型）
Count=pd.crosstab(data4["sex"],data4["weight_level"]) # 交叉计数
Fre=Count.values.T # 获取交叉计数列表（基于自变量"weight_level")
chi2_contingency(Fre)[0] # 卡方值计算
"""
若各自变量数据非均衡，那么可以采用 卡方值/count(自变量=1) 来计算加权卡方值
类似的，还可以计算信息增益率，此处不作过多讲述
"""

#-------------------------------------------------#

#（3）基于原有变量构造新特征
"""
简而言之，就是利用现有的自变量来构造新的，有效的特征。如上，我们根据体重"weight"的数值将其二值化("weight">=40时赋值为1，小于则赋值为0)

构造新的特征要求我们对现有的特征足够了解，在此基础上，才能灵活构造新的，更有价值的特征。
"""
import pandas as pd

data4=pd.DataFrame(pd.read_csv("data/4.csv",header=0,encoding="utf-8"))

for i in range(len(data4)): # 由体重和身高信息构造BMI体重指数
    data4.loc[i,"BMI"]=data4.loc[i,"weight"]/data4.loc[i,"height"]**2

#--------------------------------------------------#

#（4）对原有变量作降维处理以提取特征
"""
当自变量数量过多时（甚至于出现维数灾难），我们往往需要对自变量信息进行压缩处理，通过降维的方法提取出新的，数量更少的自变量

此处，我们介绍一种最常用的降维方法：主成分分析法(PCA)
"""
import pandas as pd
from sklearn.decomposition import PCA

data4=pd.DataFrame(pd.read_csv("data/4.csv",header=0,encoding="utf-8"))

pca_data=PCA(n_components=2).fit_transform(data4.drop(columns="sex").values) # 对因变量"sex"外的自变量作PCA降维
data4["factor1"]=pca_data.T[0]; data4["factor2"]=pca_data.T[1] # 将提取的主成分添加到原数据集中

