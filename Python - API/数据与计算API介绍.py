# numpy：主要用于矩阵和高维数组（不同于列表在于元素全为数字）运算，下面是最基本的运用

#  数组创建
import numpy as np

A1=np.array([1,2,3]) # 一维数组，也可以是np.array((1,2,3))

A2=np.array([[1, 2, 3],[4, 5, 6]]) # 二维数组

A3=np.array([[[1, 2, 3],[4, 5, 6]],  [[1, 2, 3],[4, 5, 6]]]) # 三维数组

A4=np.full((2,3,4),0) # 元素值全为0的三维数组

"""
[[[0 0 0 0]
  [0 0 0 0]
  [0 0 0 0]]

 [[0 0 0 0]
  [0 0 0 0]
  [0 0 0 0]]]
"""

A5=np.arange(5) # 一维等差数组（0，1，2，3，4） 也可以是np.array(range(5))

A6=np.arange(1,7).reshape(2,3) # 二维等差数组，reshape函数用于对数组形状重构

"""
[[1 2 3]
 [4 5 6]]
"""

A6_0=np.arange(1,5).reshape(2,2);  A6_1=np.arange(5,9).reshape(2,2)
A6=np.append(A6_0,A6_1,axis=0) # 两个数组按列拼接，axis=1则为按行拼接

A7=np.linspace(0,1,10) # 将0到1以内的数等分为30份
"""
[0.         0.11111111 0.22222222 0.33333333 0.44444444 0.55555556
 0.66666667 0.77777778 0.88888889 1.        ]
"""

A8=np.eye(3) # 秩为3的单位矩阵（方阵）

"""
[[1. 0. 0.]
 [0. 1. 0.]
 [0. 0. 1.]]
"""

A9=np.random.rand(2,3) # 二维随机数(0到1内)数组
"""
[[0.81562224 0.77571423 0.43941307]
 [0.46618214 0.67260011 0.03868521]]

类似的，还有np.random.randint(1,10,size=(2,3)),  np.random.uniform(1,10,size=(2,3))
"""
A10=np.random.uniform(size=1000) # 均匀分布随机数组（0-1均匀分布）
A10=np.random.normal(0,1,size=1000) # 正态分布随机数组（均值为0，方差为1）
A10=np.random.binomial(10,0.6,size=1000) # 二项分布随机数组（n为10，概率为0.6）
A10=np.random.exponential(0.2,size=1000) # 指数分布随机数（lambda=0.2）
A10=np.random.poisson(5,size=1000) # 泊松分布随机数（lambda=5）
"""
类似的，还有np.random.geometric(0.6,size=1000)_几何分布  np.chisquare(10,size=1000)_卡方分布等等
"""

# 数组运算
# (1)数组内元素运算
import numpy as np

A2=np.array([[1, 2, 3],[4, 5, 6]]) 

sum=np.sum(A2) # 数组内所有元素求和，结果为21

sum_col=np.sum(A2,axis=0) # 列元素求和，结果为[5 7 9]

sum_row=np.sum(A2,axis=1) # 行元素求和，结果为[6 15]

mean=np.mean(A2) # 数组内元素求均值，结果为3.5，类似的，可求中位数np.median

var=np.var(A2) # 数组内元素求方差，结果为2.9166666666666665，类似的，可求标准差np.std

min=np.min(A2) # 数组内元素求最小值，结果为1，类似的，可求最大值np.max

# (2)数组间元素运算
import numpy as np

A2=np.array([[1, 2, 3],[4, 5, 6]]) 
A3=np.array([[11, 12, 13], [14, 15, 16]])

A2+A3 # 数组（矩阵）元素相加
"""
[[12 14 16]
 [18 20 22]]
"""
A2-A3 # 数组（矩阵）元素相减

A2*A3 # 数组（矩阵）元素相乘

A2/A3 # 数组（矩阵）元素相除

# (3)矩阵运算
import numpy as np

A2=np.array([[1, 2, 3],[4, 5, 6]]) # 2x3矩阵（数组）
A3=np.array([[1, 2], [3, 4], [5, 6]]) # 3x2矩阵（数组）

B=np.dot(A2,A3) # 矩阵相乘
"""
[[22 28]
 [49 64]]
"""

B=A2.T # 矩阵转置

A4=np.array([[1, 2], [3, 4]]) # 2x2方阵
det=np.linalg.det(A4) # 矩阵求行列式
B=np.linalg.inv(A4) # 矩阵求逆

# (4)广播运算
import numpy as np

A2=np.array([[1, 2, 3],[4, 5, 6]])

A2+2 # 数组内所有元素加2

A2-2 # 数组内所有元素减2

A2*2 # 数组内所有元素乘以2

A2/2 # 数组内所有元素除以2

A3=np.array([1, 2, 3])
A2+A3 # 数组每一行加上一维数组A3

# (5)数组排序
import numpy as np

A2=np.array([1,5,7,2,9,3,10])

A3=np.array([[1,4,6],[7,3,5],[9,5,6]])

A2=np.sort(A2)[::1] # 升序并返回元素值，[::-1]则为降序

A2=np.argsort(A2)[::1] # 升序并返回索引号，[::-1]则为降序

A3=np.sort(A3,axis=0) # 按列排序，axis=1则为按行排序


# 数组元素索引
import numpy as np

A2=np.array([[1, 2, 3],[4, 5, 6]])

A2[1,0] # 类比A2[1][0]

A2[:2,1:3] # 取第一行到第二行，第二列到第三列的元素
"""
[[2 3]
 [5 6]]
"""

A2[1,2] # 取第二行，第三列的元素，即6

A2[A2>2 & A2<5] # 从数组中选出大于2，小于5的元素值，组成一个一维列表[3,4]

#-------------------------------------------------------#

# scipy：主要用于科学计算，包括积分，方程组求解，概率统计，拟合与插值等等，下面是最基本的运用

# 积分
from scipy.integrate import quad,dblquad # 其中quad为一重积分函数，ablquad为二重积分函数

def f1(x): # 定义积分函数
  return (1-x**2)**0.5

value,error=quad(f1,-1,1) # 输入积分函数和下，上限，返回积分值和误差
"""
以上也可采用lambda表达式，即：quad(lambda x : (1-x**2)**0.5, -1, 1)，不过一般采用独立定义函数的方式
"""
def f2(x,y):
  return (1-x**2-y**2)**0.5

value,error=dblquad(f2,-1,1,(lambda x:-f1(x)),(lambda x: f1(x)))
"""
与上面不同的是，(lambda x:-f1(x))和(lambda x: f1(x))为二重积分第二个参数y的下限和上限
"""

# 线性方程组求解（可用numpy模块中的函数，也可用scipy模块中的函数求解）
import numpy as np
A=np.random.rand(3,3)
B=np.random.rand(3,1)
X1=np.linalg.solve(A,B) # 方法一：直接求解，得到解向量
X2=np.dot(np.linalg.inv(A),B) # 方法二：求A^(-1)*B，前提条件是A可逆

from scipy import linalg
AA=linalg.lu_factor(A)
X3=linalg.lu_solve(AA,B) # 方法三：先对矩阵A作LU分解，然后传递给lu_solve并求解（推荐）

"""
det=linalg.det(A) 方阵求行列式值
A_inv=linalg.inv(A) 满秩矩阵求逆
e_values,e_vector=linalg.eig(A) 矩阵求特征值与特征向量

注意，上述求解只当系数矩阵的秩等于增广矩阵的秩时，并且系数矩阵为满秩矩阵时，求解才可能；下面我们介绍更常用
最小二乘求解法，此时无论系数矩阵的秩小于未知数的个数（无穷解），还是大于未知数的个数，都可以求解
"""
import numpy as np
from scipy import linalg
A1=np.random.rand(3,4); B1=np.random.rand(3,1)
A2=np.random.rand(3,2); B2=np.random.rand(3,1)
X11=np.linalg.lstsq(A1,B1)[0] # 方法一：使用numpy的linalg模块中的lstsq函数求解
X12=np.linalg.lstsq(A2,A2)[0]
X12=linalg.lstsq(A1,B1)[0] # 方法二：使用scipy的linalg模块中的lstsq函数求解（推荐）
X22=linalg.lstsq(A2,B2)[0]

# 非线性方程组求解
from math import sin
from scipy import optimize

def f(X): 
  return [
    5*X[1]+3,
    4*X[0]*X[0]-2*sin(X[1]*X[2]),
    X[1]*X[2]-1.5
  ]
"""
定义非线性方程组，如下：
5*x1+3=0
4*x0^2-2*sin(x1*x2)=0
x1*x2-1.5=0
"""
X=optimize.fsolve(f,[1,1,1]) # [1,1,1]为给定的初始解，经过不断迭代求解后得出近似解X
X=optimize.root(f,[1,1,1]) # 非线性求解的第二种方法

# 线性拟合：本质上和最小二乘法求解未知数相同，求出参数来得到线性拟合/回归函数
import numpy as np
from matplotlib import pyplot
from scipy import optimize

X=np.array([1, 1.5, 2.5, 3, 5, 5.5, 7.5, 8, 9, 9.5]) # 定义解释变量X和响应变量Y
Y=np.array([2.3, 3.5, 3, 4.5, 7, 6, 9, 13, 15, 16.5])

def residual(Par): # 定义残差函数，其中Par[0]为斜率k，Par[1]为截距b
  return Y-(Par[0]*X)-Par[1]

Par=optimize.leastsq(residual,[1,0])[0]  # 求解斜率和截距，得到拟合函数

Y1=Par[0]*X+Par[1] # 得到拟合值y1

pyplot.scatter(X,Y,color="blue")
pyplot.plot(X,Y1,color="red")
pyplot.show()


# 插值（与拟合不同，需要插值曲线/曲面经过每一个已知点，且可非线性）
import numpy as np
from matplotlib import pyplot
from scipy.interpolate import interp1d  # 一维插值模块interp1d，类似的还有二位插值函数interp2d等等

X=np.linspace(0,1,30) # 用已有的(x,y)来构建插值函数 
Y=np.sin(5*X)+np.cos(10*X) 

fun=interp1d(X,Y, kind="zero") # 零次插值函数
fun=interp1d(X,Y, kind="linear") # 一次插值函数
fun=interp1d(X,Y, kind="quadratic") # 二次插值函数
fun=interp1d(X,Y, kind="cubic") # 三次插值函数
X1=np.linspace(0,1,100);  Y1=fun(X1) # 由插值函数fun来求x1的所有函数值Y1
pyplot.plot(X1,Y1)
pyplot.show()

# 概率统计方法：以正态分布为例,loc为模型的位置参数，此处指正态分布均值；scale为尺度参数，此处指正态分布标准差
import numpy as np
from scipy import stats
from matplotlib import pyplot

stats.norm.pdf(0,loc=0,scale=1) # x=0时的概率密度值
stats.norm.cdf(0,loc=0,scale=1) # x=0时的累积概率值
stats.norm.ppf(0.5,loc=0,scale=1) # 概率p=0.5时的分位数
stats.binom.pmf(5,n=10,p=0.6) # x=5时的二项分布B(10,0.6)的概率值

X=stats.norm.rvs(loc=0,scale=1,size=10000) # 获取正态分布N(0,1)的10000个样本
np.mean(X); np.median(X); np.var(X); np.std(X); stats.skew(X); stats.kurtosis(X) # 均值，方差等统计指标
pyplot.hist(X) # 查看直方图，观察可知所取样本来自于正态分布N(mean=1, variance=4)
pyplot.show() 

Y=stats.binom.rvs(n=10,p=0.6,size=10000) # 获取二项分布B(10,0.6)的10000个样本
pyplot.hist(Y)
pyplot.show()


#----------------------------------------------------------#

# pandas：主要用于对数据表（变量/特征 与 样本/观测 所组成的数据表）的运用，下面是最基本的运用
import pandas as pd

# 数据表创建/导入与导出
data1=pd.DataFrame(pd.read_csv("1.csv",header=0,encoding="utf8")) # 导入数据表，以csv文件为例

data2=pd.DataFrame(  # 创建数据表
  {"id":["06","07","08","09","10"],
  "name":["Alex","Lio","Peng","Eksi","Shu"],
  "age":[12,10,12,15,11]
  },
  columns=["id","name","age"]
)

data1.info() # 查看数据表基本信息，包括维度，变量名称，数据格式

data1.values # 查看全部数据值

print(data1) # 查看数据表

data1["name","age"] # 选择“name"列和"age"列的数据值

data1.loc[0:3,["name","age"]] # 选择第1到第4行，“name"列和"age"列的数据值
"""
    name  age
0   Judy   12
1    Bob   13
2    Tom   11
3  Linda   14
"""

data11=data1[data1["age"]>12 & data1["age"]<15] # 选择年龄大于12，小于15的所有的样本/观测

data11.to_csv("2.csv",index=False,header=True) # 将数据集写入csv文件中

# 数据表预处理

data1.rename(columns={"age":"age1"},inplace=True) # 修改列名，age改为age1

data1["age"]=data1["age"].astype("float") # 将age列的数据替换由整型改为浮点型

data1["name"]=data1["name"].replace("Tom","Tony") # 将name中Tom改为Tony

data1=data1.sort_values(by=["age"],ascending=True) # 对数据表中age列作升序处理，ascending为False则为降序

data1=data1.drop(labels="age",axis=1,inplace=True) # 删除age列，axis为0则为删除行

data1=data1.dropna(axis=0) # 删除带有NaN空值的行，axis=1则为删除列

data1=data1.fillna(0) # 用指定数值0填充NaN空值

# 数据表操作

data1=pd.concat([data1,data2],axis=0) 
"""
纵向拼接两个表格，axis=1则为横向拼接
    id   name  age
0  NaN   Judy   12
1  2.0    Bob   13
2  3.0    Tom   11
3  4.0  Linda   14
4  5.0  Cindy   13
0   06   Alex   12
1   07    Lio   10
2   08   Peng   12
3   09   Eksi   15
4   10    Shu   11
"""

data1=data1[["age"]].apply(lambda value : value.sum(), axis=1) 
"""
对数据表中age列所有数值按行求和,axis=0则为按列操作
0    12
1    13
2    11
3    14
4    13
dtype: int64
"""

data1=pd.melt(data1,id_vars="id",var_name="variable",value_name="value")
"""
以变量id为标识，将其余两个变量展开到一列中，如下：

  id variable value
[[nan 'name' 'Judy'] 
 [2.0 'name' 'Bob']  
 [3.0 'name' 'Tom']  
 [4.0 'name' 'Linda']
 [5.0 'name' 'Cindy']
 [nan 'age' 12]      
 [2.0 'age' 13]      
 [3.0 'age' 11]      
 [4.0 'age' 14]      
 [5.0 'age' 13]]

"""

data2=pd.DataFrame(pd.read_csv("2.csv",header=0,encoding="utf8"))
data=data2.groupby("year").sum() 
"""
类似于excel中的分类汇总功能，根据变量year的变量值对value进行求和运算，并自动排除字符串变量id
      value
year
2010     52
2011     35

"""
data=data2.groupby("id").count() 
"""
根据变量id的变量值对year, value进行计数
    year  value
id
A      3      3
B      2      2
C      2      2
"""
data=data2.groupby(["id","year"]).count() 
"""
根据变量id和year的变量值对value进行交叉计数
         value
id year
A  2010      2
   2011      1
B  2010      1
   2011      1
C  2010      1
   2011      1
"""

