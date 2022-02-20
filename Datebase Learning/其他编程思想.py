# 贪婪算法
#背包问题：
"""
一小偷携带负荷20的背包来珠宝店偷东西，现列出物品重量和价值以供考虑:
(1,3),(2,4),(3,5),(5,6),(6,8),(8,10),(12,15),(13,20)，问小偷采取如下何种贪婪策略，可偷取得最大价值:
【1】在不超过负荷情况下，每次价值最大【2】在不超过负荷情况下，每次平均价值最大
"""
max=20
Goods_Value=[[1,3],[2,4],[3,5],[5,6],[6,8],[8,10],[12,15],[13,20]]

def f1(max, Goods_Value):  # 策略1：每次偷取价值最大
    sum_value=0; sum_goods=0
    for i in range(len(Goods_Value)-1,-1,-1):
        sum_value=sum_value+Goods_Value[i][1]
        sum_goods=sum_goods+Goods_Value[i][0]
        if sum_goods>max:
            sum_value=sum_value-Goods_Value[i][1]
            sum_goods=sum_goods-Goods_Value[i][0]
        else:
            print("("+str(Goods_Value[i][0])+","+str(Goods_Value[i][1])+")")
    return sum_goods,sum_value

def f2(max, Goods_Value):  # 策略2：每次偷取平均价值最大
    sum_value=0; sum_goods=0
    for i in range(len(Goods_Value)):
        Goods_Value[i].append(Goods_Value[i][1]/Goods_Value[i][0])
    Goods_Value.sort(key=(lambda x:x[2]))  # 对二维列表按照关键字x[2]倒序排序
    for i in range(len(Goods_Value)-1,-1,-1):
        sum_value=sum_value+Goods_Value[i][1]
        sum_goods=sum_goods+Goods_Value[i][0]
        if sum_goods>max:
            sum_value=sum_value-Goods_Value[i][1]
            sum_goods=sum_goods-Goods_Value[i][0]
        else:
            print("("+str(Goods_Value[i][0])+","+str(Goods_Value[i][1])+")")
    return sum_goods,sum_value

print(f1(max,Goods_Value))

(13,20)
(6,8)
(1,3)
(sum_goods=20, sum_value=31)

print(f2(max,Goods_Value))

(1,3)
(2,4)
(3,5)
(13,20)
(sum_goods=19, sum_value=32)

#----------------------------------#
# 动态规划算法
"""
同样是背包问题，在上题中我们还可以用动态规划算法来求解，定义动态规划矩阵，横行代表所考虑的物品个数，从物品(1,3)直到全部物品
；纵列代表背包负荷的变化，从负荷为1到负荷为20。以i行j列为例，此时我们需要的问题是:在考虑前i个物品且负荷为j的情况下，可偷取
的最大价值。不难得知当G[i][0]>=j时，A[i][j]=max(A[i-1][j], G[i][1]+A[i-1][j-G[i][0]]) 
"""

max=20
Goods_Value=[[1,3],[2,4],[3,5],[5,6],[6,8],[8,10],[12,15],[13,20]]
def f(max,Goods_Value):
    A=[]; length=len(Goods_Value)
    for i in range(length+1):
        A.append([])
        for j in range(max+1):
            A[i].append(0)

    for i in range(1,length+1):
        for j in range(1,max+1):
            A[i][j]=A[i-1][j]
            if Goods_Value[i-1][0]<=j and A[i-1][j]<Goods_Value[i-1][1]+A[i-1][j-Goods_Value[i-1][0]]:
                A[i][j]=Goods_Value[i-1][1]+A[i-1][j-Goods_Value[i-1][0]]
    
    for i in range(length,-1,-1): # 根据动态规划矩阵A的结果，我们从A[length][max]一直回溯至A[1][1]，判断G[i-1]是否被偷取
        if A[i][max]==Goods_Value[i-1][1]+A[i-1][j-Goods_Value[i-1][0]]:
            max=max-Goods_Value[i-1][0]
            print(Goods_Value[i-1])

f(max,Goods_Value)

[13, 20]
[3, 5]
[2, 4]
[1, 3]         


#----------------------------------#
# 分治算法

# 二分查找问题
A2=[1,2,3,5,5,8,9,12,23,43,45,64]; item=13

def find_21(List,item): # 非递归
    flag=False; first=0; last=len(List)-1
    while first<=last and flag==False:
        mid=(first+last)//2  # 模运算
        if List[mid]==item:
            flag=True
        else:
            if List[mid]<item:
                first=first+1
            else:
                last=last-1
    return flag
print(find_21(A2,item))

#----------------------------------#
# 回溯算法

# 马的遍历问题：nxm的棋盘，选定棋盘上一点(x,y)并按照“马”的规则走，规划出该点不重复地走过棋盘上任一位置的路线？

class Check:    # 用于判断走位的正确性
    def __init__(self,n,m,x,y):
        self.n=n
        self.m=m
        self.x=x
        self.y=y
        self.A=[]  
        for i in range(self.n):
            self.A.append([])
            for j in range(self.m):
                self.A[i].append(0)

    def check(self):
        if (self.x<1 or self.x>self.m) or (self.y<1 or self.y>self.n) or self.A[self.x-1][self.y-1]!=0:
            return False
        return True


class Run(Check):     # 用于递归走位
    def __init__(self, n, m, x, y):
        super().__init__(n, m, x, y)
        self.step=None
        self.D_x=[-2,-2,2,2,1,-1,1,-1]
        self.D_y=[1,-1,1,-1,-2,-2,2,2]

    def run(self):
        for i in range(8):
            if self.check():
                self.A[self.x-1][self.y-1]=self.step
                if self.step==self.n*self.m:
                    return self.A
                else:
                    self.step=self.step+1
                    self.run()
                    self.step=self.step-1
                    self.A[self.x-1][self.y-1]=0
            else:
                self.x=self.x-self.D_x[i]; self.y=self.y-self.D_y[i]

class Game(Run):  # 用于初始化棋盘和起始点，并输出路线
    def __init__(self, n, m, x, y):
        super().__init__(n, m, x, y)
        self.A[self.x-1][self.y-1]=1
        self.step=2
        
    def game(self):
        if self.run()!=None:
            for i in range(len(self.A)):
                print(concat(self.A[i],"\t"))
        else:
            print("try again!")

p=Game(5,5,1,1)
p.run()

1       14      3       8       21
4       9       20      13      16
19      2       15      22      7
10      5       24      17      12
25      18      11      6       23