# 用“递归”代替“循环”计算 1+2+3+……+9

# 循环方法
from tkinter import N


sum=0
for i in range(1,10):
    sum=sum+i
# 递归方法：通过执行sum(n-1)使得n不断减小靠近1，在n=1之后，开始朝着n增大的方向返回 sum(n-1)+n的值：1、1+2、1+2+3………
def sum(n):
    if n<=1:  # 满足终止条件，返回sum(1)=1的值至sum(2)
        return 1
    return sum(n-1)+n # 满足终止条件前，一直执行sum(n-1)；在满足条件后，不断返回sum(n) = sum(n-1) + n的值至上一层


# （1）十进制换二进制
def Ten_to_Two(n):
    if n<=1:
        print(int(n),end="")
    else:
        Ten_to_Two((n-n%2)/2)
        print(int(n%2),end="")

Ten_to_Two(18)

10010

# （2）汉诺塔问题:共有n个盘子，求把所有盘子正确地从柱子A移到柱子C的操作？
# 思路：先把前n-1个盘子正确移到柱子B，然后把第n个盘子移到柱子C，再把前n-1个盘子从柱子B移到柱子C
def MoveTower(n,start="A",middle="B",end="C"):
    if n==1:
        print(start+"->"+end)  # 终止条件：只有一个盘子时，停止调用函数，直接将盘子移到目标柱子
    else:
        MoveTower(n-1,start,end,middle) 
        MoveTower(1,start,middle,end)
        MoveTower(n-1,middle,start,end)  
    
MoveTower(5)

A->C
A->B
C->B
A->C
B->A
B->C
A->C
A->B
C->B
C->A
B->A
C->B
A->C
A->B
C->B
A->C
B->A
B->C
A->C
B->A
C->B
C->A
B->A
B->C
A->C
A->B
C->B
A->C
B->A
B->C0
A->C



        



        











