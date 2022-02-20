# 排序：python的内置排序函数sort底层算法为Timsort，是归并排序和插入排序的结合，运行时间复杂度为O(nlon(n))，空间复杂度为O(n)
# 以下排序算法默认从小到大


# （1）冒泡排序：逐轮遍历，每轮遍历中通过比较不断交换，先排好最大/小的，再排好第二大/小的……
A=[1,5,3,7,9,4,2,4,10,3,13,43,32,21,6]
def sort1(List):
    exchange=True
    length=len(List)-1
    while length>0 and exchange:
        exchange=False
        for i in range(length):
            if List[i]>List[i+1]:
                exchange=True
                List[i],List[i+1]=List[i+1],List[i]
        length=length-1

    return List
print(sort1(A))

# 最坏时间复杂度：O(n^2)  平均时间复杂度：O(n^2) 适用于小列表

# （2）选择排序：逐轮遍历，每轮遍历中寻找最大/小的，第二大/小的……并通过一次交换将其放在正确位置上
A=[1,5,3,7,9,4,2,4,10,3,13,43,32,21,6]
def sort2(List):
    for i in range(len(List)-1,0,-1):
        position_max=i
        for j in range(i-1,-1,-1):
            if List[position_max]<List[j]:
                position_max=j
        List[i],List[position_max]=List[position_max],List[i]

    return List
print(sort2(A))

# 最坏时间复杂度：O(n^2)  平均时间复杂度：O(n^2) 适用于小列表

# （3）插入排序：采用类似动态规划的思想，以第一个元素作为有序子列表，不断扩充这个子列表直至与原列表等长
A=[1,5,3,7,9,4,2,4,10,3,13,43,32,21,6]
def sort3(List):
    for i in range(1,len(List)):
        value=List[i] # 子列表外第一个值
        index=i-1
        while index>=0 and List[index]>value: # 类似于保序插入思想
            List[index+1]=List[index]
            index=index-1
        List[index+1]=value
    
    return A
print(sort3(A))

# 最坏时间复杂度：O(n^2)  平均时间复杂度：O(n^2) 适用于列表大部分元素已排好序

# （4）希尔排序：插入排序的改良版，先按照一定步长（如列表长度的1/2）尽可能将原列表等分为多个子列表，对每个子列表进行插入排序；
#  接着按照列表长度的1/4为步长进行插入排序……以此直到为整个列表进行插入排序。
A=[1,5,3,7,9,4,2,4,10,3,13,43,32,21,6]
def sub_sort3(List,start,step):
    for i in range(start+step,len(List),step):
        value=List[i]
        index=i-step
        while index>=start and List[index]>value:
            List[index+step]=List[index]
            index=index-step
        List[index+step]=value
    return List

def sort4(List):
    step=len(List)//2 # 模运算，求子列表个数
    while step>0:
        for j in range(step):
            sub_sort3(List,j,step)
        step=step//2
    
    return List
print(sort4(A))

# 最坏时间复杂度：O(nlog(n))~O(n^2)  平均时间复杂度：O(nlog(n))

# （5）快速排序：选择列表第一个元素为基准值，通过不断交换，将列表其余元素中大于基准值的放在它右边，小于的放在左边
#  然后在左边和右边的子列表递归实现如上操作……直到排好整个列表
A=[1,5,3,7,9,4,2,4,10,3,13,43,32,21,6]
def sort5(List,left,right):
    if left>right:
        return 
    
    index_left, index_right, base = left, right, List[left]
    while index_left<index_right:
        while index_left<index_right and List[index_right]>=base: # 列表的右端索引号不断减小，当索引值小于基准时，交换
            index_right=index_right-1
        List[index_left]=List[index_right]
        while index_left<index_right and List[index_left]<base: # 列表的左端索引号不断增大，当索引值大于基准时，交换
            index_left=index_left+1
        List[index_right]=List[index_left]
    List[index_left]=base # 当左右端索引号等同时，一次划分结束，此时左侧子列表元素值小于基准值，右侧的的大于基准值
    sort5(List,left,index_left-1) # 左侧子列表快速排序
    sort5(List,index_right+1,right) # 右侧子列表快速排序

    return List
print(sort5(A,0,len(A)-1)) # left一般默认为最小索引号0，right默认为最大索引号len(A)-1

# 最坏时间复杂度：O(n^2)  平均时间复杂度：O(nlog(n))

# （6）归并排序：原理十分简单！采用递归的方法，不断将列表“等长”地二分，直至列表长度为1，然后通过交换令每一层元素有序
#  不断归并，直至返回有序的整个列表！
A=[1,5,3,7,9,4,2,4,10,3,13,43,32,21,6]
def sort6(List):
    if len(List)>1:
        mid=len(List)//2
        List_left=List[:mid]
        List_right=List[mid:]

        sort6(List_left) # 列表不断二分至单元素列表
        sort6(List_right)

        left=0; right=0; index=0
        while left<len(List_left) and right<len(List_right): # 将二分的两个列表排序后归并至原列表内，覆盖原有的元素
            if List_left[left]<List_right[right]:
                List[index]=List_left[left]
                left=left+1
            else:
                List[index]=List_right[right]
                right=right+1
            index=index+1
        
        while left<len(List_left):
            List[index]=List_left[left]
            left=left+1
            index=index+1

        while right<len(List_right):
            List[index]=List_right[right]
            right=right+1
            index=index+1

        return List
print(sort6(A))

# 最坏时间复杂度：O(nlog(n))  平均时间复杂度：O(nlog(n))

# 其余诸如堆排序，基数排序等等不作更多介绍