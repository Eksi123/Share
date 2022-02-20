# 物理结构（存储结构）

# （1）顺序结构：如常见的列表
# （2）链式结构：如单向链表，如下：

# 定义节点（元素）类
class Node:
    def __init__(self,data,next=None):
        self.data=data
        self.next=next

# 定义链表类
class LinkList:
    def __init__(self):
        self.head=None        # 头指针
    
    def len(self):            # 求链表长度
        cur=self.head; length=0
        while cur!=None:
            length=length+1
            cur=cur.next
        return length

    def printf(self):            # 遍历链表
        node=A.head
        for i in range(A.len()):     
            print(node.data)
            node=node.next

    def append(self,data):   # 在链表末尾添加节点
        node=Node(data)
        if self.len()==0:
            self.head=node
        else:
            cur=self.head
            pre=None
            while cur!=None:
                pre=cur
                cur=cur.next
            pre.next=node

    def insert(self,sp,data):  # 在指定位置插入新节点
        node=Node(data)
        if sp<=0:
            node.next=self.head
            self.head=node
        elif sp>=self.len():
            self.append(data)
        else:
            cur=self.head
            for i in range(sp-1):
                cur=cur.next
            node.next=cur.next
            cur.next=node

    def pop(self,sp=None):     # 删除指定位置的节点（若sp不填，则默认删除末尾节点）
        if sp==None:
            if self.len()>1:
                cur=self.head
                pre=None
                length=self.len()
                for i in range(length-1):
                    pre=cur
                    cur=cur.next 
                pre.next=None    
            else:
                self.head=None
        else:
            if self.len()-1<=sp:
                self.pop()
            else:
                cur=self.head
                pre=None
                for i in range(sp):
                    pre=cur
                    cur=cur.next
                pre.next=cur.next
                cur.next=None    
    
    def insert_1(node,data):   # 在指点节点node后插入一个新节点
        newnode=Node(data)
        node1=node.next
        node.next=newnode
        newnode.next=node1

    def pop_1(node):          # 删除指定节点node后的那个节点
        node1=node.next
        node2=node1.next
        node.next=node2
        node1.next=None

    def found(self,data): # 判断链表中是否含有数据data的节点，有就返回True
        cur=self.head
        while cur!=None:
            if cur.data==data:
                return True
            cur=cur.next
        return False


A=LinkList()         # 创建链表
for i in range(5):
    A.append(i)

0 1 2 3 4

A.insert(2,7)        # 在第三个节点的位置插入新节点（数据为7）

0 1 7 2 3 4

A.pop(1)             # 删除第二个节点

0 7 2 3 4

A.printf()           # 遍历并输出链表各节点数据
print(A.len())       # 输出链表长度

5

print(A.found(6))    # 判断链表内是否含有某节点数据为6

False

# （3）散列结构：如散列表/哈希表，又说字典，其存储原理简单模拟如下,实际运用时无需定义：
class HashTable:
    def __init__(self,size):
        self.size=size # 自定义散列表/哈希表长度，真实的哈希表长度可动态调整
        self.keys=[None]*self.size # 存放键key
        self.values=[None]*self.size # 存放值value
    
    def printf(self): # 打印散列表/哈希表
        length=len(self.keys)
        for i in range(length):
            if self.keys[i]!=None:
                print(self.keys[i]+":"+str(self.values[i]))

    def Hash_Function(self,key): # 散列函数/哈希函数
        sum=0; length=len(key)
        for i in range(length): # 求取字符串各字符Unicode值之和
            sum=sum+ord(key[i])
        hash=sum%self.size # 散列函数-取余函数，返回哈希值
        return hash

    def Rehash(self,hash): # 重新求哈希值
        new_hash=(hash+1)%self.size 
        return new_hash

    def set(self,key,value): # 可实现添加，修改，删除键值对的功能
        hash=self.Hash_Function(key)
        if self.keys[hash]==None: # 如果哈希值对应键为空，则添加该键值对
            self.keys[hash]=key
            self.values[hash]=value
        else:
            if self.keys[hash]==key: # 如果哈希值对应键不为空，且和输入键相同，则修改该键对应的值
                if value==None:
                    self.keys[hash]=None # 上述情况下，如果输入值为空，则删除该键值对
                else:
                    self.values[hash]=value
            else:  # 哈希值对应键不为空，但与输入键不同，此时重新求哈希值，直到满足上述两种情况任一种
                new_hash=self.Rehash(hash)
                while self.keys[new_hash]!=None and self.keys[new_hash]!=key:
                    new_hash=self.Rehash(new_hash)
                
                if self.keys[new_hash]==None: # 重复上述操作
                    self.keys[new_hash]=key
                    self.values[new_hash]=value
                else:
                    if self.keys[new_hash]==key: 
                        if value==None:
                            self.keys[new_hash]=None 
                        else:
                            self.values[new_hash]=value

    def get(self,key): # 输入键，索引值
        hash=self.Hash_Function(key)
        if self.keys[hash]==None: # 如果该键不存在，返回空值
            return None
        else:
            if self.keys[hash]==key:
                return self.values[hash]
            else:
                new_hash=self.Rehash(hash) # 操作类似get函数，不断刷新哈希值来检索
                while self.keys[new_hash]!=key :
                    new_hash=self.Rehash(new_hash)
                if new_hash!=hash:
                    return self.values[new_hash]
                else:
                    return None

A=HashTable(5)
A.set("a",2); A.set("b",4); A.set("c",8)
A.printf()
a:2
b:4
c:8

A.set("a",1)
A.printf()
a:1
b:4
c:8

A.set("c",None)
A.printf()
a:1
b:4

A.get("a")
1


#----------------------------------------------------#

# 逻辑结构（可由上述物理结构组成）

# （1）无序集合，如字典
A={"a":1,"b":2,"c":3}
A["d"]=4; print(A) # 进入字典
del A["d"]; print(A)  # 移出字典

#（2）线性表，如列表，栈，队列
#（2.1）列表（可在任意位置进入或退出）
A=[1,2,3]
A.insert(2,4); print(A)
A.pop(2); print(A)
# （2.2）栈（先进后出，后进先出）
A=[1,2,3]
A.append(4); print(A) #顶部入栈
A.pop(); print(A) #顶部出栈
A.insert(0,4); print(A) #底部入栈
A.pop(0); print(A) #底部出栈
# （2.3）队列（先进先出，后进后出）
A=[1,2,3] 
A.append(4); print(A) #入列（自右向左）
A.pop(0); print(A) #出列
A.insert(0,4); print(A) #入列（自左向右）
A.pop(); print(A) #出列
# （2.4）双端队列（进出方向不限）
A=[1,2,3]
A.append(4); print(A) #入列
A.pop() #出列（先进先出）
print(A) 
A.pop(0) #出列（先进后出）
print(A) 

# （3）非线性表，如树，图
#  (3.1)树，以二叉树为例模拟树的构建和遍历：
"""
一颗树由节点和边组成，边是节点之间的联系，是一种虚指；节点包括根节点（只有出边），普通节点和叶子节点（只有入边）。

根据节点之间上下级联系，又可以分为父节点和子节点，父节点的出边指向子节点，，一个父节点和其子节点，以及节点间的边构成一颗子树
当然父节点，子节点和子树都是相对的。

最后，某一节点的层数等同于从根节点到该节点的边数，所有叶子节点的最大层数即为该树的高度。

二叉树是最常见也最常用的树结构，根据其外观可分为：【1】满二叉树（每个叶子节点均位于最深层，且其父节点一定有左，右两个子节点）
【2】完全二叉树：除最深层外，其余层所构成的二叉树为满二叉树，最深层叶子节点从左往右依次排列，可以不满）。其余如非满二叉树
，非完全二叉树同理，下面我们来构建一般意义上的二叉树，并对其实现遍历
"""
class BinaryTree:
    def __init__(self,rootnode): 
        self.rootnode=rootnode # 树的根节点/父节点
        self.left_tree=None # 左子树
        self.right_tree=None # 右子树
    
    def find(self,data): # 查找某一结点值是否存在
        if self.rootnode==data:
            return True
        else:
            if self.left_tree:
                return self.left_tree.find(data)
            if self.right_tree:
                return self.right_tree.find(data)

    def add_left_tree(self,newnode): # 添加左子树
        self.left_tree=BinaryTree(newnode)
        return self.left_tree
        
    def add_right_tree(self,newnode): # 添加右子树
        self.right_tree=BinaryTree(newnode)
        return self.right_tree

    def del_tree(self): # 删除所在节点的整棵树
        self.rootnode=None
        self.left_tree=None
        self.right_tree=None
        return self.rootnode, self.left_tree, self.right_tree

    def set_rootnode(self,newnode): # 不添加新子树，仅修改父节点值
        self.rootnode=newnode
        return self.rootnode

Tree=BinaryTree(4) # 通过定义根节点来定义一棵二叉树
node1=Tree.add_left_tree(3) # 树的节点，但相对来看也是一颗子树
node2=node1.add_left_tree(1)
node3=node1.add_right_tree(2)
node4=Tree.add_right_tree(5)
"""
二叉树构造如下：
                       4
                      / \
                     3   5
                    / \
                   1   2
"""

def breadth_travel(tree): # 广度优先遍历，也叫层次遍历，从根节点开始，逐层从左往右遍历
    if tree:
        List=[tree] 
        while List:
            cur_tree=List.pop(0) # 创建一个队列，左进左出
            print(cur_tree.rootnode, end=" ")
            if cur_tree.left_tree:
                List.append(cur_tree.left_tree)
            if cur_tree.right_tree:
                List.append(cur_tree.right_tree)

def pre_order_travel(tree): # 深度优先遍历之一：先序遍历，从根节点开始，先递归地遍历左子树，再遍历右子树
    if tree.rootnode:
        print(tree.rootnode, end=" ")
        if tree.left_tree:
            pre_order_travel(tree.left_tree)
        if tree.right_tree:
            pre_order_travel(tree.right_tree)

def mid_order_travel(tree): # 深度优先遍历之二：中序遍历，先递归地遍历左子树，再访问根节点，最后递归地遍历右子树
    if tree.rootnode:
        if tree.left_tree:
            mid_order_travel(tree.left_tree)
        print(tree.rootnode, end=" ")
        if tree.right_tree:
            mid_order_travel(tree.right_tree)

def post_order_travel(tree): # 深度优先遍历之三：后序遍历，先递归地遍历左子树，再遍历右子树，最后访问根节点
    if tree.rootnode:
        if tree.left_tree:
            post_order_travel(tree.left_tree)
        if tree.right_tree:
            post_order_travel(tree.right_tree)
        print(tree.rootnode, end=" ")

breadth_travel(Tree)
pre_order_travel(Tree)
mid_order_travel(Tree)
post_order_travel(Tree)

"""
遍历结果如下:
广度优先遍历：4 3 5 2 1 
先序遍历：4 3 2 1 5 
中序遍历：2 1 3 4 5 
后序遍历：2 1 3 5 4 
"""
    
# (3.2) 图
"""
图是一种比树更复杂的数据结构，但其形象的表示方法使其可用于模拟日常生活中的许许多多图形结构，如网络，路径，迷宫等等，进而借助
计算机的强大能力来解决十分困难的实际问题。

与树一样，图同样由节点/顶点和边组成，不同在于节点之间是平等的，不存在”父子关系“以及根节点，叶子节点等等，此外图的边不再是虚指，
它不仅可用于表达两个节点间的联系，还可以表示指向（有向图和无向图）以及节点联系的权重（无权图/等权图，带权图/非等权图）。
最后，有向图可以组成闭环这一几何结构，进而将图分为有环图和无环图。

对于树这一数据结构，我们的侧重点在于用其来补充列表的不足；而对于图来说，我们将完全聚焦于其强大的图形和路径表达能力上。
下面我们来构建一般意义上的图，并对其实现遍历
"""
import numpy as np
class Graph:  # 出于简洁，本例只用邻接矩阵表示法来表示图
    def __init__(self,n):
        self.List_vertex=[] # 用于存放顶点
        self.num_vertex=n # 顶点个数
        self.matrix=np.full((n ,n),0) # 初始化邻接矩阵

    def add_vertex(self,new_vertex): # 载入顶点
        self.List_vertex.append(new_vertex)
        return self.List_vertex

    def add_edge(self,index1,index2): # 载入边（输入两个顶点在列表中的索引号）
        if index1==index2:
            print("error!")
        else:
            self.matrix[index1][index2]=1 # 若其中任一元素为0，则说明两顶点之间存在指向
            self.matrix[index2][index1]=1
            return self.matrix
    
    def show_matrix(self): # 展示整个邻接矩阵
        for index in range(self.num_vertex):
            print(self.List_vertex[index]+":", end="")
            print(self.matrix[index])

    def neighbor_vertex(self,vertex): # 由某一顶点得到其邻接顶点
        List1=[]; List2=[]; index=None
        for i in range(self.num_vertex):
            if self.List_vertex[i]==vertex:
                index=i
                break
        for j in range(self.num_vertex):
            if self.matrix[index][j]==1:
                List1.append(j)
        
        for index in List1:
            List2.append(self.List_vertex[index])

        return List2

    def bfs_travel(self,start_vertex): # 广度优先遍历：与二叉树的遍历类似，不同之处在于需要考虑循环遍历的情况
        Visited=[start_vertex] # start_index为指定的遍历起点
        List=[start_vertex]
        while List:
            cur_vertex=List.pop(0) # 队列，先进先出，实现同一层顶点的优先遍历
            print(cur_vertex, end=" ")
            for vertex in self.neighbor_vertex(cur_vertex):
                if vertex not in Visited:
                    Visited.append(vertex)
                    List.append(vertex)

    def dfs_travel(self,start_vertex): # 深度优先遍历：同样与二叉树的深度优先遍历类似，不过不指定前后序，实现方法与上面几乎一致
        Visited=[start_vertex]
        List=[start_vertex]
        while List:
            cur_vertex=List.pop() # 栈，先进后出，实现同一条纵深路径的优先遍历
            print(cur_vertex, end=" ")
            for vertex in self.neighbor_vertex(cur_vertex):
                if vertex not in Visited:
                    Visited.append(vertex)
                    List.append(vertex)
    

List=["A","B","C","D","E","F"] 
graph=Graph(len(List)) # 生成图，此时没有顶点没有边，需载入
for i in range(len(List)): # 载入顶点
    graph.add_vertex(List[i]) 
graph.add_edge(1,2) # 载入边
graph.add_edge(2,4)
graph.add_edge(3,4)
graph.add_edge(1,0)
graph.add_edge(2,0)
graph.add_edge(2,5)
graph.add_edge(4,5)

graph.show_matrix() # 展示邻接矩阵

graph.dfs_travel("F")
graph.bfs_travel("F")
"""
邻接矩阵如下:
A:[0 1 1 0 0 0]
B:[1 0 1 0 0 0]
C:[1 1 0 0 1 1]
D:[0 0 0 0 1 0]
E:[0 0 1 1 0 1]
F:[0 0 1 0 1 0]

实际情况中图 如下：
       A -- B    D
        \  /    /
          C -- E
           \  /
             F

遍历结果如下：
深度优先遍历：F E D C B A
广度优先遍历：F C E A B D
"""
    

        

