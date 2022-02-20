# 二叉搜索树（二叉查找树/二叉排序树）
"""
已知有序列表的二分查找法时间复杂度低(O(logn))，但对于普通的插入删除操作来说时间复杂度高(O(n))；链表的插入删除操作时间复杂度低，
(O(1))，但查找时间复杂度低(O(n))。鉴于此，我们综合有序列表和链表的优点，创造出二叉搜索树/二叉查找树，平均情况下，无论对于查找
操作，(O(logn))还是插入删除操作，(O(logn))其平均时间复杂度都比较低。特别地，当二叉搜索树为平衡树时，最糟糕时间复杂度为(O(logn))

二叉搜索树首先是二叉树结构，但它有其自身特性：对于二叉搜索树的任一非空左子树，其所有节点值均小于该左子树的父结点值，
同理对于任一非空右子树，其所有节点值均大于该右子树的父结点值。不难发现，二叉搜索树中是不允许相同节点值同时存在的，所以
某种意义上，二叉搜索树的节点可用于保存键值对，只对其键进行二叉搜索，本文为简化，故不讨论键。
"""
class Binary_Search_Tree: # 二叉搜索树
    def __init__(self, rootnode):
        self.rootnode=rootnode # 根节点
        self.left_tree=None
        self.right_tree=None
    
    def find(self, data): # 查找某一结点值是否存在
        if self.rootnode==data: 
            return True
        else:
            if self.rootnode>data and self.left_tree:
                return self.left_tree.find(data)
            elif self.rootnode<data and self.right_tree:
                return self.right_tree.find(data)
            else:
                return None

    def find_min(self): # 查找当前节点所对应二叉搜索树的最小节点值（易知其一定位于叶子节点上）
        if self.left_tree:
            return self.left_tree.find_min()
        else:
            return self.rootnode
    
    def find_max(self): # 查找当前节点所对应二叉搜索树的最大节点值
        if self.right_tree:
            return self.right_tree.find_max()
        else:
            return self.rootnode

    def add_tree(self, newnode): # 遵循规则插入新的子树（该插入方法通俗易懂，但会做重复查找操作，故时间复杂度较高）
        if not self.find(newnode):
            if newnode<self.rootnode:
                if self.left_tree:
                    return self.left_tree.add_tree(newnode)
                else:
                    tree=Binary_Search_Tree(newnode)
                    self.left_tree=tree
            else:
                if self.right_tree:
                    return self.right_tree.add_tree(newnode)
                else:
                    tree=Binary_Search_Tree(newnode)
                    self.right_tree=tree

    def del_tree(self,data): # 仅删除该节点值所对应的节点，并保持二叉搜索树的特性
        if self.find(data):
            if self.rootnode<data:
                self.right_tree=self.right_tree.del_tree(data)
            elif self.rootnode>data:
                self.left_tree=self.left_tree.del_tree(data)
            else: # 若找到了对应值的节点，下面进行删除操作
                if self.left_tree and self.right_tree: # 若左子树和右子树均存在
                    min_node=self.right_tree.find_min()
                    self.rootnode=min_node # 取右子树中最小节点值来替代当前节点
                    self.right_tree=self.right_tree.del_tree(min_node) # 删除替代前的右子树的最小节点
                    return self
                else: # 若左子树和右子树不全存在
                    if self.left_tree:
                        return self.left_tree
                    else:
                        return self.right_tree

    def travel(self): # 采用先序遍历方法
        if self.rootnode:
            print(self.rootnode, end=" ")
            if self.left_tree:
                self.left_tree.travel()
            if self.right_tree:
                self.right_tree.travel()

Tree=Binary_Search_Tree(7)
Tree.add_tree(5)
Tree.add_tree(8)
Tree.add_tree(4)
Tree.add_tree(6)
Tree.add_tree(3)
Tree.travel()
"""
二叉搜索树结构如下：
                    7
                   / \
                  5   8
                 / \
                4   6
               /
              3
遍历结构如下：7 5 4 3 6 8 
"""


# 图的最短路径：只考虑无向图，用列表保存已经访问的顶点
# 【1】等权图求最短路径
from operator import countOf
import numpy as np
class Graph:  # 沿用数据结构中图的邻接矩阵表示法
    def __init__(self,n):
        self.List_vertex=[] 
        self.num_vertex=n 
        self.matrix=np.full((n ,n),0) 

    def add_vertex(self,new_vertex): # 载入顶点
        self.List_vertex.append(new_vertex)
        return self.List_vertex

    def add_edge(self,index1,index2,weight=1): # 载入边（输入两个顶点在列表中的索引号）
        if index1==index2:
            print("error!")
        else:
            self.matrix[index1][index2]=weight
            self.matrix[index2][index1]=weight
            return self.matrix
    
    def show_matrix(self): # 展示整个邻接矩阵
        for index in range(self.num_vertex):
            print(self.List_vertex[index]+":", end="")
            print(self.matrix[index])

List=["A","B","C","D","E","F","G"] 
graph1=Graph(len(List)) 
for i in range(len(List)): # 载入顶点
    graph1.add_vertex(List[i]) 
graph1.add_edge(0,1) # 载入边
graph1.add_edge(0,2)
graph1.add_edge(1,2)
graph1.add_edge(1,3)
graph1.add_edge(1,4)
graph1.add_edge(2,4)
graph1.add_edge(3,5)
graph1.add_edge(4,5)
graph1.add_edge(4,6)
graph1.add_edge(5,6)

"""
图的结构如下：
           
           B --- D -- F
         / | \      / |
        A  |  \    /  |
         \ |   \  /   |
           C --- E -- G
"""
# 现在，要求我们求从点A到点G的最短路径以及路径长度
def Bfs_Path(graph, start_vertex, end_vertex): # 广度优先遍历求解，输入图，起始顶点，目标顶点

    def neighbor_vertex(vertex): # 由某一顶点得到其邻接顶点
        List1=[]; List2=[]; index=None
        for i in range(graph.num_vertex):
            if graph.List_vertex[i]==vertex:
                index=i
                break

        for j in range(graph.num_vertex):
            if graph.matrix[index][j]==1:
                List1.append(j)

        for index in List1:
            List2.append(graph.List_vertex[index])

        return List2

    Visited=[start_vertex] # 保存已经遍历的顶点
    Distance={start_vertex:0} # 保存起始顶点到所有顶点的最短距离（边总数）
    Queue=[[start_vertex]] # 保存从起始顶点开始，逐层的邻接顶点，其所在层的层数就等于该顶点到起始顶点的距离
    count=0 # 计算距离
    while Queue:
        Sub_Queue=Queue.pop(0) # 获取某一层的顶点列表
        sub_queue=[] # 用于保存下一层的顶点
        count=count+1 
        for i in range(len(Sub_Queue)): # 遍历某一层的所有顶点，求出它们全部的邻接顶点并放入sub_queue中
            cur_vertex=Sub_Queue[i]
            for vertex in neighbor_vertex(cur_vertex):
                if vertex not in Visited:
                    Visited.append(vertex)
                    Distance[vertex]=count
                    sub_queue.append(vertex)
        if sub_queue:
            Queue.append(sub_queue) # 若下一层顶点列表不为空，则进入队列
        else:
            break # 若下一层顶点列表为空，说明所有顶点都已经遍历，则停止循环
    
    cur_vertex=end_vertex
    Path=[cur_vertex] # 利用回溯法，求得从目标顶点到起始顶点的路径
    while Distance[cur_vertex]>0:
        for vertex in neighbor_vertex(cur_vertex):
            if Distance[vertex]==Distance[cur_vertex]-1:
                cur_vertex=vertex
                Path.append(cur_vertex)
                break
    Path.reverse()

    print("distance = "+str(Distance[end_vertex])+"\n"+"path = ",end="")  
    print(Path)    
Bfs_Path(graph1,"A","G")
"""
结果为
distance = 3
path = ['A', 'C', 'E', 'G']
"""

def Dfs_Path(graph, start_vertex, end_vertex): # 深度优先遍历求解，输入图，起始顶点，目标顶点

    def neighbor_vertex(vertex): # 由某一顶点得到其邻接顶点
        List1=[]; List2=[]; index=None
        for i in range(graph.num_vertex):
            if graph.List_vertex[i]==vertex:
                index=i
                break

        for j in range(graph.num_vertex):
            if graph.matrix[index][j]==1:
                List1.append(j)

        for index in List1:
            List2.append(graph.List_vertex[index])

        return List2

    path=[start_vertex] # 搜索路径

    def  dfs(cur_vertex, path): # 递归方法实现深度优先遍历
        Neighbor_Vertex=neighbor_vertex(cur_vertex)
        for i in range(len(Neighbor_Vertex)):
            vertex=Neighbor_Vertex[i]
            if vertex not in path:
                cur_vertex=vertex
                path.append(cur_vertex)
                if cur_vertex==end_vertex:
                    print("distance = "+str(len(path))+"\n"+"path = ", end="")
                    print(path)
                    path.pop()
                else:
                    dfs(cur_vertex, path)
                    path.pop()

    dfs(start_vertex, path)

Dfs_Path(graph1,"A","G")
"""
结果为
distance = 5
path = ['A', 'B', 'C', 'E', 'F', 'G']
distance = 4
path = ['A', 'B', 'C', 'E', 'G']
distance = 5
path = ['A', 'B', 'D', 'F', 'E', 'G']
distance = 4
path = ['A', 'B', 'D', 'F', 'G']
distance = 4
path = ['A', 'B', 'E', 'F', 'G']
distance = 3
path = ['A', 'B', 'E', 'G']
distance = 6
path = ['A', 'C', 'B', 'D', 'F', 'E', 'G']
distance = 5
path = ['A', 'C', 'B', 'D', 'F', 'G']
distance = 5
path = ['A', 'C', 'B', 'E', 'F', 'G']
distance = 4
path = ['A', 'C', 'B', 'E', 'G']
distance = 6
path = ['A', 'C', 'E', 'B', 'D', 'F', 'G']
distance = 4
path = ['A', 'C', 'E', 'F', 'G']
distance = 3
path = ['A', 'C', 'E', 'G']
"""

"""
当边权重相同时，推荐用广度优先遍历方法来解决图的最小路径问题，其优点在于利用层次遍历的特点，可以明确每一个顶点到起始点的
最短距离。进而由局部最优实现全局最优（动态规划），最终仅通过一次完整的遍历即可搜索出最短路径。

而深度优先遍历方法在寻找起始点到终点的某一条可行路径时效果更突出（如迷宫，棋盘周游），因为无需遍历所有的顶点即可快速找到
该路径。在本题中用于解决最短路径问题并不合适。
"""


# 【2】非等权图(权值非负)求最短路径：Dijkstra算法
"""
看似从无权图变成了带权图，变化不大，但实际上并非如此，理由如下：
【1】在无权图中，由于边的权值相同，因此严格遵循某一顶点到其邻接顶点的直接路径距离，小于其他任何间接路径的距离，即
“绕路一定更远”。所以某种意义上，同一层级（到起始顶点距离相同）上顶点之间的连接与否并不影响最短路径
【2】相反地，在有权图中，由于各边权值不一定相同，因此存在“绕路更近”的情况，图的层级这一概念也就被取消了，从而易知
，我们无法用一般意义上的广度优先遍历方法解决最小路径问题。

对于上述情况，可以用Dijkstra算法来求最小路径。Dijkstra算法在广度优先遍历的基础上，引入贪婪算法的思想：每次尽可能选择
离起始顶点近的顶点（不重复），通过遍历其邻接顶点的信息来更新起始顶点到各顶点的距离。不难得知，当所有顶点都更新一次后，
我们一定可以得到起始顶点到目标顶点的最短距离！
"""
List=["A","B","C","D","E","F","G"] 
graph2=Graph(len(List)) 
for i in range(len(List)): # 载入顶点
    graph2.add_vertex(List[i]) 
graph2.add_edge(0,1,3) # 载入边
graph2.add_edge(0,2,1)
graph2.add_edge(1,2,2)
graph2.add_edge(1,3,4)
graph2.add_edge(1,4,2)
graph2.add_edge(2,4,5)
graph2.add_edge(3,5,3)
graph2.add_edge(4,5,2)
graph2.add_edge(4,6,5)
graph2.add_edge(5,6,2)
"""
图的结构如下：
           
           B --- D -- F
         / | \      / |
        A  |  \    /  |
         \ |   \  /   |
           C --- E -- G
权重：
A<->B(3) A<->C(1) 
B<->C(2) B<->D(4) B<->E(2)
C<->E(5)
D<->F(3)
E<->F(2) E<->G(5)
F<->G(2)
"""
def Dijkstra(graph, start_vertex, end_vertex): # 深度优先遍历求解，输入图，起始顶点，目标顶点

    def get_index(vertex):
        index=None
        for i in range(graph.num_vertex):
            if graph.List_vertex[i]==vertex:
                index=i
                break
        return index

    def neighbor_vertex(vertex): # 由某一顶点得到其邻接顶点
        List1=[]; List2=[]; index=get_index(vertex)

        for j in range(graph.num_vertex):
            if graph.matrix[index][j]>0:
                List1.append(j)

        for index in List1:
            List2.append(graph.List_vertex[index])

        return List2

    def update_vertex(Distance, Update): # 贪婪策略，从Distance中依次不重复地选择值（距离）较小的键（顶点）
        min_distance=float("inf"); min_vertex=None
        for vertex in Distance:
            if (Distance[vertex]<min_distance) and (vertex not in Update):
                min_distance=Distance[vertex]
                min_vertex=vertex
        return min_vertex

    Update=[] # 保存已经更新过邻接顶点信息的顶点
    Parent={start_vertex:None} # 保存被更新后距离变小的顶点的“父顶点”（邻接顶点）
    Distance={} # 保存各顶点到起始顶点的距离
    for vertex in graph.List_vertex: # 初始化各顶点到起始顶点的距离，除起始顶点自身为0外，其余默认为无穷大
        if vertex!=start_vertex:
            Distance[vertex]=float("inf")
        else:
            Distance[vertex]=0

    while len(Update)<graph.num_vertex: # 若贪婪策略尚未实施完成，则继续选择顶点来更新其邻接顶点的距离信息
        min_vertex=update_vertex(Distance, Update)
        Update.append(min_vertex)

        for vertex in neighbor_vertex(min_vertex):
            index1=get_index(min_vertex); index2=get_index(vertex)
            new_distance=Distance[min_vertex] + graph.matrix[index1][index2] # 更新后邻接顶点的距离
            if new_distance<Distance[vertex]: # 如果更新后的距离小于更新前的距离，则更新该距离，以及它的“父顶点”
                Distance[vertex]=new_distance
                Parent[vertex]=min_vertex
    
    path=[end_vertex]
    cur_vertex=end_vertex
    while cur_vertex!=start_vertex: # 利用回溯法，求得从目标顶点到起始顶点的路径
        cur_vertex=Parent[cur_vertex]
        path.append(cur_vertex)
    path.reverse()

    print("distance= "+str(Distance[end_vertex])+"\n"+"path= ",end="")
    print(path)

Dijkstra(graph2, "A", "G")

"""
结果为
distance= 9
path= ['A', 'B', 'E', 'F', 'G']
"""

"""
需要注意的是，Dijkstra算法无法处理边权值为负的图，举例如下：
    A --- B  
     \   /
       C
若各边权值非负，则无论顶点之间是单向边还是双向边，我们在运用Dijkstra算法时，根据贪婪策略，我们依次选择离起始顶点最近的
顶点更新其邻接顶点的信息，可知次序靠后的顶点更新距离信息后，始终不会影响次序靠前的顶点离起始顶点的距离！这一点直接
保证了贪婪策略的稳定。

现在，若规定A->B(3) A->C(2) B->C(-2)，以A为起始顶点，则Distance={A:0, B:3, C:2}，我们依次更新C，B的邻接顶点的信息，
最终得Distance={A:0, B:3, C:1}，此时顶点B的更新使得顶点C到A更近了，这破坏了贪婪策略的稳定性，最终有可能影响结果的
正确性。

极端的，当A,B,C均为双向边时，三个顶点构成一个闭环，那么会发现无论更新C，还是更新B，它们到A都更近了！这就陷入了一个无解的
死循环！
"""