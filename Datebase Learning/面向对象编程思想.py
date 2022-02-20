## 类与对象

class Dog:                                 # 类：狗狗
    def __init__(self,name,sex,age):       # 属性（变量）： 名字、性别、年龄
        self.name=name
        self.sex=sex
        self.age=age
    
    def Introduction(self):                # 方法（函数）：作自我介绍
        print("The dog "+self.name+" is a "+self.sex+" dog,and "+str(self.age)+
        " years old now") 

# 每个类均由属性（变量）和方法（函数）两部分组成，其中特殊如方法_init_是用来初始化或定义类属性
# 注意，参数self仅用来封装和传递类属性，自身并无任何含义

dog1=Dog("Bob","male",3)                    # 类实例化为不同的对象
dog2=Dog("Mary","female",2) 

print(dog1.name,dog1.sex,dog1.age)          # 访问不同对象的属性值
print(dog2.name,dog2.sex,dog2.age) 

dog1.Introduction()                           # 调用不同对象的方法
dog2.Introduction()

# 类的实例化本质上是调用方法_init_并对其中的参数赋值，python中方法_init_不作显示
# 由于类已经实例化为对象了，那么当调用某一对象的方法时，参数self（name、sex、age）已知。python中参数self不作显示

class Dog:                                 
    def __init__(self,name,age):       
        self.name=name
        self.sex="male"                         # 公共属性 sex=“male”
        self.age=age
    
    def Introduction(self):                
        print("The dog "+self.name+" is a "+self.sex+" dog,and "+str(self.age)+
        " years old now") 

dog1=Dog("Bob",3); print(dog1.sex)
dog2=Dog("Mary",2); print(dog2.sex)    # 类的实例化过程中，公共属性无需赋值（即使赋值为female也不作用）
dog2.sex="female"; print(dog2.sex,dog1.sex)     # 对某一对象修改属性sex的值，但不影响其他对象

#也可采用如下方法
#方法1
class Dog:                                 
    def __init__(self,name,age):       
        self.name=name
        self.sex="male"                         
        self.age=age
    
    def Introduction(self):                
        print("The dog "+self.name+" is a "+self.sex+" dog,and "+str(self.age)+
        " years old now") 
    
    def Update_sex(self,new_sex):
        self.sex=new_sex

dog1=Dog("Bob",3); print(dog1.sex)
dog2=Dog("Mary",2); print(dog2.sex) 
dog2.Update_sex("female"); print(dog2.sex,dog1.sex)

#方法2
class Dog:                                 
    def __init__(self,name,age,sex="male"):          # 属性sex若未赋值，则实例化后默认为male  
        self.name=name
        self.sex=sex                        
        self.age=age
    
    def Introduction(self):                
        print("The dog "+self.name+" is a "+self.sex+" dog,and "+str(self.age)+
        " years old now") 

dog1=Dog("Bob",3); print(dog1.sex)
dog2=Dog("Mary",2); print(dog2.sex) 
dog2=Dog("Mary",2,"female"); print(dog2.sex,dog1.sex)


# 类的继承和调用
class Dog:                                
    def __init__(self,name,age):      
        self.name=name
        self.sex=None
        self.age=age
    
    def Introduction(self):              
        print("The dog "+self.name+" is a "+self.sex+" dog,and "+str(self.age)+
        " years old now") 

class Movement:
    def __init__(self,sex,movement="running"):
        self.sex=sex
        self.movement=movement

    def describe(self):
        if self.sex=="male":
            print("Look, he is "+self.movement)
        else:
            print("Look, she is "+self.movement)

class Male_Dog(Dog):                             # 子类Male_Dog继承父类Dog中所有方法(_init_和Introduction)
    def __init__(self, name, age,action):
        super().__init__(name, age)              # 方法super().__init__使子类继承父类中所有非公共属性   
        self.sex="male"                          # 在子类中增加属性sex="male"
        self.action=action
        self.move=Movement(self.sex,action)               # 将类Movement实例化，并作为Male_Dog的一个属性来调用

class Female_Dog(Dog):                             # 同理
    def __init__(self, name, age,action):
        super().__init__(name, age)               
        self.sex="female"                          
        self.action=action
        self.move=Movement(self.sex,action)                  
            
dog=Male_Dog("Tom",4,"sitting")
dog.Introduction()
dog.move.describe()

dog=Female_Dog("Judy",3,"sleeping")
dog.Introduction()
dog.move.describe()


##面向过程与面向对象

def concat(Table,s1,start,end):                                # 列表元素拼接
    s=str(Table[0])
    for i in range(start+1,end+1):
        s=s+s1+str(Table[i])
    return s

#面向过程编程
M=[1]; N=[1]; sp=0; dividend=1
while sp<=999:                                                 # 若小数位小于1000位，则继续运算；反之停止运算
    for j in range(sp,1010):
        if len(N)-1>=j+1: N[j+1]=N[j+1]+(N[j]%dividend)*10     # 等价于N[j+1]=(N[j+1] or 0)+(N[j]%dividend)*10
        else: N.append((N[j]%dividend)*10)

        N[j]=int((N[j]-N[j]%dividend)/dividend)

        if len(M)-1>=j: M[j]=M[j]+N[j]         
        else: M.append(N[j])
    while N[sp]==0:
        sp=sp+1
    dividend=dividend+1

for j in range(1009,-1,-1):                                   # 十进制调整
    M[j-1]=M[j-1]+int((M[j]-M[j]%10)/10)
    M[j]=M[j]%10
print(concat(M,"",0,999))


#面向对象编程
class Division:                                            # 描述数组除法运算
    def __init__(self,Arr,dividend=1):
        self.Arr=Arr
        self.dividend=dividend

    def division(self):
        for j in range(len(self.Arr)-1):
            self.Arr[j+1]=self.Arr[j+1]+(self.Arr[j]%self.dividend)*10
            self.Arr[j]=int((self.Arr[j]-self.Arr[j]%self.dividend)/self.dividend)
        return self.Arr

class Product:                                            # 描述数组乘法运算
    def __init__(self,Arr,exponent):
        self.Arr=Arr
        self.exponent=exponent

    def product(self):
        for j in range(len(self.Arr)):
            self.Arr[j]=self.Arr[j]*self.exponent
        for j in range(len(self.Arr)-1,-1,-1):
            self.Arr[j-1]=self.Arr[j-1]+int((self.Arr[j]-self.Arr[j]%10)/10)
            self.Arr[j]=self.Arr[j]%10
        return self.Arr


class Sum:                                              # 描述数组之间加和运算
    def __init__(self,Arr,Arr1):
        self.Arr=Arr
        self.Arr1=Arr1
    
    def sum(self):
        for j in range(len(self.Arr1)):
            self.Arr1[j]=self.Arr[j]+self.Arr1[j]
        for j in range(len(self.Arr1)-1,-1,-1):
            self.Arr1[j-1]=self.Arr1[j-1]+int((self.Arr1[j]-self.Arr1[j]%10)/10)
            self.Arr1[j]=self.Arr1[j]%10
        return self.Arr1

class Count:                                            # 找出数组元素的非零起始位置
    def __init__(self, Arr):
        self.Arr=Arr
        self.sp=0
    
    def update_sp(self):
        while self.Arr[self.sp]==0 and self.sp<1009:
            self.sp=self.sp+1
        return self.sp 

class Compute:                                          # 计算e的任一指数的1000位小数
    def __init__(self,Arr,Arr1,exponent):
        self.Arr=Arr
        self.Arr1=Arr1
        self.exponent=exponent
        self.dividend=1
        self.sp=Count(self.Arr).update_sp()

    def compute(self):
        while self.sp<=999:
            self.Arr=Product(self.Arr, self.exponent).product()[:]
            self.Arr=Division(self.Arr, self.dividend).division()[:]
            self.Arr1=Sum(self.Arr, self.Arr1).sum()[:]
            self.sp=Count(self.Arr).update_sp()
            self.dividend=self.dividend+1
        print(concat(self.Arr1,"",0,999))
        return self.sp

Arr=[1]; Arr1=[1]; exponent=1
for j in range(1,1010): Arr.append(0); Arr1.append(0)
A=Compute(Arr,Arr1,exponent)
A.compute()

