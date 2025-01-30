class Parent:
    def __init__(self,name):
        self.name=name
    def Gen(self,num):
        print(f"{self.name}'s number is {num}")
        
class Child(Parent):
    def __init__(self,name):
        self.name=name
        
    def greet(self,num):
        super().Gen(num)      # 调用父类的方法
        
    


child=Child("bob")
child.greet(10)

# 子类与父类使用同一命名。 直接使用super\super.__init__ 对父类操作
# 或直接使用相同函数或变量名对父类直接修改（重写）。