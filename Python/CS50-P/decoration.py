class Worker:

    def __init__(self, name, salary):
        self.name = name
        self.salary = salary

    @property     
    # 先使用property装饰原函数，装饰之后就可以使用setter和deleter方法来进行装饰并有更多的功能了
    # 这里默认已经有了getter的功能，也就是能够获取到这个属性的能力
    def view_salary(self):
        return self.salary

    @view_salary.setter
    # setter装饰器能够使我们在类外部重新设置view_salary这个属性
    # 而没有setter的话就会报错  AttributeError: can't set attribute
    def view_salary(self, new_salary):
        self.salary = new_salary
        return new_salary

    @view_salary.deleter
    # deleter装饰器的设置使我们可以在类外部使用del删除这个属性，但是这里我试了将self.salary
    # 设置为其它值也是可以的哦，这个值设置非空的话还是会有属性继续存在的哦。也就是在类外部使用
    # del的功能取决于你自己在这里设置的操作
    # 我这里是设置为空值了
    def view_salary(self):
        self.salary = None

A = Worker('J',1000)

A.view_salary = 2000

print(A.view_salary)
