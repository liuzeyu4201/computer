# property
class Salary:
    def __init__(self,name,salary):
        self.name=name
        self.salary=salary
    @property
    def check_salary(self):
        return self.salary
    
    @check_salary.setter
    def check_salary(self,num):
        self.salary=num
    
Bob=Salary('Bob',100)
Bob.check_salary=200
print(Bob.check_salary)






# classmethod
class Student:
    def __init__(self, name, house,age):
        self.name = name
        self.house = house
        self.age = age

    def __str__(self):
        return f"{self.name} from {self.house},age is {self.age} "

    @classmethod
    def get(cls):
        name = input("Name: ")
        house = input("House: ")
        age= input("age:") 
        return cls(name, house,age)  # 类实例化


def main():
    student = Student.get()
    student2=Student("hailen","whitehouse",21)
    print(student2)
    print(student)


if __name__ == "__main__":
    main()
