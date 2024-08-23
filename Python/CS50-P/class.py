class Person:
    def __init__(self,name,age,num):
        self.name=name
        try:
            self.age=int(age)
        except:
            print("ValueError: not a number")
        try: 
            self.num=int(num)
        except:
            print("ValueError: not a number")
    def prime(self):
        return self.age + self.num
    

def main():
    person=getname()
    pr=person.prime()
    print(f"he's name is{person.name},{person.age} years old,{pr}")
    print(Person.__dict__)
    
    
def getname():
    name=input("name:")
    age=input("age:")
    num=input("num:")
    return Person(name,age,num)
    
if   __name__=="__main__":
    main()