# Notes-Python

## Loop
For loop 
While loop
Match
## Exceptions
Exceptions are things that go wrong within our coding


## package library API
<font size=3>API</font>
    1. API 提供了一种标准化的方法，让开发者可以与系统、服务或其他软件组件进行交互，而不需要了解其内部实现。
    2. API 通常包括函数、类、方法、变量和数据结构的定义，并且可以用于网络服务（如 REST API）、操作系统功能、数据库访问等。


<font size=3>package</font>
在编程中，Package 是一种将相关模块或库组织在一起的机制。它通常包含多个模块、类和函数，打包在一起以便分发和使用。 一个跑中可能含有多个库
<font size=3>library</font>
Library 是一个或多个可重用代码的集合，通常包括多个函数、类、方法等，供其他程序调用。库通常专注于完成特定的任务或一组任务。
[Except](./Exceptions.py)
[library](./library.py)

## Regular expressions
.   any character except a new line
*   0 or more repetitions
+   1 or more repetitions
?   0 or 1 repetition
{m} m repetitions
{m,n} m-n repetitions
[]    set of characters
[^]   complementing the set
\d    decimal digit
\D    not a decimal digit
\s    whitespace characters
\S    not a whitespace character
\w    word character, as well as numbers and the underscore
\W    not a word character

```
import re

name = input("What's your name? ").strip()
matches = re.search(r"^(.+), (.+)$", name)
if matches:
    last, first = matches.groups()
    name = first + " " + last
print(f"hello, {name}")
```


## Class

### namespace and scopes
```
def scope_test():
    def do_local():
        spam = "local spam"

    def do_nonlocal():
        nonlocal spam
        spam = "nonlocal spam"

    def do_global():
        global spam
        spam = "global spam"

    spam = "test spam"
    do_local()
    print("After local assignment:", spam)
    do_nonlocal()
    print("After nonlocal assignment:", spam)
    do_global()
    print("After global assignment:", spam)

scope_test()
print("In global scope:", spam)
```
### class Objects 
类对象支持两种操作： 属性引用和实例化

1. 属性引用：指针对类级进行操作
```
class MyClass:
    """A simple example class"""
    i = 12345

    def f(self):
        return 'hello world'

```
2. 实例化： __init__
```
class Complex:
    def __init__(self, realpart, imagpart):
        self.r = realpart
        self.i = imagpart

x = Complex(3.0, -4.5)
x.r, x.i
(3.0, -4.5)

```
[class](./class.py)
## class and instance variavles : https://docs.python.org/3/tutorial/classes.html
1. 实例变量用于每个实例的唯一数据，类变量用于所有实例的共享数据
```
class Dog:

    tricks = []             # mistaken use of a class variable

    def __init__(self, name):
        self.name = name

    def add_trick(self, trick):
        self.tricks.append(trick)

>>> d = Dog('Fido')
>>> e = Dog('Buddy')
>>> d.add_trick('roll over')
>>> e.add_trick('play dead')
>>> d.tricks                # unexpectedly shared by all dogs
['roll over', 'play dead']
```
```
class Dog:

    def __init__(self, name):
        self.name = name
        self.tricks = []    # creates a new empty list for each dog

    def add_trick(self, trick):
        self.tricks.append(trick)

>>> d = Dog('Fido')
>>> e = Dog('Buddy')
>>> d.add_trick('roll over')
>>> e.add_trick('play dead')
>>> d.tricks
['roll over']
>>> e.tricks
['play dead']

```


## inherient
继承方法: 子类会自动继承父类的方法。
重写方法: 子类可以通过定义与父类同名的方法来重写父类的方法。
调用父类方法: 使用 super() 函数可以在子类中调用父类的方法。
扩展父类方法: 子类可以在调用父类方法的基础上扩展或修改行为。

[inherience](./inheritance.py)

## decoration
[decoration](./decoration.py)
https://blog.csdn.net/keepaware/article/details/112909406
https://blog.csdn.net/keepaware/article/details/111655393f