# Swift 数据类型

## 基本数据类型

| Int                             | Uint   | Float      | Double |
| ------------------------------- | ------ | ---------- | ------ |
| Bool                            | String | Character  |        |
| Array let numbers:[Int]=[1,2,3] | Set    | Dictionary |        |
|                                 |        |            |        |
##  高级数据类型

| 元组                                 | 类                                                           | 结构体                                                       |
| ------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| let person:(String,Int)=("Alice",25) | class Animal {<br/>    var name: String<br/>    init(name: String) {<br/>        self.name = name<br/>    }<br/>}<br/><br/>let dog = Animal(name: "Dog") | struct User{var name :string<br />var age:Int}<br />let user=User(name:"Bob",age:30) |
|                                      |                                                              |                                                              |

```
类
class Animal {
    var name: String
    init(name: String) {
        self.name = name
    }
}

let dog = Animal(name: "Dog")
let anotherDog = dog
anotherDog.name = "Cat"
print(dog.name) // "Cat" （共享同一个对象）

```

```
结构体
struct Point {
    var x: Int
    var y: Int
}
var p1 = Point(x: 3, y: 4)
var p2 = p1
p2.x = 10
print(p1.x) // 3 （因为是值拷贝）

```

```
区别展示
struct User {
    var name: String
    var age: Int
}

var user1 = User(name: "Alice", age: 20)
var user2 = user1
user2.name = "Bob"

print(user1.name) // Alice
print(user2.name) // Bob



class UserClass {
    var name: String
    var age: Int

    init(name: String, age: Int) {
        self.name = name
        self.age = age
    }
}

var user1 = UserClass(name: "Alice", age: 20)
var user2 = user1
user2.name = "Bob"

print(user1.name) // Bob
print(user2.name) // Bob
结构体是数据拷贝，类是数据共用。

```







## 数据结构

### 链表

```swift
class ListNode{
	var value :Int
    var next: ListNode？
    init(value:Int){
        self.value=value
    }
}
let first=ListNode(value:1)
let second=ListNode(value:2)
let third=ListNode(value:3)
first.next=second
second.next=third

注意在Swift中加入？ 表示可选类型（可以是nil~None）否则不传入报错。
var next: ListNode?

if let nextnode = first.next{
	print(nextnode.value)
}else{
	print("first.next=nil")
}
不能直接打印 first.next.value 因为对可选类型的值可能为nil。
```



## 语法

Var 可变声明变量。let不可变声明变量。



### 计算属性

```swift
func getCurrentCalibrationPoint() -> CGPoint? {
    guard currentPointIndex < calibrationPositions.count else { return nil }
    let position = calibrationPositions[currentPointIndex]
    let screenSize = UIScreen.main.bounds.size
    return CGPoint(x: position.x * screenSize.width,
                   y: position.y * screenSize.height)
}
等于
var currentCalibrationPoint: CGPoint? {
    guard currentPointIndex < calibrationPositions.count else { return nil }
    let position = calibrationPositions[currentPointIndex]
    let screenSize = UIScreen.main.bounds.size
    return CGPoint(x: position.x * screenSize.width,
                   y: position.y * screenSize.height)
}

```





