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




import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import torchvision.transforms as transforms
from torchvision import datasets

# 定义卷积神经网络
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        
        # 定义卷积层和池化层
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)  # 输入1通道（灰度图像），输出32通道
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)  # 输入32通道，输出64通道
        self.pool = nn.MaxPool2d(2, 2)  # 最大池化层，池化大小为2
        
        # 全连接层
        self.fc1 = nn.Linear(64 * 7 * 7, 128)  # 输入64*7*7，输出128
        self.fc2 = nn.Linear(128, 10)  # 输出10类（比如MNIST数据集）

    def forward(self, x):
        # 卷积层 + 激活函数 + 池化层
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        
        # 将二维的图像数据展平成一维
        x = x.view(-1, 64 * 7 * 7)
        
        # 全连接层
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# 加载数据集（以MNIST为例）
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
trainset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
trainloader = DataLoader(trainset, batch_size=64, shuffle=True)

# 实例化模型
model = SimpleCNN()

# 损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 训练网络
epochs = 5
for epoch in range(epochs):
    running_loss = 0.0
    for inputs, labels in trainloader:
        # 前向传播
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        
        # 后向传播和优化
        optimizer.zero_grad()  # 清空梯度
        loss.backward()  # 计算梯度
        optimizer.step()  # 更新参数
        
        running_loss += loss.item()
    
    print(f'Epoch [{epoch+1}/{epochs}], Loss: {running_loss/len(trainloader):.4f}')

print("Finished Training")
