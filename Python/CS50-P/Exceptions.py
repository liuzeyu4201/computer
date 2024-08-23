# try -- except--else : 用于错误管理。

try:
    x = int(input("What's x?"))
    print(f"x is {x}")
except ValueError:  
    print("x is not an integer")
else :
    print(f"x is {x}")
    # pass