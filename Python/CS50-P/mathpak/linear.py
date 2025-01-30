import sys

def add():
    try:
        a = int(sys.argv[1])
        b = int(sys.argv[2])
        return a+b
        
    except ValueError:
        return "wrong data type"
        
    