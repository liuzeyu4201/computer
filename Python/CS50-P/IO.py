name=input("what is your name?")
file=open('config.txt',"a")
file.write(f"name is :{name}\n")
file.close
