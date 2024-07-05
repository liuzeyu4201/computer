# Git 

## command
|clone|add|commit|push|status|remote|branch|checkout|merge|pull|
|---|---|---|---|---|---|---|---|---|---|
|从网络克隆一个仓库到本地|将某一文件添加到git中管理|添加某次更改对的评论|将更改推送到网络|查看状态|将某个本地文件关联到远程仓库|查看分支|创建、切换分支|合并分支|获取并合并分支




## create a new repo


```
1. git clone   https://github.com/liuzeyu4201/computer.git
2. git init 
    git remote add orgin https://github.com/liuzeyu4201/computer.git
```

## creat and check delete branch

```
git branch   # check exsiting branch
git checkout -b name  # creating a new branch
git checkout main    # switch the main branch
git branch -d name
```

## modify branch
```
git merge name  # merge two branches


git checkout name
git add 
git commit
git push -u origin name
git checkout main
git merge name
git checkout name 
git pull origin main # that include two commands{ git fetch 命令从 origin 拉取 main 分支的最新更改，然后 git merge 将这些更改合并到当前活动的本地分支。}

```
## SSH
<!--  -->
https://docs.github.com/en/authentication/connecting-to-github-with-ssh/generating-a-new-ssh-key-and-adding-it-to-the-ssh-agent

ls -la :  check all files 

## reference
youtube :   https://www.youtube.com/watch?v=RGOj5yH7evk&t=2822s