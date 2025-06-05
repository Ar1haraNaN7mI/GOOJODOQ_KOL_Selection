# GOOJODOQ达人匹配系统 - GitHub上传指南

## 前置准备

1. 确保已在GitHub创建名为"GOOJODOQ_KOL_Selection"的空仓库
2. 确保本地安装了Git

## 上传步骤

使用PowerShell执行以下命令：

```powershell
# 1. 进入项目目录(示例路径，请根据实际情况修改)
cd C:\Users\MSIK\Desktop\ChatBot\infound-inner

# 2. 初始化Git仓库
git init

# 3. 添加所有文件到暂存区
git add .

# 4. 提交更改
git commit -m "初始提交：GOOJODOQ达人匹配系统"

# 5. 添加远程仓库
git remote add origin https://github.com/Ar1haraNaN7mI/GOOJODOQ_KOL_Selection.git

# 6. 推送代码到GitHub
git push -u origin master
```

如果出现错误，可能需要按照提示进行其他操作，如：

```powershell
# 如果远程仓库有内容，需要先拉取合并
git pull --rebase origin master

# 如果默认分支是main而不是master
git push -u origin main
```

## 常见问题解决

1. **认证失败**：确保已配置GitHub账号，或使用个人访问令牌(PAT)认证
2. **推送被拒绝**：可能是因为远程仓库有你本地没有的提交，使用`git pull`先同步
3. **大文件错误**：删除过大的文件(如`.venv`目录)，可以添加到`.gitignore`

## 提交后确认

成功推送后，在浏览器中访问 https://github.com/Ar1haraNaN7mI/GOOJODOQ_KOL_Selection 查看上传结果。 