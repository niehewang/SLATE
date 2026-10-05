# 用 Git 上传 SLATE 到 GitHub

建议在**能访问 GitHub 的本地电脑**上操作，而不是在无法连接 GitHub 的实验服务器上操作。

## 1. GitHub 创建空仓库

在 GitHub 新建仓库，例如 `SLATE`。第一次创建时不要勾选自动生成 README、`.gitignore` 或 License，因为本代码包里已经有 README 和 `.gitignore`。

## 2. 安装并确认 Git

```bash
git --version
```

Windows 可安装 Git for Windows；macOS 可使用 Xcode Command Line Tools/Homebrew；Linux 使用系统包管理器。

## 3. 解压代码包并进入目录

```bash
cd /path/to/SLATE
```

## 4. 初始化仓库

```bash
git init
git branch -M main
git status
```

## 5. 配置提交身份（第一次使用 Git 时）

```bash
git config --global user.name "Your Name"
git config --global user.email "your-email@example.com"
```

## 6. 首次提交

```bash
git add .
git status
git commit -m "Initial public release of SLATE"
```

提交前务必用 `git status` 确认没有模型、数据、checkpoint、token 或大文件被加入。

## 7. 连接 GitHub 远端

HTTPS：

```bash
git remote add origin https://github.com/YOUR_USERNAME/SLATE.git
git remote -v
```

或 SSH：

```bash
git remote add origin git@github.com:YOUR_USERNAME/SLATE.git
git remote -v
```

## 8. 推送

```bash
git push -u origin main
```

GitHub 已不支持使用账号密码进行 Git HTTPS 身份验证。HTTPS 可使用浏览器/Git Credential Manager 或 Personal Access Token；SSH 可配置 SSH key。

## 9. 后续更新

```bash
git status
git add .
git commit -m "Describe the update"
git push
```

## 10. 常见问题

### `remote origin already exists`

```bash
git remote set-url origin https://github.com/YOUR_USERNAME/SLATE.git
```

### GitHub 仓库不是空的，push 被拒绝

如果 GitHub 端已经创建 README 等文件：

```bash
git pull --rebase origin main
git push -u origin main
```

### 文件超过 GitHub 100 MB 限制

本公开包已排除模型/数据等大文件。若以后确实需要发布大模型或数据，不建议直接放 Git；使用 release、数据仓库或 Git LFS，并先确认许可证。

### 实验服务器无法访问 GitHub

不要在该服务器上 `git push`。把代码 ZIP 下载到能访问 GitHub 的电脑，在本地执行上述命令即可。
