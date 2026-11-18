# 将 SLATE 上传/更新到 GitHub

仓库地址：`https://github.com/niehewang/SLATE`

## 首次配置当前仓库身份

```bash
git config user.name "Hewang Nie"
git config user.email "1667279805@qq.com"
```

## 日常更新

在 SLATE 仓库根目录：

```bash
git status
git add .
git commit -m "Polish public repository metadata"
git push
```

## 检查远端

```bash
git remote -v
```

应指向：

```text
https://github.com/niehewang/SLATE.git
```

## 如果远端有新提交

先同步再提交：

```bash
git pull --rebase origin main
```

如有明确的 merge 冲突，应先解决冲突再 push；不要习惯性使用 `--force`。
