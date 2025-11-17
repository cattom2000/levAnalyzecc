# 设置CI/CD工作流说明

由于GitHub Personal Access Token权限限制，CI/CD工作流文件需要手动添加到仓库中。

## 步骤

1. **在GitHub仓库中创建工作流文件**:
   - 访问: https://github.com/cattom2000/levAnalyzecc
   - 点击 `.github/workflows/` 目录
   - 创建新文件 `ci-cd.yml`

2. **复制工作流内容**:
   复制 `.github/workflows/ci-cd.yml` 文件的内容到新创建的文件中。

3. **内容已准备**:
   CI/CD工作流文件已创建在本地 `.github/workflows/ci-cd.yml`，包含:
   - 多Python版本测试 (3.8-3.11)
   - 代码质量检查 (flake8, black, mypy, bandit)
   - 测试覆盖率报告
   - 性能基准测试
   - 安全漏洞扫描
   - 自动化部署

## 工作流特性

- ✅ 自动化测试执行
- ✅ 覆盖率报告生成
- ✅ 性能回归检测
- ✅ 安全漏洞扫描
- ✅ 代码质量检查
- ✅ 多环境支持

## 注意事项

- 工作流需要 `workflow` 权限的Personal Access Token
- 可以在GitHub设置中配置Secrets (如PYPI_API_TOKEN)
- 支持手动触发和定时执行

添加工作流文件后，每次推送都会自动触发CI/CD流水线。