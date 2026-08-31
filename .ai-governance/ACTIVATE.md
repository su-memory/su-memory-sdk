# 激活 governance 守卫（每个项目一次性操作）

## v2 更新（第一性原理审计后）
- **L2 凭证改为 review-token**（不可伪造）：旧版 `AI_REVIEW_CONFIRMED=1` 已废弃（单人可绕过=安全剧场）
- **L0 免审承诺诚实化**：门禁抓不到逻辑缺陷，L0 免逐行审的前提是**变异测试分数 ≥ 80%**（装 mutmut）
- **圈复杂度只查改动函数**：不再因触碰老文件被既有超标函数误伤
- **CI 服务端执行点**：本地 hook 可绕过，CI（governance.yml）是最终不可绕过防线

## 激活步骤

### 1. 安装工具链
```bash
pip install radon bandit ruff pytest pytest-cov pre-commit
pip install mutmut   # 可选, 启用 L0 变异测试(推荐)
```

### 2. 配置团队 review 密钥
管理员生成一个团队密钥（强随机串），通过内部门户/1Password 分发给每个开发者：
```bash
# 每个开发者在 shell 配置里（不进仓库!）
export GOV_REVIEW_SECRET="<团队密钥>"
```
GitHub 仓库: Settings → Secrets → 添加 `GOV_REVIEW_SECRET`（同名，供 CI 校验）

### 3. pre-commit 接入 governance 增强守卫
`.pre-commit-config.yaml` 的 ai-guard 指向 governance 版（已替换旧版的直接用 ai-guard.sh）：
```yaml
- repo: local
  hooks:
    - id: ai-guard
      name: ai-guard (governance)
      entry: bash scripts/ai-verify/ai-guard.sh
      language: system
      pass_filenames: false
```
```bash
pre-commit install
```

### 4. 启用 CI 不可绕过防线
`.github/workflows/governance.yml` 已随同步部署。配合分支保护：
Settings → Branches → main → 勾 "Require status checks" → 选 `governance`

## L2 代码提交流程（新流程）
```bash
# 1. 提交 → 本地 hook 阻断（无 token）
git commit -m "..."

# 2. 请第二人 review 后签发 token（第二人执行,用其持有的团队密钥）
GOV_REVIEW_SECRET=xxx bash scripts/ai-verify/review-token.sh issue <reviewer名>
# 输出: L2:main:abc123:reviewer-zhang|<签名>

# 3. 带 token 提交
AI_REVIEW_TOKEN='L2:main:abc123:reviewer-zhang|<签名>' git commit -m "..."

# 4. 推送 → CI 复核（PR approved-reviews-count>=2 或 token 有效）
git push
```
单人无法绕过：自己虽有密钥能签 token，但 CI 会检查 PR 是否有 ≥2 个不同人 approve。

## 环境变量
- `GOV_REVIEW_SECRET` — 团队 review 密钥（签发/校验 token，必须设置）
- `AI_REVIEW_TOKEN` — 第二人签发的 L2 凭证
- `AI_REVIEW_CI=1` — CI 模式，本地不阻断，留给服务端复核
- `MAX_RETRY=3` — 门禁失败重试上限（按暂存内容 hash 计数，超限转人工）
- `MAX_DIFF_LINES=800` — 单次提交行数上限
- `COVERAGE_HARD_GATE=1` — 覆盖率不达标硬阻断（默认警告）
- `MUTATION_GATE=1` — L1/L2 也跑变异测试（默认仅 L0）
- `MUTATION_MIN=80` — 变异分数阈值
- `PYTEST_SCOPE=changed` — 只跑改动相关测试；`full` 跑全量

## 与减重项目(ai-weight-management-platform)的关系
该项目是完整参考实现：venv 工具齐全、pre-commit 已装、governance.yml 已部署。
