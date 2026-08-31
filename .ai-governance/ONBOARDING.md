# Governance 团队启用手册

> 目标：让 AI 代码风险分级（L0/L1/L2）+ 质量门禁在团队所有项目里**真正硬执行**。
> 本文是给团队 leader / 全体开发者的一次性 + 日常操作手册。

---

## 〇 一句话定位

governance = 代码改动按风险分三档（L0 工具/L1 业务/L2 支付鉴权），本地 hook 提醒 + **CI 强制**。
**CI + 分支保护是唯一不可绕过的执行点**，本地 hook 只起提醒作用（可被 owner 绕过，这是设计取舍）。

---

## 一、管理员一次性配置（团队 leader 做，约 15 分钟）

### 1.1 生成团队 review 密钥

```bash
# 在本机生成一个强随机串（32 字节 hex）
openssl rand -hex 32
# 输出示例: 7a3f9b2c8e1d4f6a0b5c3e7d9f2a1b4c8e6d0f3a5b7c9e1d3f2a4b6c8e0d5f7a
```

**分发规则（重要）**：
- 通过内部门户 / 1Password / 密码管理器分发给**每个开发者**，不进 git 仓库、不进聊天群、不写文档。
- GitHub 仓库：`Settings → Secrets and variables → Actions → New repository secret`
  - Name: `GOV_REVIEW_SECRET`
  - Value: 粘贴上面的密钥
- 每个仓库都要单独加（7 个项目 = 7 次重复，或用 org-level secret 一次分发到所有仓库）。

### 1.2 开启分支保护（最关键的一步，不开启 governance 等于摆设）

对每个项目仓库（以 ai-weight-management-platform 为例）：

1. GitHub 网页 → 仓库 → `Settings` → `Branches`
2. `Branch protection rules` → `Add rule`
3. `Branch name pattern`: `main`（生产分支）
4. 勾选并配置：
   - ☑ **Require a pull request before merging**
     - ☑ `Require approvals`: 设为 **2**（L2 代码需要双人 approve）
     - ☑ `Dismiss stale pull request approvals when new commits are pushed`
   - ☑ **Require status checks to pass before merging**
     - ☑ `Require branches to be up to date before merging`
     - `Search for status checks`: 添加 **`governance`**（这是 CI workflow 的 job 名）
   - ☑ `Do not allow bypassing the above settings`（连管理员也不绕过，最强约束；若要保留 owner 紧急绕过权可不勾此项）

> 开启后，任何直推 main 的提交（含本次会话那种 owner 直推）都会被 CI workflow `governance` 在 PR 阶段拦下，除非 review-token 有效或 PR 有 ≥2 approve。

### 1.3 确认 CI workflow 已部署

```bash
# 每个项目都应有这个文件
ls .github/workflows/governance.yml
```

已同步到全部 7 个项目。若新仓库需重新部署：
```bash
cp "/Users/mac/qoder m5pro/_ai-eng-kit/governance/governance-ci.yml" \
   .github/workflows/governance.yml
```

---

## 二、开发者日常配置（每人做一次，约 3 分钟）

### 2.1 配置 review 密钥到本地环境

把管理员分发的密钥写入 shell 配置（**不进 git**）：

```bash
# ~/.zshrc 或 ~/.bashrc 末尾追加
export GOV_REVIEW_SECRET="管理员给你的那串密钥"
```
```bash
source ~/.zshrc   # 或重开终端
echo $GOV_REVIEW_SECRET   # 应能看到密钥
```

### 2.2 安装 git hook（每个项目 clone 后必做）

```bash
cd <项目目录>
pip install pre-commit radon bandit ruff pytest pytest-cov
# 可选（启用 L0 变异测试）：
pip install mutmut   # 不装也能用 governance 自带的 mutation-check.sh

pre-commit install   # 安装 .git/hooks/pre-commit
```

> ⚠️ git hook 是**本地的**，不随仓库分发。新人 clone 后若不跑 `pre-commit install`，governance 本地不生效（但 CI 仍强制）。

---

## 三、日常开发流程（三级风险，三条通路）

### 3.1 改动前：自动知道走哪条路

写完代码 `git add` 后，本地 hook 会自动定级。想提前预判：
```bash
bash scripts/ai-verify/risk-classify.sh <你改的文件>
# 输出 L0/L1/L2 + 原因
```

### 3.2 L0 通路（工具/测试/纯函数）— 自动放行

改 `utils/`、`tests/`、纯计算函数，门禁全过 + 变异测试 ≥80% → 直接提交：
```bash
git commit -m "..."
# ✅ L0 自动放行
```

### 3.3 L1 通路（普通业务 CRUD）— 放行 + 评审标记

普通业务代码，门禁过 → 放行，但 PR 需架构/用例评审（不审源码）：
```bash
git commit -m "..."
# ✅ L1 放行(待架构/用例评审)
git push   # → 在 GitHub PR 里走架构评审
```

### 3.4 L2 通路（支付/鉴权/资金/核心架构）— 强制双人 Review

支付、订单、鉴权、加密、CI 配置改动 → **本地阻断**，需走 review-token 流程：

```bash
git commit -m "..."
# 🛑 L2 高风险：阻断提交

# 步骤1: 请第二人 review 你的改动
# 步骤2: 第二人用其持有的密钥签发 token（第二人在自己机器执行）：
GOV_REVIEW_SECRET=xxx bash scripts/ai-verify/review-token.sh issue <第二人名字>
# 输出: L2:main:abc1234:zhangsan|<签名串>

# 步骤3: 提交者带 token 提交：
AI_REVIEW_TOKEN='L2:main:abc1234:zhangsan|<签名串>' git commit -m "..."
# ⚠️ L2 review-token 有效(reviewer: zhangsan)，允许提交

# 步骤4: 推送，CI 复核（PR approved-reviews-count≥2 或 token 有效）
git push
```

**单人无法伪造**：自己虽有密钥能签 token，但 CI 会查 PR 是否真有 ≥2 个不同人 approve。

### 3.5 紧急情况 / CI 环境

- CI 环境（GitHub Actions）：本地不阻断，`AI_REVIEW_CI=1` 留给服务端复核。
- 紧急 hotfix 且无法走 PR：临时直推 main，但分支保护开启后会被拦（除非勾了 admin bypass）。建议保留 admin bypass 作为紧急逃生阀，但每次绕过应在事后补 PR + 复盘。

---

## 四、各项目落地状态（2026-07-27）

| 项目 | 脚本 | CI workflow | pre-commit hook | profile |
| --- | --- | --- | --- | --- |
| ai-weight-management-platform | ✓ 6个 | ✓ | ✓ | fastapi（参考实现） |
| MCI-SDK | ✓ 6个 | ✓ | ✓ | fastapi |
| mci-world-model | ✓ 6个 | ✓ | ✓ | python-research |
| nutrition-system-hospital | ✓ 6个 | ✓ | ✓ | fullstack |
| clinical-nutrition-finance-prod | ✓ 7个 | ✓ | ✓ | fullstack |
| mci-huan | ✓ 6个 | ✓ | ✓ | fullstack-wx |
| ai-tumor-nutrition-platform | ✓ 5个 | ✓ | ✓(monorepo) | fullstack-wx |

**全部已落地（2026-07-28 核对）**。新人 clone 后只需 §二 的开发者配置（密钥 + pre-commit install）。

> ⚠️ `ai-tumor-nutrition-platform` 是 `qoder m5pro` monorepo 的子目录（共享 `.git`），
> 它的 pre-commit hook 装在 monorepo 根 `qoder m5pro/.git/hooks/`，配置指向子目录的
> `.pre-commit-config.yaml`。操作时 `cd` 到子目录即可，pre-commit 会自动找到正确配置。

---

## 五、环境变量速查

| 变量 | 作用 | 默认 |
| --- | --- | --- |
| `GOV_REVIEW_SECRET` | 团队 review 密钥（签发/校验 token） | 必须设 |
| `AI_REVIEW_TOKEN` | 第二人签发的 L2 凭证 | — |
| `AI_REVIEW_CI=1` | CI 模式，本地不阻断 | — |
| `MAX_RETRY=3` | 门禁失败重试上限（按暂存内容 hash 计数，超限转人工） | 3 |
| `MAX_DIFF_LINES=800` | 单次提交行数上限（防 Vibe Coding） | 800 |
| `COVERAGE_HARD_GATE=1` | 覆盖率不达标硬阻断（默认警告） | 警告 |
| `MUTATION_GATE=1` | L1/L2 也跑变异测试（默认仅 L0） | 仅 L0 |
| `MUTATION_MIN=80` | 变异分数阈值 | 80 |
| `MUTATION_HARD_GATE=1` | 变异分数不达标硬阻断（默认警告） | 警告 |
| `PYTEST_SCOPE=changed` | 只跑改动相关测试；`full` 跑全量（留给 CI） | changed |

---

## 六、常见问题

**Q1: 我 `git commit` 报 "🛑 L2 高风险：阻断提交"，但我的改动其实不危险**
A: 定级器按路径/符号/关键字静态推断，偏保守。确认改动确实安全后，两种处理：
- 小拆分：把高风险部分（如改了 payment 相关）单独拆成 L2 提交走评审，其余正常提交。
- 误判：在 `.ai-requirement.md` 标注 `risk: L0`（覆盖自动定级），并在 PR 说明理由。

**Q2: review-token 怎么"第二人签发"？我团队就我一个开发**
A: governance 的 L2 双人复核依赖团队规模。单人/小团队可：
- 改 `MUTATION_HARD_GATE=1` 加强自动化验证，降低对人工的依赖；
- 或对 L2 放宽为 `AI_REVIEW_CI=1`（CI 只查测试通过，不查 approve 数）——但这削弱了防护，需自知取舍。

**Q3: 变异测试跑得太慢**
A: `mutation-check.sh` 只对单个改动文件抽样（非全量），通常 10-30 秒。若仍慢，临时 `MUTATION_GATE=0` 跳过（但 L0 免审资格失效，需人工抽审）。

**Q4: CI 的 governance check 一直 pending / 不触发**
A: 检查 `.github/workflows/governance.yml` 是否在默认分支；确认 `GOV_REVIEW_SECRET` 已加到仓库 Secrets；CI 首次跑需 PR 触发（直推不触发 PR 路径）。

**Q5: governance 报告里都是我自己测试时产生的，怎么清理？**
A: 报告在 `.ai-governance/reports/`，已加入 `.gitignore` 不入库。本地可随时清空该目录。

---

## 七、升级 governance 规则

改 `_ai-eng-kit/governance/` 源头后，全量重推到所有项目：
```bash
bash "/Users/mac/qoder m5pro/_ai-eng-kit/sync-governance.sh"
```
单项目：
```bash
bash "/Users/mac/qoder m5pro/_ai-eng-kit/sync-governance.sh" "/项目路径"
```

---

## 八、验收 checklist（启用 governance 后自检）

- [ ] 团队密钥已生成，已分发到每个开发者
- [ ] GitHub org/仓库 Secrets 已加 `GOV_REVIEW_SECRET`
- [ ] 7 个仓库都开启了 main 分支保护 + governance status check
- [ ] 每个开发者本地 `echo $GOV_REVIEW_SECRET` 能看到密钥
- [ ] 每个开发者每个项目都跑了 `pre-commit install`
- [ ] 做过一次演练：故意提一个 L2 改动，确认本地阻断 + CI 拦截 + token 放行全链路通
