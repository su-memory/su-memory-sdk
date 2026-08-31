# AI 编程工程化 · 启动指南

> 本项目已接入 AI 工程化体系。请任何 AI 编码助手（Codex/Cursor/Claude 等）在动手前先读本文件与 `.ai-rules.md`。
> 铁律：人类定义架构/规则/安全；AI 只做标准化执行；所有产出必须经校验、测试、质检、人工终审。

---

## 一、AI 助手开场三句话（粘贴给 AI）

```
请先阅读项目根目录的 .ai-rules.md（AI 宪法）与本项目的 .ai-skills/，严格遵守。
本次任务的结构化需求见 .ai-templates/structured-requirement.md（或我现在描述）。
未填结构化需求、未经架构评审前，禁止写业务代码。
```

## 二、标准开发流程（每改一处代码都走这 7 步）

1. **填需求** — 复制 `.ai-templates/structured-requirement.md`，填齐 8 要素（业务目标/入参/出参/依赖/边界/异常/性能/测试）。禁止一句话需求。
2. **架构 Agent** — 让 AI 按 `.ai-templates/architecture-output-template.md` 出方案，**人工审核通过**才继续。架构不过，禁止写代码。
3. **匹配 Skill** — 在 `.ai-skills/` 找对应能力包，让 AI 套用其 `template.code`，禁止自创格式。
4. **编码 Agent** — AI 按 `.ai-rules.md` + Skill 模板实现。禁止扩展需求、自创返回格式/异常体系。
5. **测试 Agent** — AI 生成测试（后端 pytest / 前端 vitest / 小程序 jest），覆盖正常/边界/异常/空值。
6. **质检 Agent** — AI 按 `.ai-checklist/ai-code-review-checklist.md` 自查，输出问题清单+修复方案。
7. **提交** — `git commit` 自动触发 `scripts/ai-verify/ai-guard.sh`，拦截空异常/拼接 SQL/明文敏感字段/超大 diff。人工在 `.ai-checklist/` 上逐项打勾后合并。

> 速查全流程：见 `.ai-workflow/daily-sop.md`。

## 三、四角色提示词

直接用 `.ai-workflow/agent-prompt-all.md` 里的架构/编码/测试/质检四段 prompt，分步喂给 AI。

## 四、运行命令（按栈选）

```bash
ruff check . && ruff format --check .
mypy .
pytest -m "not slow"
bash scripts/ai-verify/ai-guard.sh
```

### 手动触发守卫（不提交也能跑）
```bash
bash scripts/ai-verify/ai-guard.sh
```

## 五、红线（违反即打回）

- ❌ 无结构化需求就开工
- ❌ 未经架构评审就写业务代码
- ❌ AI 自创返回格式 / 异常体系 / 接口结构
- ❌ 跨层调用（路由直接写库、api 层写业务）
- ❌ 裸 SQL 拼接、硬编码密钥、明文打印敏感字段
- ❌ 无测试就提交 / 测试不过就合入
- ❌ 单次提交 > 800 行（疑似 Vibe Coding，被 ai-guard 拦截）

## 六、如何沉淀新 Skill

同一模式重复出现 3 次以上 → 复制 `.ai-skills/` 任意一个能力包结构，沉淀为五件套：`SKILL.md` + `template.code` + `forbid.list` + `test-case.list` + `error-fix.md`，纳入团队基线。

## 七、升级体系

体系源头在 `~/qoder m5pro/_ai-eng-kit/`。改源头后重跑：
```bash
bash ~/qoder\ m5pro/_ai-eng-kit/install.sh <本项目路径> <profile>
bash ~/qoder\ m5pro/_ai-eng-kit/sync-governance.sh <本项目路径>
```
（注意：重跑 install 会覆盖 `.ai-rules.md`，项目内若有定制请先备份。）
