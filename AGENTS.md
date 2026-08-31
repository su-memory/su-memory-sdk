# AGENTS.md — su-memory · AI 协作宪法

> 本文件是所有 AI 编程代理在本仓库工作的**硬性约束**。优先级高于个人偏好与通用最佳实践。
> 人类直接指令 > 本文件 > 通用约定。冲突时以人类指令为准。

## 1. 项目概览

Python SDK + FastAPI 服务（纯后端 API/SDK）。

**领域**: 医疗AI记忆引擎

## 2. 硬性规则（违反即返工）

### 2.1 代码改动
- **禁止臆造**: 不确定的 API/字段/常量先 `rg` 查证；查不到用 `TODO(待确认)` 占位，绝不编造。
- **最小改动**: 只改任务要求的部分，不顺手重构无关代码、不改文件名/变量名、不"修复"未要求的 bug（可在最终消息里提示人类）。
- **风格一致**: 遵循现有 lint 配置（ruff/mypy 或 eslint/prettier），不擅自引入新依赖。
- **中文注释/英文标识符**: 注释、docstring、日志、用户面文案用中文；标识符用英文。

### 2.2 安全红线
- **禁止 f-string 拼接 SQL**: 一律参数化查询。
- **禁止吞异常**: except 块必须有处理（日志/重抛/降级），不得空 except。
- **禁止明文打印敏感字段**: key/token/secret 不得进日志。

## 3. 验证命令（改动后必须跑通）

```bash
ruff check . && ruff format --check .
mypy .
pytest -m "not slow"
bash scripts/ai-verify/ai-guard.sh
```

## 4. 工作流约定

- 写业务代码前先填 `.ai-templates/structured-requirement.md`，经架构评审通过才编码。
- 提交前 `scripts/ai-verify/ai-guard.sh` 全绿（pre-commit 自动拦截）。
- 单次提交 ≤800 行 diff，超限拆分。
- 不自动 commit/push，改完汇报由人类决定。

## 5. 目录关键路径

```
api/ 或 routers/   路由层：参数解析 + 调 service + 统一响应；禁止业务逻辑
services/          业务层：业务逻辑、事务、流程控制唯一归属
models/            数据层：ORM 模型，数据库读写只在此
schemas/           DTO：Pydantic 入参/出参
core/ 或 common/   基础设施：config/security/exceptions/logging/cache
tests/             单元 + API 测试
```

详细规范见 `.ai-rules.md`。

---
*本文件由 AI 工程化套件生成（`~/qoder m5pro/_ai-eng-kit/install.sh`）。领域细节请在接入后人工补充完善。*
