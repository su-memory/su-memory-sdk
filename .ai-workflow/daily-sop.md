# 日常开发 SOP（速查）

1. **接任务** → 确认/填写 `.ai-templates/structured-requirement.md`（8 要素），提交人工评审。
2. **架构评审** → 按 `.ai-templates/architecture-output-template.md` 出方案 → 人工通过前禁止写码。
3. **定风险等级** → `bash scripts/ai-verify/risk-classify.sh` 定 L0/L1/L2（默认 L2）。
4. **套 Skill** → 从 `.ai-skills/` 选能力包，按其 `template.code` 实现，遵守 `forbid.list`。
5. **编码** → 按 `.ai-rules.md` 实现；禁止扩展需求、自创格式/异常。
6. **测试** → 写单测覆盖正常/边界/异常/空值；跑通全部测试。
7. **质检** → 按 `.ai-checklist/ai-code-review-checklist.md` 自查并修复。
8. **提交** → `git commit` 触发 ai-guard（L2 需第二人 review-token 或 CI 复核）。
9. **汇报** → 不自动 push/merge，汇报人类终审。

## 等级路由速查
| 等级 | 判定 | 提交要求 |
| --- | --- | --- |
| L0 | 纯工具/测试路径且无危险依赖 | 门禁过即放行（变异分数 ≥80% 才免抽审） |
| L1 | 普通业务/文档 | 放行 + 需架构/用例评审 |
| L2 | 支付/鉴权/用户数据/生产分支 | 阻断，需第二人 review-token 或 CI 双人 approve |

## 门禁失败处理
- 同一暂存内容自动重试 ≤3 次；达上限转人工，AI 停止重试。
- 报告在 `.ai-governance/reports/`。
