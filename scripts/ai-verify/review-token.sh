#!/usr/bin/env bash
# L2 review-token 签发与校验（不可伪造：需 HMAC 密钥 + 第二人签发）
#
# 用法1 签发(第二人执行): review-token.sh issue <reviewer> [commit-ish]
#   读取 GOV_REVIEW_SECRET 环境变量(团队共享密钥), 输出 token
# 用法2 校验(ai-guard 内部): review-token.sh verify <token> <commit-ish>
#
# 安全模型:
#   - 密钥不在仓库里, 由团队通过 1Password/内部门户分发, 每个开发者持有
#   - 第二人 review 后用自己持有的密钥签发 token, 提交者把它放进 AI_REVIEW_TOKEN
#   - 本地 hook 校验签名有效 + commit 匹配 → 才放行 L2
#   - 单人无法伪造: 自己有密钥能签, 但 token 要求 reviewer != 提交者(由 CI 强制)
#   - 最终不可绕过点在 CI: CI 用服务端密钥重新校验 + 检查 PR approved-reviews-count>=2
set -o pipefail

SECRET="${GOV_REVIEW_SECRET:-}"
if [ -z "$SECRET" ]; then
  echo "ERROR: 未设置 GOV_REVIEW_SECRET 环境变量" >&2
  echo "  团队密钥由管理员分发; 个人签发需持有密钥" >&2
  exit 2
fi

cmd="${1:-verify}"
shift || true

hmac() {
  # 用 openssl 生成 HMAC-SHA256, 输出 hex
  printf '%s' "$2" | openssl dgst -sha256 -hmac "$1" 2>/dev/null | awk '{print $NF}'
}

case "$cmd" in
  issue)
    reviewer="${1:?用法: issue <reviewer> [commit-ish]}"
    commit="${2:-HEAD}"
    commit="${commit:-HEAD}"
    short="$(git rev-parse --short "$commit" 2>/dev/null || echo "$commit")"
    branch="$(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo unknown)"
    payload="L2:${branch}:${short}:${reviewer}"
    sig="$(hmac "$SECRET" "$payload")"
    token="${payload}|${sig}"
    echo "$token"
    echo "  → 提交者使用: AI_REVIEW_TOKEN='$token' git commit ..." >&2
    ;;
  verify)
    token="${1:?用法: verify <token> [commit-ish]}"
    commit="${2:-HEAD}"
    payload="${token%|*}"
    sig="${token##*|}"
    # 重新计算期望签名
    expect_sig="$(hmac "$SECRET" "$payload")"
    if [ -z "$expect_sig" ] || [ -z "$sig" ]; then
      echo "invalid: token 格式错误"; exit 1; fi
    if [ "$sig" != "$expect_sig" ]; then
      echo "invalid: 签名不匹配(密钥错误或token被篡改)"; exit 1; fi
    # 校验 commit 匹配(防 token 跨提交复用)
    token_commit="${payload##*:}"           # 取 payload 末段(原为 reviewer, 这里需调整)
    # payload 格式 L2:branch:short:reviewer, commit 是第3段
    token_branch="$(printf '%s' "$payload" | cut -d: -f2)"
    token_short="$(printf '%s' "$payload" | cut -d: -f3)"
    cur_short="$(git rev-parse --short "$commit" 2>/dev/null || echo "")"
    if [ -n "$cur_short" ] && [ "$token_short" != "$cur_short" ]; then
      echo "invalid: token 绑定的提交($token_short)与当前($cur_short)不符"; exit 1; fi
    echo "valid: $payload"
    exit 0
    ;;
  *)
    echo "用法: review-token.sh issue|verify ..." >&2; exit 2;;
esac
