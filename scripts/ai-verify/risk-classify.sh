#!/usr/bin/env bash
# AI 代码风险自动定级器（保守兜底：默认 L2，仅显式特征可降级）
# 用法: risk-classify.sh [file1 file2 ...]   不传参则扫 git 暂存+工作区改动
# 输出: 每文件一行 "LEVEL<TAB>path<TAB>reason"，末行 "---" 后一行 MAX_LEVEL
# 退出码: 0=L0  1=L1  2=L2
set -o pipefail
CALLER_CWD="$PWD"                                   # 保存调用方 cwd（路径 resolve 用）
ROOT="$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
cd "$ROOT"

# 路径硬域：词边界清晰，误报低
L2_PATH_RE='(payment|refund|settle|auth|login|webhook|alipay|wechatpay|wxpay|credential)'
# 硬词：任意出现即 L2（词边界，误报低）
L2_SYM_HARD_RE='\b(payment|refund|settle|auth|login|webhook|alipay|wechatpay|wxpay)\b'
# 软词：需定义性出现(def/class/import/from 行)，排除 order_by 等通用用法
L2_SYM_SOFT_RE='(^|[[:space:]])(def|class|import|from)[[:space:]].{0,40}\b(order|permission|token|crypto|encrypt|decrypt)[a-zA-Z_]*\b'
L2_CONFIG_RE='(\.env$|docker-compose|\.github/workflows/|/migrations/.*\.py$|alembic/versions/.*\.py$)'
L0_PATH_RE='(^|/)(utils|helpers|scripts|tests|test|__tests__|spec)/|^.*(_test|\.test|\.spec)\.(py|ts|tsx|js)$'
DANGER_IMPORT_RE='\b(payment|order|auth|user|permission|crypto)\b|db\.session|db\.sessionmaker|\b(redis|httpx|requests|aiohttp|axios)\b|fetch\('

BRANCH="$(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo unknown)"
case "$BRANCH" in
  main|master|prod|production|release/*) PROD_BRANCH=1;;
  *) PROD_BRANCH=0;;
esac

if [ "$#" -gt 0 ]; then
  FILES=("$@")
else
  FILES=()
  while IFS= read -r line; do [ -n "$line" ] && FILES+=("$line"); done < <(
    { git diff --cached --name-only --diff-filter=ACMR 2>/dev/null; \
      git diff --name-only --diff-filter=ACMR 2>/dev/null; } | sort -u)
fi

[ "${#FILES[@]}" -eq 0 ] && { echo "No files to classify"; exit 0; }

classify_one() {
  local f="$1"
  # 路径 resolve: 相对路径按调用方 cwd 解析成绝对路径，再判存在性
  case "$f" in
    /*) : ;;                                  # 已是绝对路径
    *)  if [ -f "$CALLER_CWD/$f" ]; then f="$CALLER_CWD/$f";  # 相对调用方 cwd
        elif [ -f "$f" ]; then :                           # 相对 git 根(cd 后)
        elif [ -f "$ROOT/$f" ]; then f="$ROOT/$f";         # 相对 git 根显式
        fi ;;
  esac
  [ -f "$f" ] || { printf 'SKIP\t%s\tnot-a-file\n' "$f"; return; }

  if [[ "$f" =~ $L2_CONFIG_RE ]]; then
    printf 'L2\t%s\tconfig/migration/CI-变更强制高级审\n' "$f"; return; fi

  local low; low="$(printf '%s' "$f" | tr 'A-Z' 'a-z')"
  if [[ "$low" =~ $L2_PATH_RE ]]; then
    printf 'L2\t%s\t路径命中高风险域: %s\n' "$f" "${BASH_REMATCH[1]}"; return; fi

  # 硬词任意出现即 L2（即便在测试目录，含真实支付/鉴权符号也要审）
  if grep -qiE "$L2_SYM_HARD_RE" "$f" 2>/dev/null; then
    printf 'L2\t%s\t符号命中高风险域(硬词)\n' "$f"; return; fi

  # L0 路径优先放行（测试/工具），仅当无危险 import
  if [[ "$f" =~ $L0_PATH_RE ]] && ! grep -qE "$DANGER_IMPORT_RE" "$f" 2>/dev/null; then
    printf 'L0\t%s\t纯工具/测试路径且无危险依赖\n' "$f"; return; fi

  # 软词需定义性出现（def/class/import/from），排除 order_by 等通用用法
  if grep -qiE "$L2_SYM_SOFT_RE" "$f" 2>/dev/null; then
    printf 'L2\t%s\t符号命中高风险域(软词定义)\n' "$f"; return; fi

  if grep -nqE '(@router|@app\.route|@command|@click|app\.(get|post|put|delete)\()' "$f" 2>/dev/null; then
    printf 'L1\t%s\t含路由/命令入口\n' "$f"; return; fi

  # 文档/说明类: 即便生产分支也只到 L1(不强制双人Review文档)
  case "$f" in
    *.md|*.txt|*.rst|*.json|*.yaml|*.yml|*.toml|LICENSE*|CHANGELOG*)
      printf 'L1\t%s\t配置/文档类(生产分支L1)\n' "$f"; return;;
  esac
  if [ "$PROD_BRANCH" = "1" ]; then
    printf 'L2\t%s\t生产分支默认保守定级\n' "$f"; return; fi
  printf 'L1\t%s\t普通业务代码默认定级\n' "$f"
}

MAX_LEVEL=0
for f in "${FILES[@]}"; do
  out="$(classify_one "$f")"
  printf '%s\n' "$out"
  lvl="${out%%	*}"
  case "$lvl" in
    L0) n=0;; L1) n=1;; L2) n=2;; *) continue;;
  esac
  [ "$n" -gt "$MAX_LEVEL" ] && MAX_LEVEL=$n
done

printf -- '---\n'
printf 'MAX_LEVEL\tL%d\n' "$MAX_LEVEL"
exit "$MAX_LEVEL"
