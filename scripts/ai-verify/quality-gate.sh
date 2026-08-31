#!/usr/bin/env bash
# AI 代码质量门禁（按风险等级差异化阈值）
# 用法: quality-gate.sh <L0|L1|L2> [file1 file2 ...]   不传文件则扫暂存改动
# 退出码: 0=通过  1=失败  2=缺少工具(跳过,警告不阻断)
set -o pipefail
ROOT="$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
cd "$ROOT"

LEVEL="${1:-L2}"; shift || true
case "${LEVEL}" in L0|L1|L2) ;; *) LEVEL=L2;; esac

REPORT_DIR=".ai-governance/reports"
# 工具路径：优先环境变量，再 PATH。支持 venv（如 RADON_BIN=backend/.venv/bin/radon）
RADON_BIN="${RADON_BIN:-radon}"
RUFF_BIN="${RUFF_BIN:-ruff}"
PYTEST_BIN="${PYTEST_BIN:-pytest}"
BANDIT_BIN="${BANDIT_BIN:-bandit}"
has(){ command -v "$1" >/dev/null 2>&1; }
mkdir -p "$REPORT_DIR"
TS="$(date +%Y%m%d-%H%M%S)"
REPORT="$REPORT_DIR/$TS-gate-${LEVEL}.md"

if [ "$#" -gt 0 ]; then FILES=("$@"); else
  FILES=()
  while IFS= read -r l; do [ -n "$l" ] && FILES+=("$l"); done < <(
    git diff --cached --name-only --diff-filter=ACMR 2>/dev/null | sort -u)
fi
[ "${#FILES[@]}" -eq 0 ] && FILES=(".")

say(){ printf '\033[1m[gate]\033[0m %s\n' "$1"; }

# 差异化阈值表
case "${LEVEL}" in
  L0) COV_CORE=85; COV_BR=80; COV_UTIL=70;  CC_MAX=15; DUP=5;;
  L1) COV_CORE=90; COV_BR=85; COV_UTIL=80;  CC_MAX=15; DUP=5;;
  L2) COV_CORE=95; COV_BR=90; COV_UTIL=85;  CC_MAX=15; DUP=5;;
esac

{
  echo "# 质量门禁报告"
  echo "- 时间: $(date '+%Y-%m-%d %H:%M:%S')"
  echo "- 风险等级: ${LEVEL}"
  echo "- 阈值: 核心覆盖≥${COV_CORE}% 分支≥${COV_BR}% 工具≥${COV_UTIL}% 圈复杂度≤${CC_MAX} 重复≤${DUP}%"
  echo "- 文件: ${FILES[*]}"
  echo
} > "$REPORT"

FAIL=0
WARN=0
pass(){ echo "- ✅ $1" | tee -a "$REPORT"; }
fail(){ echo "- ❌ $1" | tee -a "$REPORT" >&2; FAIL=1; }
warn(){ echo "- ⚠️  $1" | tee -a "$REPORT"; WARN=1; }

# ---------- 1. 基础静态门禁（无外部工具，必跑） ----------
# 1a. 空 except / 空 catch
py_files=(); js_files=()
for f in "${FILES[@]}"; do
  case "$f" in
    *.py) py_files+=("$f");;
    *.js|*.ts|*.tsx) js_files+=("$f");;
  esac
done

if [ "${#py_files[@]}" -gt 0 ]; then
  # 用 AST 精确检测空 except（body 仅 pass/.../docstring），避免跨行误判
  python3 - "${py_files[@]}" <<'PYAST' 2>/dev/null && true
import ast, sys
hit=[]
for p in sys.argv[1:]:
    try: tree=ast.parse(open(p,encoding="utf-8").read(), filename=p)
    except Exception: continue
    for n in ast.walk(tree):
        if isinstance(n, ast.ExceptHandler):
            b=n.body
            # 跳过纯 docstring
            body=[x for x in b if not (isinstance(x,ast.Expr) and isinstance(x.value,ast.Constant))]
            if not body or all(isinstance(x,(ast.Pass,)) for x in body):
                hit.append(f"{p}:{n.lineno}: 空 except (仅 pass/docstring)")
    # pass-through handlers (Ellipsis ...)
    for n in ast.walk(tree):
        if isinstance(n, ast.ExceptHandler):
            body=[x for x in n.body if not (isinstance(x,ast.Expr) and isinstance(x.value,ast.Constant))]
            if body and all(isinstance(x,ast.Expr) and isinstance(x.value,ast.Constant) and x.value.value is ... for x in body):
                hit.append(f"{p}:{n.lineno}: 空 except (仅 ...)")
if hit:
    print("\n".join(hit[:10])); sys.exit(1)
PYAST
  rc=$?
  if [ "$rc" -ne 0 ]; then fail "检测到空 except（Python，AST 精确匹配）"; fi
fi
if [ "${#js_files[@]}" -gt 0 ]; then
  if grep -lnE 'catch\s*\([^)]*\)\s*\{\s*\}' "${js_files[@]}" 2>/dev/null; then
    fail "检测到空 catch（JS/TS）"; fi
fi

# 1b. f-string / 模板拼接 SQL
if grep -rnE "f['\"].*(SELECT|INSERT|UPDATE|DELETE|FROM|WHERE)" "${FILES[@]}" 2>/dev/null \
  | grep -v '_test\|\.test\.\|/tests/\|/test/' | head -5; then
  fail "疑似 f-string 拼接 SQL，必须参数化"; fi

# 1c. 硬编码密钥 / 明文敏感字段
if grep -rnE '(password|secret|apikey|api_key|token)\s*=\s*["\x27][^"\x27]{6,}' "${FILES[@]}" \
  --include='*.py' --include='*.js' --include='*.ts' 2>/dev/null | grep -v 'env\|config\|getenv\|os\.' | head -5; then
  fail "疑似硬编码密钥/敏感字段"; fi

# 1d. 裸 print 进生产（仅 src 层）
if grep -rn '^\s*print(' "${FILES[@]}" --include='*.py' 2>/dev/null \
  | grep -v '_test\|/tests/\|scripts/' | head -5; then
  warn "检测到 print（生产应用 logging）"; fi

# ---------- 2. 圈复杂度（按栈选工具） ----------
check_cc() {
  if has "$RADON_BIN" && [ "${#py_files[@]}" -gt 0 ]; then
    # 拿改动行（暂存区 -U0），与 radon 函数行做交集，只判改动函数复杂度
    python3 - "$RADON_BIN" "$CC_MAX" "${py_files[@]}" <<'PYCC' 2>/dev/null
import ast, subprocess, sys, os
radon, cc_max = sys.argv[1], int(sys.argv[2])
files = sys.argv[3:]
norm=lambda p: os.path.relpath(os.path.abspath(p), os.getcwd())
# 1. 每个文件的改动行集合(暂存区 -U0)
in_git = subprocess.run(["git","rev-parse","--is-inside-work-tree"],
                        capture_output=True,text=True).returncode==0
changed={}
for pf in files:
    ch=set()
    if in_git:
        r=subprocess.run(["git","diff","--cached","-U0","--",pf],
                         capture_output=True,text=True)
        for ln in r.stdout.splitlines():
            # @@ -a,b +c,d @@ : 改动后行从 c 起 b 行(省略=1)
            m=__import__("re").search(r"\+(\d+)(?:,(\d+))?", ln.split("@@")[-2] if "@@" in ln else "")
            if not m: continue
            c=int(m.group(1)); n=int(m.group(2) or 1)
            ch.update(range(c,c+n))
    changed[norm(pf)] = (ch if ch else None)   # None=无改动或非git→全查
# 2. radon 复杂度
out=subprocess.run([radon,"cc","-s"]+files,capture_output=True,text=True).stdout
cc_map={}   # (file, funcname) -> cc
cur=None
import re as _re
for ln in out.splitlines():
    if not ln.startswith(" ") and ln.endswith(".py"):
        cur=norm(ln); continue
    m=_re.match(r"\s*[FMC]\s+(\d+):\d+\s+(.+?)\s+-\s+[A-F]\s+\((\d+)\)",ln)
    if m and cur: cc_map[(cur,m.group(2))]=int(m.group(3))
# 3. AST 算每个函数的行范围, 与改动行求交
hits=[]
for pf in files:
    fn=norm(pf); ch=changed.get(fn)
    try: tree=ast.parse(open(pf,encoding="utf-8").read())
    except Exception: continue
    for node in ast.walk(tree):
        if not isinstance(node,(ast.FunctionDef,ast.AsyncFunctionDef)): continue
        fl=node.lineno; end=getattr(node,"end_lineno",node.lineno)
        # 函数被改动 = 无改动信息(全查) 或 任一改动行落在函数行范围
        touched = (ch is None) or any(fl<=l<=end for l in (ch or []))
        if not touched: continue
        cc=cc_map.get((fn,node.name))
        if cc and cc>cc_max:
            hits.append(f"{pf}:{fl} {node.name} 复杂度{cc}>{cc_max}")
if hits:
    print("\n".join(hits[:10])); sys.exit(1)
PYCC
    rc=$?
    if [ "$rc" -ne 0 ]; then
      fail "改动函数圈复杂度超标(>$CC_MAX，仅查改动函数)"
    else
      pass "圈复杂度检查通过(radon,≤${CC_MAX},仅查改动函数)"
    fi
  elif has eslint && [ "${#js_files[@]}" -gt 0 ]; then
    pass "圈复杂度交由 eslint complexity 规则（max $CC_MAX）"
  else
    warn "未安装 radon/eslint，圈复杂度未自动校验（设 RADON_BIN 或 pip install radon）"
  fi
}
check_cc

# ---------- 3. Lint（按栈探测） ----------
if [ "${#py_files[@]}" -gt 0 ] && has "$RUFF_BIN"; then
  if "$RUFF_BIN" check "${py_files[@]}" >/dev/null 2>&1; then pass "ruff 通过";
  else fail "ruff 检测到问题（$RUFF_BIN check 查看详情）"; fi
fi
if [ "${#js_files[@]}" -gt 0 ] && has eslint; then
  if eslint "${js_files[@]}" >/dev/null 2>&1; then pass "eslint 通过";
  else fail "eslint 检测到问题"; fi
fi

# ---------- 4. 测试覆盖率（仅 L1/L2 强制；工具缺失转人工确认） ----------
run_coverage() {
  [ "${LEVEL}" = "L0" ] && { warn "L0 不强制覆盖率门禁（建议≥${COV_CORE}%）"; return; }
  # 向上搜索测试配置位置（项目根 或 backend/ 等子目录）
  local cfgdir=""
  for cand in "." "./backend" "./server" "./api"; do
    if [ -f "$cand/pytest.ini" ] || [ -f "$cand/pyproject.toml" ] || [ -f "$cand/setup.cfg" ]; then
      cfgdir="$cand"; break; fi
  done
  if [ -z "$cfgdir" ]; then warn "无 Python 测试配置，覆盖率门禁转人工确认"; return; fi
  if ! has "$PYTEST_BIN"; then warn "未安装 pytest，覆盖率门禁转人工确认（设 PYTEST_BIN）"; return; fi
  # 在配置目录跑测试（cwd 不变，用 pytest 的 rootdir 推断）
  # COVERAGE_HARD_GATE=1 时覆盖率不达标才硬阻断；默认 warn 转人工（适配存量项目）
  # 测试范围：默认只跑改动文件关联的测试(PYTEST_SCOPE=changed)；FULL_GATE=1 跑全量(留给CI)
  local hard="${COVERAGE_HARD_GATE:-0}"
  local scope="${PYTEST_SCOPE:-changed}"
  local pytest_args=()
  if [ "$scope" = "changed" ]; then
    # 为每个改动源文件找对应测试: <path>/<mod>.py -> tests/**/test_<mod>.py
    local rel tmods=()
    for ff in "${FILES[@]}"; do
      rel="${ff#"$ROOT/"}"; rel="${rel#./}"
      case "$rel" in *test*|*spec*) continue;; esac
      local base="${rel##*/}"; base="${base%.py}"
      [ "$base" = "__init__" ] && continue
      while IFS= read -r t; do tmods+=("$t"); done < <(find "$cfgdir" -path '*/tests/*' -name "test_${base}.py" 2>/dev/null)
    done
    if [ "${#tmods[@]}" -gt 0 ]; then pytest_args=("${tmods[@]}");
    else warn "改动文件无对应单元测试，覆盖率门禁转人工确认（建议补 test_<mod>.py）"; return; fi
  fi
  if "$PYTEST_BIN" "${pytest_args[@]}" -q --cov --cov-report=term-missing >/tmp/qgate_pytest.log 2>&1; then
    local cov; cov=$(grep -E '^TOTAL' /tmp/qgate_pytest.log | tail -1 | grep -oE '[0-9]+%' | head -1 | tr -d '%')
    if [ -n "$cov" ] && [ "$cov" -lt "${COV_CORE}" ] 2>/dev/null; then
      if [ "$hard" = "1" ]; then fail "覆盖率 ${cov}% < 阈值 ${COV_CORE}%（等级 $LEVEL）"
      else warn "覆盖率 ${cov}% < 阈值 ${COV_CORE}%（默认警告；COVERAGE_HARD_GATE=1 硬阻断）"; fi
    else
      pass "pytest+coverage 通过 (覆盖 ${cov:-?}% ≥ ${COV_CORE}%, scope=$scope)"
    fi
  else
    # 测试失败：硬阻断（测试不过绝不能放行）
    fail "pytest 失败或测试未通过（见 /tmp/qgate_pytest.log, scope=$scope）"
  fi
}
run_coverage

# ---------- 5. SAST（L2 强制；误报需闭环记录） ----------
if [ "${LEVEL}" = "L2" ]; then
  if has "$BANDIT_BIN"; then
    if "$BANDIT_BIN" -r -q "${FILES[@]}" >/tmp/qgate_bandit.log 2>&1; then pass "bandit SAST 通过";
    else
      highs=$(grep -c 'severity: HIGH' /tmp/qgate_bandit.log 2>/dev/null || echo 0)
      if [ "$highs" -gt 0 ]; then fail "bandit 发现 $highs 个高危（误报须走豁免闭环: 加 # nosec 并登记）";
      else warn "bandit 仅中低危告警，需人工确认"; fi
    fi
  else warn "L2 建议安装 bandit 做静态安全扫描（设 BANDIT_BIN）"; fi
fi

# ---------- 6. 变异测试（L0 强制；L1/L2 可选,抓逻辑缺陷,门禁抓不到的）----------
# 静态门禁(ruff/radon/bandit)不做路径敏感分析,逻辑缺陷(除零/None/边界)只能靠测试覆盖。
# 变异测试自动注入缺陷验证测试套件有效性——这是"L0 免逐行审核"成立的科学依据。
MUTMIN="${MUTATION_MIN:-80}"
run_mutation() {
  # L0 强制; L1/L2 默认跳过(MUTATION_GATE=1 开启)
  if [ "$LEVEL" != "L0" ] && [ "${MUTATION_GATE:-0}" != "1" ]; then return; fi
  # 找改动文件对应的测试 + 用 governance 自带 mutation-check.sh(不依赖 mutmut)
  local target="${py_files[0]:-}"
  [ -z "$target" ] && { warn "无 py 改动文件,变异测试跳过"; return; }
  # 找对应测试文件
  local base="${target##*/}"; base="${base%.py}"
  local tfile=""
  for cand in "backend" "." "./backend"; do
    local found="$(find "$cand" -path '*/tests/*' -name "test_${base}.py" 2>/dev/null | head -1)"
    [ -n "$found" ] && { tfile="$found"; break; }
  done
  [ -z "$tfile" ] && { warn "改动文件 $base 无对应测试,变异测试跳过(逻辑缺陷门禁抓不到)"; return; }
  # 定位 mutation-check.sh(转绝对路径, 因后续可能 cd)
  local mcheck=""
  for cand in "$KIT_GUARD_DIR/mutation-check.sh" "$(dirname "$0")/mutation-check.sh" "$PWD/scripts/ai-verify/mutation-check.sh"; do
    if [ -f "$cand" ]; then
      case "$cand" in /*) mcheck="$cand";; *) mcheck="$PWD/$cand";; esac
      break
    fi
  done
  [ -z "$mcheck" ] && { warn "未找到 mutation-check.sh,变异测试跳过"; return; }
  # 把 venv/bin 加入 PATH(若 RADON_BIN 来自 venv), 让 mutation-check 用短名 pytest
  local savedpath="$PATH"
  [ -n "${RADON_BIN:-}" ] && [ -d "$(dirname "${RADON_BIN}")" ] && export PATH="$(dirname "${RADON_BIN}"):$PATH"
  say "运行变异测试(抽样: $base, 阈值≥${MUTMIN}%)..."
  local pybin="${PYTHON_BIN:-python3}"
  # 在源文件所在目录跑(那里通常有 pyproject/pytest 配置); 路径转绝对
  local abs_src abs_tst
  case "$target" in /*) abs_src="$target";; *) abs_src="$PWD/$target";; esac
  case "$tfile" in /*) abs_tst="$tfile";; *) abs_tst="$PWD/$tfile";; esac
  local srcdir="$PWD"; local _d="$(dirname "$abs_tst")"
  while [ "$_d" != "/" ]; do [ -f "$_d/pyproject.toml" ] || [ -f "$_d/conftest.py" ] && { srcdir="$_d"; break; }; _d="$(dirname "$_d")"; done
  cd "$srcdir"
    MUTATION_MIN="$MUTMIN" PYTHON_BIN="$pybin" bash "$mcheck" "$abs_src" "$abs_tst" "pytest --override-ini=addopts=" >/tmp/qgate_mut.log 2>&1
  local rc=$?
  cd "$ROOT"
  export PATH="$savedpath"
  local ms; ms="$(grep -oE 'MUTATION_SCORE=[0-9NA]+' /tmp/qgate_mut.log | cut -d= -f2)"
  if [ "$rc" -eq 0 ]; then
    pass "变异测试通过(${ms}% ≥ ${MUTMIN}%, L0 可免逐行审核的科学依据)"
  elif [ "$rc" -eq 1 ]; then
    if [ "$LEVEL" = "L0" ] && [ "${MUTATION_HARD_GATE:-0}" = "1" ]; then
      fail "变异分数 ${ms}% < ${MUTMIN}%(L0 免审前提不成立,需补测试)"
    else
      warn "变异分数 ${ms}% < ${MUTMIN}%(测试覆盖有盲区,建议补测试后再免审)"
      grep -E '^  - 存活' /tmp/qgate_mut.log | head -5 | while read -r l; do say "   $l"; done
    fi
  else
    # rc=2: 环境错误(基准失败/路径问题), 不判通过,转人工
    warn "变异测试环境错误(rc=$rc): $(tail -1 /tmp/qgate_mut.log 2>/dev/null)"
    warn "L0 无法自动验证测试有效性,建议人工抽审边界"
  fi
}
run_mutation

# ---------- 汇总 ----------
echo >> "$REPORT"
if [ "$FAIL" -eq 1 ]; then
  say "门禁未通过（等级 ${LEVEL}）→ 废弃/重试（重试上限 MAX_RETRY=3,超限转人工）"
  echo "**结论：❌ 未通过**" >> "$REPORT"
  exit 1
fi
if [ "$WARN" -eq 1 ]; then
  say "门禁通过（等级 ${LEVEL}，含警告，需人工确认警告项）"
  echo "**结论：⚠️ 通过(含警告)**" >> "$REPORT"
  exit 0
fi
say "门禁全部通过（等级 ${LEVEL}）"
echo "**结论：✅ 通过**" >> "$REPORT"
echo "报告: $REPORT"
exit 0
