#!/usr/bin/env bash
# 轻量变异测试器（governance 自带，跨平台，不依赖 mutmut）
# 原理: 对源文件逐条应用典型变异(运算符翻转/常量变更),跑测试
#   测试仍 PASS = 变异存活 = 测试覆盖有洞; FAIL = 变异被杀 = 测试有效
#   变异分数 = 被杀 / 有效变异总数 × 100%
# 用法: mutation-check.sh <源文件> <测试目标> [pytest前缀]
# 退出码: 0=达标 1=不达标 2=环境错误
set -o pipefail
SRC="${1:?用法: mutation-check.sh <源文件> <测试> [pytest]}"
TESTS="${2:?缺测试目标}"
PYTEST="${3:-pytest}"
PY="${PYTHON_BIN:-python3}"

[ -f "$SRC" ] || { echo "ERROR: 源文件不存在: $SRC"; exit 2; }

# run_test: PYTEST 含空格时用 sh -c
run_test(){ sh -c "PYTEST=\"$PYTEST\" TESTS=\"$TESTS\" \"\$PYTEST\" -x -q --no-header \"\$TESTS\"" >/dev/null 2>&1; }

echo "[mut] 基准测试..."
if ! sh -c "$PYTEST -x -q --no-header \"$TESTS\"" >/tmp/mut_base.log 2>&1; then
  echo "ERROR: 基准测试未通过,先修测试:"; tail -5 /tmp/mut_base.log; exit 2
fi
echo "[mut] 基准通过 ✓"

cp "$SRC" "$SRC.mutorig"
trap 'cp "$SRC.mutorig" "$SRC" 2>/dev/null; rm -f "$SRC.mutorig"' EXIT

# 变异规则: 描述|正则|替换。用 python 应用(单点替换第1处匹配)
apply_mut(){
  "$PY" - "$SRC" "$2" "$3" <<'PYMUT'
import sys, re, tokenize, io
path, pat, rep = sys.argv[1], sys.argv[2], sys.argv[3]
code = open(path, encoding="utf-8").read()
# 收集所有非注释/非字符串的代码区间,只在其中查找变异点
def code_regions(src):
    regions=[]; last=0
    try:
        for tok in tokenize.generate_tokens(io.StringIO(src).readline):
            if tok.type in (tokenize.COMMENT, tokenize.STRING):
                if tok.start[1] > last:
                    regions.append((last, tok.start[1]))  # 之前的是代码(简化:行级)
                last = tok.end[1]
    except tokenize.TokenError:
        pass
    return regions
# 简化: 用 tokenize 找出所有 STRING/COMMENT 的 (start_offset, end_offset), 排除它们
def get_skip_offsets(src):
    skip=[]; 
    try:
        for tok in tokenize.generate_tokens(io.StringIO(src).readline):
            if tok.type in (tokenize.COMMENT, tokenize.STRING):
                # 行列转偏移
                start_off = sum(len(l)+1 for l in src.splitlines(True)[:tok.start[0]-1]) + tok.start[1]
                end_off = sum(len(l)+1 for l in src.splitlines(True)[:tok.end[0]-1]) + tok.end[1]
                skip.append((start_off, end_off))
    except tokenize.TokenError:
        pass
    return skip
def in_skip(pos, skips):
    for s,e in skips:
        if s <= pos < e: return True
    return False
skips = get_skip_offsets(code)
try: rx = re.compile(pat)
except re.error:
    if pat not in code: sys.exit(2)
    new = code.replace(pat, rep, 1)
else:
    # 找第一个不在 skip 区间内的匹配
    m = rx.search(code)
    while m and in_skip(m.start(), skips):
        m = rx.search(code, m.end())
    if not m: sys.exit(2)
    try: sub = m.expand(rep)
    except (re.error, ValueError): sub = rep
    new = code[:m.start()] + sub + code[m.end():]
if new == code: sys.exit(2)
open(path, "w", encoding="utf-8").write(new)
PYMUT
}

MUTS=(
  "比较 == 变 !=|==|!="
  "比较 != 变 ==|!=|=="
  "比较 >= 变 >|>=|>"
  "比较 <= 变 <|<=|<"
  "比较 > 变 <|[^<>=!]>(?!=)|<"
  "比较 < 变 >|[^<>=!]<(?!=)|>"
  "布尔 True→False|True|False"
  "布尔 False→True|False|True"
  "None 变 0|\\bNone\\b|0"
  "数值常量 +1 (整数)|\\b(\\d+)\\b|__inc__"
)

killed=0; survived=0; applied=0; surv_list=""
for m in "${MUTS[@]}"; do
  desc="${m%%|*}"; rest="${m#*|}"; pat="${rest%%|*}"; rep="${rest##*|}"
  cp "$SRC.mutorig" "$SRC"
  # 数值+1 特殊处理: 替换成 n+1
  if [ "$rep" = "__inc__" ]; then
    "$PY" - "$SRC" "$pat" <<'PYINC'
import sys, re, tokenize, io
path=sys.argv[1]; pat=sys.argv[2]
code=open(path,encoding="utf-8").read()
def get_skip_offsets(src):
    skip=[]
    try:
        for tok in tokenize.generate_tokens(io.StringIO(src).readline):
            if tok.type in (tokenize.COMMENT, tokenize.STRING):
                start_off=sum(len(l)+1 for l in src.splitlines(True)[:tok.start[0]-1])+tok.start[1]
                end_off=sum(len(l)+1 for l in src.splitlines(True)[:tok.end[0]-1])+tok.end[1]
                skip.append((start_off,end_off))
    except tokenize.TokenError: pass
    return skip
def in_skip(pos,skips):
    for s,e in skips:
        if s<=pos<e: return True
    return False
skips=get_skip_offsets(code)
rx=re.compile(pat)
m=rx.search(code)
while m and in_skip(m.start(),skips):
    m=rx.search(code,m.end())
if not m: sys.exit(2)
n=int(m.group(1))
new=code[:m.start()]+f"({n}+1)"+code[m.end():]
if new==code: sys.exit(2)
open(path,"w",encoding="utf-8").write(new)
PYINC
    rc=$?
  else
    apply_mut "$desc" "$pat" "$rep"; rc=$?
  fi
  [ "$rc" -ne 0 ] && continue
  diff -q "$SRC.mutorig" "$SRC" >/dev/null 2>&1 && continue
  applied=$((applied+1))
  if sh -c "$PYTEST -x -q --no-header \"$TESTS\"" >/tmp/mut_run.log 2>&1; then
    survived=$((survived+1)); surv_list="$surv_list\n  - 存活: $desc"
  else
    killed=$((killed+1))
  fi
done

if [ "$applied" -eq 0 ]; then
  echo "[mut] 未产生有效变异"; echo "MUTATION_SCORE=N/A"; exit 0
fi
score=$((killed * 100 / applied))
echo "[mut] 杀死 $killed / 存活 $survived / 有效变异 $applied"
echo "[mut] 变异分数: ${score}%"
[ -n "$surv_list" ] && { echo "[mut] 存活变异(测试未捕获):"; printf "$surv_list\n"; }
echo "MUTATION_SCORE=$score"
MIN="${MUTATION_MIN:-80}"
if [ "$score" -ge "$MIN" ]; then echo "[mut] ✅ 达标(≥${MIN}%)"; exit 0
else echo "[mut] ❌ 不达标(<${MIN}%), 测试套件有覆盖盲区"; exit 1; fi
