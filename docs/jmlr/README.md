# JMLR / TMLR 投稿指南

## 文件夹内容
- `MCI_World_Model_v3.5.0_JMLR.pdf` — PDF 论文（单栏扩展版）
- `MCI_World_Model_v3.5.0_JMLR.tex` — LaTeX 源文件

## 两个可选期刊

### JMLR (Journal of Machine Learning Research)
- 网站: https://jmlr.org/
- IF ~6.0，CCF A类，开放获取，无版面费
- 接受长文，无页数限制
- 审稿周期：3-6个月
- 偏好：算法+理论+实验完整论文

### TMLR (Transactions on Machine Learning Research)
- 网站: https://jmlr.org/tmlr/
- JMLR 姊妹刊，审稿更快（~2个月）
- 开放获取 + 开放审稿
- 接受系统+理论混合论文
- LeJEPA 有很多论文发在这里

## 投稿步骤

### 1. 获取 JMLR 模板
```bash
# JMLR 使用 jmlr2e 样式
# 下载: https://jmlr.org/format/jmlr2e.sty
```

### 2. 格式调整
```latex
\documentclass[jmlr2e]{article}  % 替换当前 documentclass
```
- JMLR 无页数限制，当前长文版可直接使用
- 需添加 \jmlrheading{}{2026}{}{}{Qiang Su}{}

### 3. 提交
- **JMLR**: https://jmlr.org/ → Submit
- **TMLR**: https://openreview.net/group?id=TMLR

## JMLR vs TMLR 选择建议
- 如果你希望论文作为**正式期刊论文**被引用 → JMLR
- 如果你希望**快速发表并获得开放审稿意见** → TMLR
- TMLR 审稿更快，但 JMLR 学术认可度略高

## 当前论文适配说明
已按 JMLR 风格调整：
- 单栏扩展排版（完整内容）
- 含目录导航
- 扩展版包含全部相关内容
- 补充了能量模型（LeCun 2006）引用
