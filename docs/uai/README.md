# UAI 2027 投稿指南

## 文件夹内容
- `MCI_World_Model_v3.5.0_UAI.pdf` — PDF 论文（双栏紧凑版，约8页）
- `MCI_World_Model_v3.5.0_UAI.tex` — LaTeX 源文件

## 关于 UAI
UAI (Conference on Uncertainty in Artificial Intelligence) 是**因果推理领域 #1 会议**。
- 网站: https://www.auai.org/
- CCF B类，AI 领域顶级专业会议
- 你的 PC/LiNGAM/Granger 对比是其核心话题

## 投稿步骤

### 1. 获取 UAI 官方模板
UAI 2027 模板尚未发布（预计2026年底）。
当前使用 `article` 类 + `twocolumn` 近似 UAI 格式。
正式投稿时需替换为官方 `uai2027.sty`。

下载地址（发布后）：https://www.auai.org/uai2027/

### 2. 格式调整（正式投稿前）
```latex
\documentclass{uai2027}  % 替换 \documentclass[10pt,twocolumn]{article}
```
- 页面限制：8页正文 + 不限页数参考文献
- 匿名审稿：第一轮投稿需去除作者信息

### 3. 提交
1. 通过 CMT 系统提交（https://cmt3.research.microsoft.com/）
2. 选择主题：Causal Discovery / Causal Reasoning
3. 上传 PDF

## 时间线（预估）
- 截稿：2027年 1-2月
- 通知：2027年 4月
- 会议：2027年 7-8月

## 当前论文适配说明
已按 UAI 风格调整：
- 双栏紧凑排版
- 摘要浓缩至 6 行
- 相关工作大幅精简
- 表格紧凑化
- 参考文献精选 25 篇
