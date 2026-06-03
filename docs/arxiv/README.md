# arXiv 投稿指南

## 文件夹内容
- `MCI_World_Model_v3.5.0_ARXIV.pdf` — PDF 论文
- `MCI_World_Model_v3.5.0_ARXIV.tex` — LaTeX 源文件

## 投稿步骤

### 1. 注册 arXiv 账号
访问 https://arxiv.org/register
- 需要学术机构邮箱（.edu）或邀请 endorsement
- 首次投稿可能需要 3-5 个已注册用户为你 endorsement

### 2. 准备投稿文件
```bash
# 编译 PDF（确保无错误）
pdflatex MCI_World_Model_v3.5.0_ARXIV.tex
pdflatex MCI_World_Model_v3.5.0_ARXIV.tex
pdflatex MCI_World_Model_v3.5.0_ARXIV.tex
```

### 3. 提交到 arXiv
1. 登录 https://arxiv.org/user
2. 点击 "START NEW SUBMISSION"
3. 选择分类：
   - **主分类**: Computer Science → Artificial Intelligence (cs.AI)
   - **交叉列表**: cs.LG, stat.ML
4. 上传 PDF 文件
5. 填写元数据
6. 提交 → 等待处理（1-3天公告）

## 注意事项
- 提交后约 24-48 小时出现在网站上
- 可随时提交修订版（revision）
- PDF 已去除页码（arXiv 会自动添加）
