const { Document, Packer, Paragraph, TextRun, Table, TableRow, TableCell, WidthType, BorderStyle, AlignmentType, HeadingLevel, ShadingType } = require("docx");
const fs = require("fs");

// Helper: create a styled heading
function heading(text, level) {
  return new Paragraph({ text, heading: level, spacing: { before: 300, after: 150 } });
}

// Helper: create a normal paragraph
function para(text, opts = {}) {
  const runs = [];
  if (opts.bold) {
    runs.push(new TextRun({ text, bold: true, size: opts.size || 22, font: "Microsoft YaHei" }));
  } else {
    runs.push(new TextRun({ text, size: opts.size || 22, font: "Microsoft YaHei" }));
  }
  return new Paragraph({ children: runs, spacing: { after: 100 }, alignment: opts.align });
}

// Helper: bullet point
function bullet(text) {
  return new Paragraph({
    children: [new TextRun({ text, size: 22, font: "Microsoft YaHei" })],
    bullet: { level: 0 },
    spacing: { after: 50 },
  });
}

// Helper: sub-bullet
function subBullet(text) {
  return new Paragraph({
    children: [new TextRun({ text, size: 22, font: "Microsoft YaHei" })],
    bullet: { level: 1 },
    spacing: { after: 50 },
  });
}

// Shared border style
const borders = {
  top: { style: BorderStyle.SINGLE, size: 1 },
  bottom: { style: BorderStyle.SINGLE, size: 1 },
  left: { style: BorderStyle.SINGLE, size: 1 },
  right: { style: BorderStyle.SINGLE, size: 1 },
};

// Helper: create a table
function makeTable(headers, rows) {
  const headerShading = { type: ShadingType.SOLID, color: "2B579A" };
  const headerCells = headers.map(h => new TableCell({
    children: [new Paragraph({ children: [new TextRun({ text: h, bold: true, size: 20, font: "Microsoft YaHei", color: "FFFFFF" })], alignment: AlignmentType.CENTER })],
    shading: headerShading,
    borders,
    width: { size: Math.floor(9000 / headers.length), type: WidthType.DXA },
  }));

  const dataRows = rows.map((row, ri) => {
    const rowShading = ri % 2 === 0 ? { type: ShadingType.SOLID, color: "F2F2F2" } : undefined;
    return new TableRow({
      children: row.map(cell => new TableCell({
        children: [new Paragraph({ children: [new TextRun({ text: String(cell), size: 20, font: "Microsoft YaHei" })], alignment: AlignmentType.CENTER })],
        borders,
        shading: rowShading,
        width: { size: Math.floor(9000 / headers.length), type: WidthType.DXA },
      })),
    });
  });

  return new Table({
    rows: [new TableRow({ children: headerCells }), ...dataRows],
    width: { size: 9000, type: WidthType.DXA },
  });
}

async function main() {
  const children = [];

  // Title
  children.push(new Paragraph({
    children: [new TextRun({ text: "su-memory SDK 全量测试报告 v2.0", bold: true, size: 36, font: "Microsoft YaHei", color: "1F4E79" })],
    alignment: AlignmentType.CENTER,
    spacing: { after: 100 },
  }));
  children.push(new Paragraph({
    children: [new TextRun({ text: "周易AI记忆引擎 · 测试日期：2026-04-23", size: 24, font: "Microsoft YaHei", color: "666666" })],
    alignment: AlignmentType.CENTER,
    spacing: { after: 400 },
  }));

  // Section 1: Overview
  children.push(heading("一、测试概览", HeadingLevel.HEADING_1));
  children.push(makeTable(
    ["指标", "数值"],
    [
      ["总用例数", "297"],
      ["通过", "297"],
      ["失败", "0"],
      ["通过率", "100%"],
      ["测试环境", "macOS Darwin 26.4.1 / Python 3.11.15"],
      ["测试框架", "pytest 9.0.3"],
      ["总耗时", "~160s（含性能基准测试）"],
    ]
  ));

  // Section 2: Test Suites
  children.push(heading("二、测试套件详情", HeadingLevel.HEADING_1));

  // 2.1
  children.push(heading("2.1 Yi 核心层测试（155/155 passed, 2.87s）", HeadingLevel.HEADING_2));
  children.push(para("测试文件：tests/test_yi_core_comprehensive.py", { bold: true }));
  children.push(para("覆盖模块：", { bold: true }));
  children.push(bullet("su_core/_sys/encoders.py（524行）— 语义编码器"));
  children.push(bullet("su_core/_sys/_c1.py（270行）— 一层核心：八卦推断、全息映射"));
  children.push(bullet("su_core/_sys/_c2.py（188行）— 二层核心：增强检索、融合评分"));
  children.push(bullet("su_core/_sys/fusion.py（252行）— 五维连续融合检索"));
  children.push(bullet("su_core/_sys/chrono.py（438行）— 干支时序系统"));
  children.push(bullet("su_core/_sys/causal.py（600行）— 因果推理引擎"));
  children.push(bullet("su_core/_sys/yijing.py（569行）— 周易卦爻推理引擎"));
  children.push(para("关键验证项：", { bold: true }));
  children.push(bullet("语义编码：向量生成、维度正确性、批量编码一致性"));
  children.push(bullet("八卦推断：8卦分类准确性、概率分布归一化"));
  children.push(bullet("五行生克：金木水火土五态完整性、相生相克关系"));
  children.push(bullet("全息检索：64卦空间映射、top-k排序"));
  children.push(bullet("时序计算：干支纪年、月令旺相、六十甲子编码"));
  children.push(bullet("因果推理：多层因果链遍历、能量传播"));
  children.push(bullet("周易卦爻：卦象推理、爻辞关联、变卦计算"));
  children.push(bullet("公共API：所有导出符号可正常import"));

  // 2.2
  children.push(heading("2.2 SDK 功能测试（28/28 passed, 2.67s）", HeadingLevel.HEADING_2));
  children.push(para("测试文件：tests/test_sdk.py", { bold: true }));
  children.push(bullet("核心CRUD（14项）：add/query/delete/link/stats 全流程"));
  children.push(bullet("边界条件（7项）：空查询、超长内容、Unicode、特殊字符等"));
  children.push(bullet("八卦五行集成（4项）：枚举完整性、映射验证"));
  children.push(bullet("因果链（3项）：100%覆盖率、因果关联验证"));

  // 2.3
  children.push(heading("2.3 四位一体集成测试（6/6 passed, 2.28s）", HeadingLevel.HEADING_2));
  children.push(para("测试文件：tests/test_yiti_integration.py", { bold: true }));
  children.push(bullet("记忆创建含四位一体编码（语义+八卦+五行+全息）"));
  children.push(bullet("多层因果链构建与遍历"));
  children.push(bullet("元认知信念追踪"));
  children.push(bullet("动态优先级（干支时序调节）"));
  children.push(bullet("端到端集成流程"));
  children.push(bullet("跨模块一致性验证"));

  // 2.4
  children.push(heading("2.4 Hindsight 对标测试（12/12 passed, 6.58s）", HeadingLevel.HEADING_2));
  children.push(para("对标 Hindsight 91.4% LongMemEval 的 12 个场景：", { bold: true }));
  children.push(makeTable(
    ["#", "场景", "说明", "结果"],
    [
      ["1", "单跳检索", "直接语义匹配", "PASS"],
      ["2", "多跳推理", "跨记忆关联推理", "PASS"],
      ["3", "时序理解", "时间相关记忆排序", "PASS"],
      ["4", "多会话", "跨会话记忆持久化", "PASS"],
      ["5", "开放域", "非结构化自由文本", "PASS"],
      ["6", "全息检索", "vs 纯向量检索对比", "PASS"],
      ["7", "象压缩", "语义保真压缩", "PASS"],
      ["8", "五行因果推理", "因果链覆盖率", "PASS"],
      ["9", "干支动态优先级", "季节性权重调节", "PASS"],
      ["10", "元认知", "认知间隙发现", "PASS"],
      ["11", "可解释性", "检索结果解释", "PASS"],
      ["12", "对比报告", "综合对比生成", "PASS"],
    ]
  ));

  // 2.5
  children.push(heading("2.5 性能基准测试（14/14 passed, 141.70s）", HeadingLevel.HEADING_2));
  children.push(bullet("延迟基准（5项）：编码/全息/压缩/SDK写入/SDK检索延迟"));
  children.push(bullet("吞吐量基准（5项）：编码/全息/SDK写入/SDK检索/混合读写 QPS"));
  children.push(bullet("资源占用（2项）：1K/10K条记忆内存"));
  children.push(bullet("扩展性（2项）：检索/写入延迟非线性增长"));

  // 2.6
  children.push(heading("2.6 象压缩测试（21/21 passed, 0.04s）", HeadingLevel.HEADING_2));
  children.push(bullet("压缩率验证（4项）：医疗文本、通用对话、结构化数据、短文本"));
  children.push(bullet("保真度验证（3项）：无损保真、语义保留、zlib对比"));
  children.push(bullet("极端输入（7项）：纯数字、纯符号、中英混合、空字符串等"));
  children.push(bullet("性能与结构验证（7项）"));

  // 2.7
  children.push(heading("2.7 元认知测试（34/34 passed, 0.02s）", HeadingLevel.HEADING_2));
  children.push(bullet("认知间隙检测（3项）"));
  children.push(bullet("知识老化检测（3项）"));
  children.push(bullet("信念生命周期（8项）"));
  children.push(bullet("因果链（10项）"));
  children.push(bullet("阶段分布与复杂因果链（2项）"));

  // 2.8
  children.push(heading("2.8 Phase 2 验证测试（27/27 passed, 3.52s）", HeadingLevel.HEADING_2));
  children.push(makeTable(
    ["Task", "测试数", "验证内容"],
    [
      ["Task 12", "5", "语义编码器、八卦软推断、全息连续性、向后兼容"],
      ["Task 13", "5", "融合权重、五行五态、五行相似度、融合检索"],
      ["Task 14", "4", "因果推理、多跳检索、周易推理"],
      ["Task 15", "6", "月令五行态、甲子位置、时序相似度、时间衰减"],
      ["Task 16", "7", "增删查、语义质量、性能、索引结构、多跳"],
    ]
  ));

  // Section 3: Phase 2 Summary
  children.push(heading("三、Phase 2 改造总结", HeadingLevel.HEADING_1));

  children.push(heading("3.1 Task 12：语义编码重建", HeadingLevel.HEADING_2));
  children.push(para("MD5 hash → sentence-transformers 语义向量 + 四位一体投影", { bold: true }));
  children.push(bullet("SemanticEncoder 基于 sentence-transformers 生成真实语义向量"));
  children.push(bullet("八卦推断从硬编码规则改为概率分布软推断"));
  children.push(bullet("全息映射从离散ID改为连续评分空间"));
  children.push(bullet("encoders.py：524行 | _c1.py：270行"));

  children.push(heading("3.2 Task 13：多维融合检索", HeadingLevel.HEADING_2));
  children.push(para("6路硬匹配 → 5维连续融合评分", { bold: true }));
  children.push(makeTable(
    ["维度", "权重", "说明"],
    [
      ["semantic", "0.40", "语义向量余弦相似度"],
      ["bagua", "0.15", "八卦概率分布KL散度"],
      ["wuxing", "0.15", "五行生克关系评分"],
      ["holographic", "0.15", "全息空间距离评分"],
      ["causal", "0.15", "因果关联强度评分"],
    ]
  ));

  children.push(heading("3.3 Task 14：推理能力释放", HeadingLevel.HEADING_2));
  children.push(para("新增 CausalInference + YiJingInference + 多跳检索", { bold: true }));
  children.push(bullet("因果推理引擎：五行生克因果链，支持多层传播（causal.py 600行）"));
  children.push(bullet("周易推理引擎：卦象变换推理、爻辞关联（yijing.py 569行）"));
  children.push(bullet("多跳检索：query_multi_hop 跨记忆关联推理"));

  children.push(heading("3.4 Task 15：干支时序全集成", HeadingLevel.HEADING_2));
  children.push(para("完整月令旺相 + 六十甲子编码 + 非线性衰减", { bold: true }));
  children.push(bullet("完整六十甲子编码（天干地支组合）"));
  children.push(bullet("月令旺相系统（春木旺/夏火旺/秋金旺/冬水旺）"));
  children.push(bullet("节气精确计算与季节判定"));
  children.push(bullet("chrono.py 438行 | priority_boost.py 632行 | ganzhi.py 257行"));

  children.push(heading("3.5 Task 16：SDK 优化", HeadingLevel.HEADING_2));
  children.push(para("分桶索引替代线性扫描 + 语义向量核心评分", { bold: true }));
  children.push(bullet("O(n) 线性扫描 → 分桶索引检索"));
  children.push(bullet("100条记忆查询 P50 延迟：9.235ms"));
  children.push(bullet("client.py：336行"));

  // Section 4: Performance
  children.push(heading("四、性能指标", HeadingLevel.HEADING_1));

  children.push(heading("4.1 延迟指标（实测数据）", HeadingLevel.HEADING_2));
  children.push(makeTable(
    ["操作", "P50", "P95", "P99", "达标阈值"],
    [
      ["语义编码（encode）", "4.247ms", "5.740ms", "6.602ms", "P50<15ms ✅"],
      ["全息检索（holographic）", "0.280ms", "0.308ms", "0.314ms", "P50<5ms ✅"],
      ["象压缩（compress）", "0.020ms", "0.022ms", "0.025ms", "P50<10ms ✅"],
      ["SDK 写入（add）", "4.509ms", "6.349ms", "6.869ms", "P50<150ms ✅"],
      ["SDK 检索（query 100条）", "9.235ms", "9.650ms", "10.048ms", "P50<100ms ✅"],
    ]
  ));

  children.push(heading("4.2 吞吐量指标", HeadingLevel.HEADING_2));
  children.push(makeTable(
    ["操作", "并发数", "达标阈值", "错误率"],
    [
      ["编码吞吐量", "10", ">100 QPS ✅", "<0.1% ✅"],
      ["全息检索吞吐量", "10", ">500 QPS ✅", "<0.1% ✅"],
      ["SDK 写入吞吐量", "10", ">50 QPS ✅", "<0.1% ✅"],
      ["SDK 检索吞吐量", "10", ">20 QPS ✅", "<0.1% ✅"],
      ["混合读写吞吐量", "20", ">40 QPS ✅", "<0.5% ✅"],
    ]
  ));

  children.push(heading("4.3 资源占用与扩展性", HeadingLevel.HEADING_2));
  children.push(makeTable(
    ["指标", "达标"],
    [
      ["1K 条记忆内存", "< 800MB ✅"],
      ["10K 条记忆内存", "< 1.5GB ✅"],
      ["检索延迟 100→1K", "增幅 < 10x ✅"],
      ["写入延迟 100→1K", "增幅 < 5x ✅"],
    ]
  ));

  // Section 5: Code Scale
  children.push(heading("五、代码规模总览", HeadingLevel.HEADING_1));
  children.push(makeTable(
    ["文件", "行数", "职责"],
    [
      ["encoders.py", "524", "语义编码 + 四位一体投影"],
      ["_c1.py", "270", "八卦推断、全息映射"],
      ["_c2.py", "188", "增强检索、融合评分"],
      ["fusion.py", "252", "五维连续融合"],
      ["chrono.py", "438", "干支纪年、节气"],
      ["causal.py", "600", "因果推理引擎"],
      ["yijing.py", "569", "周易推理引擎"],
      ["priority_boost.py", "632", "优先级提升"],
      ["ganzhi.py", "257", "六十甲子编码"],
      ["codec.py", "672", "象压缩核心"],
      ["states.py", "214", "信念追踪"],
      ["awareness.py", "310", "认知间隙检测"],
      ["client.py", "336", "SDK 客户端"],
      ["其他 5 文件", "260", "导出/辅助"],
      ["合计 18 文件", "5,522", "—"],
    ]
  ));

  // Section 6: Conclusion
  children.push(heading("六、结论", HeadingLevel.HEADING_1));
  children.push(para("全量 297 测试用例 100% 通过，Phase 2 改造成功实现：", { bold: true }));
  children.push(new Paragraph({ children: [], spacing: { after: 50 } }));

  const conclusions = [
    "语义编码从无到有 — MD5 hash → sentence-transformers 真实语义向量",
    "检索从离散硬匹配到连续多维融合 — 5维加权融合 (semantic 0.40 + bagua/wuxing/holographic/causal 各 0.15)",
    "推理从无到因果+周易双引擎 — CausalInference + YiJingInference + 多跳检索",
    "时序从简化到精确节气+甲子 — 六十甲子编码 + 月令旺相 + 非线性衰减",
    "SDK 性能从 O(n) 到索引检索 — 分桶索引，100条记忆查询 P50 仅 9.2ms",
  ];
  conclusions.forEach((c, i) => {
    children.push(new Paragraph({
      children: [
        new TextRun({ text: `${i + 1}. `, bold: true, size: 22, font: "Microsoft YaHei" }),
        new TextRun({ text: c, size: 22, font: "Microsoft YaHei" }),
      ],
      spacing: { after: 80 },
    }));
  });

  children.push(new Paragraph({ children: [], spacing: { after: 200 } }));
  children.push(new Paragraph({
    children: [new TextRun({ text: "报告生成时间：2026-04-23 | su-memory SDK v2.0 | 周易AI记忆引擎", size: 20, font: "Microsoft YaHei", color: "999999", italics: true })],
    alignment: AlignmentType.CENTER,
  }));

  const doc = new Document({
    sections: [{ children }],
    styles: {
      default: {
        heading1: { run: { size: 32, bold: true, font: "Microsoft YaHei", color: "1F4E79" } },
        heading2: { run: { size: 26, bold: true, font: "Microsoft YaHei", color: "2B579A" } },
      },
    },
  });

  const buffer = await Packer.toBuffer(doc);
  const outPath = "/Users/mac/.openclaw/workspace/su-memory/su-memory-SDK测试报告-v2.docx";
  fs.writeFileSync(outPath, buffer);
  console.log(`DOCX report generated: ${outPath} (${(buffer.length / 1024).toFixed(1)} KB)`);
}

main().catch(err => { console.error(err); process.exit(1); });
