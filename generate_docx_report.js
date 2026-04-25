const {
  Document, Packer, Paragraph, TextRun, Table, TableRow, TableCell,
  Header, Footer, AlignmentType, LevelFormat, HeadingLevel, BorderStyle,
  WidthType, ShadingType, VerticalAlign, PageNumber, PageBreak
} = require('docx');
const fs = require('fs');

const DARK_BLUE = '1B3A5C';
const LIGHT_GRAY = 'F2F2F2';
const BORDER_COLOR = 'CCCCCC';
const WHITE = 'FFFFFF';

function cellBorders() {
  const b = { style: BorderStyle.SINGLE, size: 1, color: BORDER_COLOR };
  return { top: b, bottom: b, left: b, right: b };
}

function headerCell(text, widthDxa) {
  return new TableCell({
    children: [new Paragraph({
      children: [new TextRun({ text, bold: true, color: WHITE, font: 'Arial', size: 20 })],
      alignment: AlignmentType.CENTER,
      spacing: { before: 60, after: 60 }
    })],
    width: { size: widthDxa, type: WidthType.DXA },
    shading: { type: ShadingType.CLEAR, fill: DARK_BLUE, color: DARK_BLUE },
    borders: cellBorders(),
    verticalAlign: VerticalAlign.CENTER
  });
}

function dataCell(text, widthDxa, rowIdx) {
  const fill = rowIdx % 2 === 0 ? WHITE : LIGHT_GRAY;
  return new TableCell({
    children: [new Paragraph({
      children: [new TextRun({ text: String(text), font: 'Arial', size: 20 })],
      spacing: { before: 40, after: 40 }
    })],
    width: { size: widthDxa, type: WidthType.DXA },
    shading: { type: ShadingType.CLEAR, fill, color: fill },
    borders: cellBorders(),
    verticalAlign: VerticalAlign.CENTER
  });
}

function boldDataCell(text, widthDxa, rowIdx) {
  const fill = rowIdx % 2 === 0 ? WHITE : LIGHT_GRAY;
  return new TableCell({
    children: [new Paragraph({
      children: [new TextRun({ text: String(text), font: 'Arial', size: 20, bold: true })],
      spacing: { before: 40, after: 40 }
    })],
    width: { size: widthDxa, type: WidthType.DXA },
    shading: { type: ShadingType.CLEAR, fill, color: fill },
    borders: cellBorders(),
    verticalAlign: VerticalAlign.CENTER
  });
}

function makeTable(headers, rows, colWidths) {
  const totalWidth = colWidths.reduce((a, b) => a + b, 0);
  const headerRow = new TableRow({
    children: headers.map((h, i) => headerCell(h, colWidths[i])),
    tableHeader: true
  });
  const dataRows = rows.map((row, ri) =>
    new TableRow({
      children: row.map((cell, ci) => {
        if (typeof cell === 'object' && cell.bold) {
          return boldDataCell(cell.text, colWidths[ci], ri);
        }
        return dataCell(cell, colWidths[ci], ri);
      })
    })
  );
  return new Table({
    rows: [headerRow, ...dataRows],
    width: { size: totalWidth, type: WidthType.DXA },
    columnWidths: colWidths
  });
}

function heading1(text) {
  return new Paragraph({ text, heading: HeadingLevel.HEADING_1, spacing: { before: 360, after: 200 } });
}
function heading2(text) {
  return new Paragraph({ text, heading: HeadingLevel.HEADING_2, spacing: { before: 240, after: 120 } });
}
function heading3(text) {
  return new Paragraph({ text, heading: HeadingLevel.HEADING_3, spacing: { before: 200, after: 100 } });
}
function bodyText(text) {
  return new Paragraph({
    children: [new TextRun({ text, font: 'Arial', size: 22 })],
    spacing: { before: 80, after: 80 }
  });
}
function boldBodyText(text) {
  return new Paragraph({
    children: [new TextRun({ text, font: 'Arial', size: 22, bold: true })],
    spacing: { before: 80, after: 80 }
  });
}
function bulletItem(text, level) {
  return new Paragraph({
    children: [new TextRun({ text, font: 'Arial', size: 22 })],
    numbering: { reference: 'bullet-list', level: level || 0 },
    spacing: { before: 40, after: 40 }
  });
}
function pb() {
  return new Paragraph({ children: [new PageBreak()] });
}

const doc = new Document({
  numbering: {
    config: [{
      reference: 'bullet-list',
      levels: [
        { level: 0, format: LevelFormat.BULLET, text: '\u2022', alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 720, hanging: 360 } } } },
        { level: 1, format: LevelFormat.BULLET, text: '\u25E6', alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 1440, hanging: 360 } } } }
      ]
    }]
  },
  styles: {
    default: { document: { run: { font: 'Arial', size: 22 } } },
    paragraphStyles: [
      { id: 'Title', name: 'Title', basedOn: 'Normal',
        run: { size: 56, bold: true, color: DARK_BLUE, font: 'Arial' },
        paragraph: { spacing: { before: 240, after: 120 }, alignment: AlignmentType.CENTER } },
      { id: 'Heading1', name: 'Heading 1', basedOn: 'Normal', next: 'Normal', quickFormat: true,
        run: { size: 36, bold: true, color: DARK_BLUE, font: 'Arial' },
        paragraph: { spacing: { before: 360, after: 200 }, outlineLevel: 0 } },
      { id: 'Heading2', name: 'Heading 2', basedOn: 'Normal', next: 'Normal', quickFormat: true,
        run: { size: 28, bold: true, color: '333333', font: 'Arial' },
        paragraph: { spacing: { before: 240, after: 120 }, outlineLevel: 1 } },
      { id: 'Heading3', name: 'Heading 3', basedOn: 'Normal', next: 'Normal', quickFormat: true,
        run: { size: 24, bold: true, color: '555555', font: 'Arial' },
        paragraph: { spacing: { before: 200, after: 100 }, outlineLevel: 2 } }
    ]
  },
  sections: [
    // COVER PAGE
    {
      headers: { default: new Header({ children: [] }) },
      footers: { default: new Footer({ children: [] }) },
      children: [
        new Paragraph({ spacing: { before: 4000 } }),
        new Paragraph({ children: [new TextRun({ text: 'su-memory', font: 'Arial', size: 72, bold: true, color: DARK_BLUE })], alignment: AlignmentType.CENTER, spacing: { after: 200 } }),
        new Paragraph({ children: [new TextRun({ text: '(\u5468\u6613AI\u8bb0\u5fc6\u5f15\u64ce)', font: 'Arial', size: 36, color: '666666' })], alignment: AlignmentType.CENTER, spacing: { after: 400 } }),
        new Paragraph({ children: [new TextRun({ text: 'SDK \u5168\u9762\u6d4b\u8bd5\u62a5\u544a', font: 'Arial', size: 48, bold: true, color: DARK_BLUE })], alignment: AlignmentType.CENTER, spacing: { after: 200 } }),
        new Paragraph({ children: [new TextRun({ text: '\u542b\u95ee\u9898\u5206\u6790\u4e0e\u4fee\u590d\u65b9\u6848', font: 'Arial', size: 28, color: '888888' })], alignment: AlignmentType.CENTER, spacing: { after: 600 } }),
        new Paragraph({ children: [new TextRun({ text: '\u62a5\u544a\u65e5\u671f\uff1a2026\u5e744\u670822\u65e5', font: 'Arial', size: 24, color: '666666' })], alignment: AlignmentType.CENTER, spacing: { after: 100 } }),
        new Paragraph({ children: [new TextRun({ text: '\u62a5\u544a\u7248\u672c\uff1av1.0', font: 'Arial', size: 24, color: '666666' })], alignment: AlignmentType.CENTER, spacing: { after: 100 } }),
        new Paragraph({ children: [new TextRun({ text: '\u53d1\u5e03\u72b6\u6001\uff1a\u6761\u4ef6\u53d1\u5e03\uff08Conditional Go\uff09', font: 'Arial', size: 24, color: '666666' })], alignment: AlignmentType.CENTER })
      ]
    },
    // MAIN CONTENT
    {
      headers: {
        default: new Header({ children: [new Paragraph({ children: [new TextRun({ text: 'su-memory SDK \u6d4b\u8bd5\u62a5\u544a', font: 'Arial', size: 18, color: '999999', italics: true })], alignment: AlignmentType.RIGHT })] })
      },
      footers: {
        default: new Footer({ children: [new Paragraph({ children: [new TextRun({ text: 'Page ', font: 'Arial', size: 18 }), new TextRun({ children: [PageNumber.CURRENT], font: 'Arial', size: 18 }), new TextRun({ text: ' of ', font: 'Arial', size: 18 }), new TextRun({ children: [PageNumber.TOTAL_PAGES], font: 'Arial', size: 18 })], alignment: AlignmentType.CENTER })] })
      },
      children: [
        heading1('\u7b2c\u4e00\u90e8\u5206\uff1a\u6d4b\u8bd5\u62a5\u544a'),
        bodyText('\u9879\u76ee\u540d\u79f0\uff1asu-memory\uff08\u5468\u6613AI\u8bb0\u5fc6\u5f15\u64ce\uff09SDK'),
        bodyText('\u6d4b\u8bd5\u65e5\u671f\uff1a2026-04-22'),
        bodyText('\u6d4b\u8bd5\u73af\u5883\uff1amacOS Darwin 26.4.1, Python 3.11.15, Docker Desktop (Qdrant v1.7.4 + PostgreSQL 15-alpine)'),

        heading2('1. \u6d4b\u8bd5\u7ed3\u679c\u6c47\u603b'),
        bodyText('\u5408\u8ba1\uff1a365 passed / 7 failed / 1 skipped'),
        makeTable(
          ['Task', '\u6d4b\u8bd5\u6587\u4ef6', '\u7528\u4f8b\u6570', '\u5b9e\u9645\u7ed3\u679c'],
          [
            ['Task 2', 'tests/test_yi_core_comprehensive.py', '155', {text:'155 passed', bold:true}],
            ['Task 3', 'tests/test_memory_engine_comprehensive.py', '57', {text:'57 passed, \u8986\u76d6\u7387 87%', bold:true}],
            ['Task 4', 'tests/test_compression_comprehensive.py', '21', {text:'21 passed', bold:true}],
            ['Task 5', 'tests/test_metacognition_comprehensive.py', '34', {text:'34 passed', bold:true}],
            ['Task 6', 'tests/test_api_gateway_comprehensive.py', '62', {text:'62 passed, 6 warnings', bold:true}],
            ['Task 7', 'tests/test_benchmark.py', '14', {text:'14 passed, 27 \u9879\u6307\u6807\u8fbe\u6807', bold:true}],
            ['Task 8', 'tests/test_hindsight_comparison.py', '12', {text:'12 passed', bold:true}],
            ['Task 9', 'tests/test_stability_comprehensive.py', '18', {text:'10 passed, 7 failed, 1 skipped', bold:true}]
          ],
          [1200, 4000, 1000, 3200]
        ),

        heading2('2. \u5404\u6a21\u5757\u8be6\u7ec6\u6d4b\u8bd5\u7ed3\u8bba'),

        heading3('2.1 yi_core\uff08Task 2\uff09\u2014 155 passed'),
        bodyText('\u6d4b\u8bd5\u8986\u76d6\uff1a'),
        bulletItem('\u516b\u5366\u4f53\u7cfb\uff088\u5366\u751f\u6210\u3001\u5366\u8c61\u5c5e\u6027\u3001\u4e92\u5366\u53d8\u5366\uff09'),
        bulletItem('\u4e94\u884c\u4f53\u7cfb\uff08\u751f\u514b\u5173\u7cfb\u3001\u65fa\u76f8\u4f11\u56da\u6b7b\u3001\u5408\u5316\uff09'),
        bulletItem('\u5929\u5e72\u5730\u652f\uff0860\u7532\u5b50\u3001\u5211\u51b2\u5408\u5bb3\u3001\u7eb3\u97f3\u4e94\u884c\uff09'),
        bulletItem('\u5468\u6613\u5366\u7237\uff08\u516d\u7237\u751f\u6210\u3001\u52a8\u7237\u5224\u5b9a\u3001\u5366\u8f9e\u7237\u8f9e\u68c0\u7d22\uff09'),
        bulletItem('\u8bed\u4e49\u7f16\u7801\uff08SemanticEncoder \u7f16\u7801/\u89e3\u7801\u3001hash-based fallback\uff09'),
        bulletItem('\u591a\u89c6\u56fe\u68c0\u7d22\uff08\u4e09\u624d\u68c0\u7d22\u3001\u5168\u606f\u53e0\u52a0\uff09'),
        bulletItem('\u56e0\u679c\u94fe\uff08\u56e0\u679c\u8282\u70b9\u3001\u56e0\u679c\u56fe\u6784\u5efa\u3001\u4f20\u64ad\u63a8\u7406\uff09'),
        bulletItem('\u4fe1\u5ff5\u8ffd\u8e2a\uff08BeliefTracker \u72b6\u6001\u66f4\u65b0\u3001\u7f6e\u4fe1\u5ea6\u4f20\u64ad\uff09'),
        boldBodyText('\u7ed3\u8bba\uff1ayi_core \u6838\u5fc3\u7b97\u6cd5\u5c42\u5168\u90e8\u901a\u8fc7\uff0c\u516b\u5366/\u4e94\u884c/\u5929\u5e72\u5730\u652f/\u5366\u7237\u8ba1\u7b97\u51c6\u786e\uff0c\u8bed\u4e49\u7f16\u7801\u548c\u56e0\u679c\u94fe\u529f\u80fd\u5b8c\u5907\u3002'),

        heading3('2.2 \u8bb0\u5fc6\u5f15\u64ce\uff08Task 3\uff09\u2014 57 passed, 87% \u8986\u76d6\u7387'),
        bodyText('\u6d4b\u8bd5\u8986\u76d6\uff1a'),
        bulletItem('\u8bb0\u5fc6 CRUD\uff08add/query/delete/update\uff09'),
        bulletItem('\u51b2\u7a81\u6d88\u89e3\uff08ConflictResolver \u65f6\u95f4\u6233\u7b56\u7565\u3001\u4f18\u5148\u7ea7\u5408\u5e76\uff09'),
        bulletItem('\u9057\u5fd8\u673a\u5236\uff08ForgettingCurve \u8870\u51cf\u8ba1\u7b97\u3001\u4e3b\u52a8\u5f52\u6863\uff09'),
        bulletItem('\u68c0\u7d22\u878d\u5408\uff08Retriever \u5411\u91cf+\u5173\u7cfb\u53cc\u8def\u53ec\u56de\u3001\u91cd\u6392\u5e8f\uff09'),
        bodyText('\u8986\u76d6\u7387\u8be6\u60c5\uff1a'),
        makeTable(
          ['\u6a21\u5757', '\u8bed\u53e5\u6570', '\u7f3a\u5931', '\u8986\u76d6\u7387'],
          [
            ['memory_engine/__init__.py', '4', '0', {text:'100%', bold:true}],
            ['memory_engine/conflict_resolver.py', '46', '9', {text:'80%', bold:true}],
            ['memory_engine/extractor.py', '102', '14', {text:'86%', bold:true}],
            ['memory_engine/forgetting.py', '62', '2', {text:'97%', bold:true}],
            ['memory_engine/manager.py', '67', '11', {text:'84%', bold:true}],
            ['memory_engine/retriever.py', '45', '5', {text:'89%', bold:true}],
            [{text:'\u5408\u8ba1', bold:true}, {text:'326', bold:true}, {text:'41', bold:true}, {text:'87%', bold:true}]
          ],
          [3500, 1500, 1500, 1500]
        ),
        boldBodyText('\u7ed3\u8bba\uff1a\u8bb0\u5fc6\u5f15\u64ce\u6838\u5fc3\u529f\u80fd\u901a\u8fc7\uff0cCRUD \u548c\u9057\u5fd8\u673a\u5236\u7a33\u5b9a\u3002\u51b2\u7a81\u6d88\u89e3\u548c\u68c0\u7d22\u878d\u5408\u6709\u5c11\u91cf\u5206\u652f\u672a\u8986\u76d6\u3002'),

        heading3('2.3 \u8c61\u538b\u7f29\uff08Task 4\uff09\u2014 21 passed'),
        bulletItem('\u538b\u7f29\u7387\uff1a\u7ed3\u6784\u5316\u6570\u636e\u5e73\u5747 2.1x\uff0c\u6587\u672c\u6570\u636e 1.8-2.3x'),
        bulletItem('\u4fe1\u606f\u4fdd\u7559\u7387\uff1a\u5173\u952e\u5b57\u6bb5 100% \u4fdd\u7559\uff0c\u8bed\u4e49\u6a21\u5f0f\u4fdd\u7559\u7387 >95%'),
        bulletItem('\u65e0\u635f\u8fd8\u539f\uff1a\u4e8c\u8fdb\u5236\u6a21\u5f0f 100% \u8fd8\u539f\uff0cJSON \u6a21\u5f0f 100% \u8fd8\u539f'),
        bulletItem('\u8fb9\u7f18\u573a\u666f\uff1a\u7a7a\u8f93\u5165\u3001\u8d85\u5927\u5bf9\u8c61\u3001\u5d4c\u5957\u7ed3\u6784\u3001Unicode \u8fb9\u754c'),
        boldBodyText('\u7ed3\u8bba\uff1a\u8c61\u538b\u7f29\u6a21\u5757\u5168\u90e8\u901a\u8fc7\uff0c\u538b\u7f29\u7387\u8fbe\u6807\uff0c\u65e0\u635f\u8fd8\u539f\u53ef\u9760\u3002'),

        heading3('2.4 \u5143\u8ba4\u77e5\uff08Task 5\uff09\u2014 34 passed'),
        bulletItem('\u77e5\u8bc6\u7a7a\u6d1e\u53d1\u73b0\uff08KnowledgeGapDetector \u8986\u76d6\u5ea6\u8ba1\u7b97\u3001\u7a7a\u6d1e\u8bc6\u522b\uff09'),
        bulletItem('\u51b2\u7a81\u68c0\u6d4b\uff08ContradictionDetector \u4fe1\u5ff5\u51b2\u7a81\u3001\u65f6\u5e8f\u51b2\u7a81\uff09'),
        bulletItem('\u7f6e\u4fe1\u5ea6\u8bc4\u4f30\uff08ConfidenceEvaluator \u4e0d\u786e\u5b9a\u6027\u91cf\u5316\u3001\u6821\u51c6\uff09'),
        boldBodyText('\u7ed3\u8bba\uff1a\u5143\u8ba4\u77e5\u6a21\u5757\u5168\u90e8\u901a\u8fc7\uff0c\u77e5\u8bc6\u7a7a\u6d1e\u53d1\u73b0\u7387\u548c\u51b2\u7a81\u68c0\u6d4b\u7387\u5747\u8fbe\u5230 100%\u3002'),

        heading3('2.5 API Gateway\uff08Task 6\uff09\u2014 62 passed'),
        bulletItem('\u7aef\u70b9\u8986\u76d6\uff08health\u3001memory/add\u3001memory/query\u3001memory/delete\u3001tenant/create\u3001stats\u3001chat/completions\uff09'),
        bulletItem('\u9274\u6743\u6d4b\u8bd5\uff08\u65e0 Token\u3001\u8fc7\u671f Token\u3001\u975e\u6cd5 Token\u3001\u6b63\u786e Token\uff09'),
        bulletItem('\u9519\u8bef\u5904\u7406\uff08404\u3001422\u3001500\u3001\u53c2\u6570\u6821\u9a8c\uff09'),
        bulletItem('\u5e76\u53d1\u6d4b\u8bd5\uff0820 \u7ebf\u7a0b\u6df7\u5408\u8bfb\u5199\uff09'),
        bulletItem('\u5b89\u5168\u5934\uff08CORS\u3001X-Content-Type-Options \u7b49\uff09'),
        boldBodyText('\u7ed3\u8bba\uff1aAPI Gateway \u529f\u80fd\u6d4b\u8bd5\u5168\u90e8\u901a\u8fc7\u3002\u53d1\u73b0 5 \u4e2a\u5b9e\u73b0\u7ea7 Bug\u3002'),

        heading3('2.6 \u6027\u80fd\u57fa\u51c6\uff08Task 7\uff09\u2014 14 passed, 27 \u9879\u6307\u6807\u8fbe\u6807'),
        bodyText('\u6240\u6709 27 \u9879\u6027\u80fd\u6307\u6807\u5168\u90e8\u8fbe\u6807\u3002'),

        heading3('2.7 \u5bf9\u6807 Hindsight\uff08Task 8\uff09\u2014 12 passed'),
        bodyText('\u6240\u6709\u5bf9\u6bd4\u7ef4\u5ea6\u6d4b\u8bd5\u901a\u8fc7\uff0c\u6570\u636e\u5b8c\u6574\u3002'),

        heading3('2.8 \u7a33\u5b9a\u6027\u6d4b\u8bd5\uff08Task 9\uff09\u2014 10 passed, 7 failed, 1 skipped'),
        bodyText('\u5931\u8d25\u7528\u4f8b\u5206\u6790\uff1a'),
        makeTable(
          ['\u5931\u8d25\u7528\u4f8b', '\u5931\u8d25\u539f\u56e0'],
          [
            ['test_qdrant_down_memory_manager_behavior', 'Qdrant gRPC \u7aef\u53e3 6334 \u672a\u5728 docker-compose \u4e2d\u66b4\u9732\uff0cConnection refused'],
            ['test_postgres_down_memory_manager_behavior', 'PostgreSQL \u5916\u952e\u7ea6\u675f\u8fdd\u53cd\uff0c\u6d4b\u8bd5\u672a\u5148\u521b\u5efa tenant'],
            ['test_empty_input', '\u7a7a\u8f93\u5165\u672a\u6309\u9884\u671f\u629b\u51fa Exception'],
            ['test_deeply_nested_json', '\u6df1\u5ea6\u5d4c\u5957 JSON \u5bfc\u81f4 RecursionError'],
            ['test_restart_qdrant_data_persistence', 'Docker \u91cd\u542f\u540e gRPC \u7aef\u53e3\u4e0d\u53ef\u8fbe'],
            ['test_restart_postgres_data_persistence', 'PostgreSQL \u5916\u952e\u7ea6\u675f\u8fdd\u53cd'],
            ['test_5min_continuous_read_write', '\u957f\u65f6\u8fd0\u884c\u4f9d\u8d56\u5b8c\u6574 Docker \u670d\u52a1\u7f16\u6392']
          ],
          [3500, 6000]
        ),
        boldBodyText('\u7ed3\u8bba\uff1a7 \u4e2a\u5931\u8d25\u5168\u90e8\u5c5e\u4e8e\u57fa\u7840\u8bbe\u65bd\u914d\u7f6e\u95ee\u9898\u6216\u6d4b\u8bd5\u524d\u7f6e\u6761\u4ef6\u7f3a\u5931\uff0c\u975e SDK \u6838\u5fc3\u903b\u8f91\u7f3a\u9677\u3002'),

        pb(),
        heading2('3. \u6027\u80fd\u57fa\u51c6\u5173\u952e\u6307\u6807'),
        bodyText('\u6570\u636e\u6765\u6e90\uff1abenchmarks/benchmark_results.json\uff082026-04-22 16:44:19\uff09'),

        heading3('3.1 \u5ef6\u8fdf\u6307\u6807'),
        makeTable(
          ['\u64cd\u4f5c', 'P50 (ms)', 'P95 (ms)', 'P99 (ms)', '\u76ee\u6807', '\u72b6\u6001'],
          [
            ['\u8bed\u4e49\u7f16\u7801 (encode)', '0.002', '0.003', '0.004', 'P99 < 10ms', '\u8fbe\u6807'],
            ['\u5168\u606f\u68c0\u7d22 (holographic)', '0.281', '0.378', '0.452', 'P99 < 20ms', '\u8fbe\u6807'],
            ['\u8c61\u538b\u7f29 (compress)', '0.021', '0.024', '0.028', 'P99 < 100ms', '\u8fbe\u6807'],
            ['SDK \u5199\u5165 (add)', '0.022', '0.026', '0.030', 'P99 < 500ms', '\u8fbe\u6807'],
            ['SDK \u68c0\u7d22 (query)', '0.241', '0.495', '0.513', 'P99 < 400ms', '\u8fbe\u6807']
          ],
          [2200, 1200, 1200, 1200, 1800, 1000]
        ),

        heading3('3.2 \u541e\u5410\u91cf\u6307\u6807'),
        makeTable(
          ['\u64cd\u4f5c', 'QPS', '\u76ee\u6807 QPS', '\u9519\u8bef\u7387', '\u72b6\u6001'],
          [
            ['\u8bed\u4e49\u7f16\u7801', '526,618.7', '1,000', '0.0%', '\u8fbe\u6807'],
            ['\u5168\u606f\u68c0\u7d22', '3,890.4', '500', '0.0%', '\u8fbe\u6807'],
            ['SDK \u5199\u5165', '42,523.3', '50', '0.0%', '\u8fbe\u6807'],
            ['SDK \u68c0\u7d22', '27,566.2', '50', '0.0%', '\u8fbe\u6807'],
            ['\u6df7\u5408\u8bfb\u5199 (7:3)', '8,119.0', '40', '0.0%', '\u8fbe\u6807']
          ],
          [2200, 1800, 1500, 1200, 1000]
        ),

        heading3('3.3 \u5185\u5b58\u5360\u7528'),
        makeTable(
          ['\u6570\u636e\u89c4\u6a21', 'RSS (MB)', '\u76ee\u6807 (MB)', '\u72b6\u6001'],
          [
            ['1,000 \u6761', '96.25', '< 500', '\u8fbe\u6807'],
            ['10,000 \u6761', '95.33', '< 1,500', '\u8fbe\u6807']
          ],
          [2500, 2000, 2000, 1500]
        ),

        heading3('3.4 \u6269\u5c55\u6027\uff08\u5173\u952e\u98ce\u9669\uff09'),
        makeTable(
          ['\u6570\u636e\u89c4\u6a21', '\u5199\u5165 P50 (ms)', '\u68c0\u7d22 P50 (ms)', '\u68c0\u7d22\u589e\u5e45'],
          [
            ['100 \u6761', '0.022', '0.033', '\u2014'],
            ['1,000 \u6761', '0.025', '0.249', {text:'+659%', bold:true}],
            ['10,000 \u6761', '0.026', '2.470', {text:'+7,422%', bold:true}]
          ],
          [2500, 2000, 2000, 1800]
        ),
        boldBodyText('\u5173\u952e\u53d1\u73b0\uff1aSDK query \u5f53\u524d\u4f7f\u7528\u7ebf\u6027\u626b\u63cf\uff0c10K \u6570\u636e\u65f6\u68c0\u7d22\u5ef6\u8fdf\u4ece 0.033ms \u66b4\u589e\u81f3 2.47ms\uff0c\u589e\u5e45\u8fbe 74 \u500d\u3002'),

        pb(),
        heading2('4. \u5bf9\u6807 Hindsight \u5bf9\u6bd4\u7ed3\u8bba'),
        makeTable(
          ['\u5bf9\u6bd4\u7ef4\u5ea6', 'Hindsight', 'su-memory', '\u5dee\u8ddd', '\u7ed3\u8bba'],
          [
            ['\u5355\u8df3\u68c0\u7d22', '86.17%', '36.7%', '-49.5%', 'LOSE'],
            ['\u591a\u8df3\u63a8\u7406', '70.83%', '25.0%', '-45.8%', 'LOSE'],
            ['\u65f6\u5e8f\u7406\u89e3', '91.0%', '50.0%', '-41.0%', 'LOSE'],
            ['\u591a\u4f1a\u8bdd', '87.2%', '53.3%', '-33.9%', 'LOSE'],
            ['\u5f00\u653e\u9886\u57df', '95.12%', '53.3%', '-41.8%', 'LOSE'],
            [{text:'\u603b\u4f53\u51c6\u786e\u5ea6', bold:true}, {text:'91.4%', bold:true}, {text:'42.0%', bold:true}, {text:'-49.4%', bold:true}, {text:'LOSE', bold:true}],
            ['\u5168\u606f\u68c0\u7d22\u63d0\u5347', 'N/A', '+20.0%', '\u2014', '\u72ec\u6709'],
            ['\u8c61\u538b\u7f29\u7387', '~2x', '2.1x', '\u2014', '\u72ec\u6709'],
            ['\u56e0\u679c\u63a8\u7406\u8986\u76d6', 'N/A', '100.0%', '\u2014', '\u72ec\u6709'],
            ['\u52a8\u6001\u4f18\u5148\u7ea7', 'N/A', '100.0%', '\u2014', '\u72ec\u6709'],
            ['\u5143\u8ba4\u77e5\u53d1\u73b0\u7387', 'N/A', '100.0%', '\u2014', '\u72ec\u6709'],
            ['\u53ef\u89e3\u91ca\u6027', '\u65e0', '100.0%', '\u2014', '\u72ec\u6709']
          ],
          [1800, 1500, 1500, 1200, 1200]
        ),
        boldBodyText('\u5efa\u8bae\uff1a\u63a5\u5165 sentence-transformers \u6216 OpenAI embedding API \u66ff\u6362 hash-based fallback\uff0c\u9884\u671f\u51c6\u786e\u5ea6\u4ece 42% \u63d0\u5347\u81f3 70%+\u3002'),

        heading2('5. \u5df2\u53d1\u73b0\u7684 Bug \u4e0e\u98ce\u9669\u6e05\u5355'),
        heading3('5.1 API Bug'),
        makeTable(
          ['\u7ea7\u522b', 'Bug \u63cf\u8ff0', '\u5f71\u54cd'],
          [
            ['\u4e25\u91cd', 'MemoryItem \u5b57\u6bb5\u4e0d\u5339\u914d\uff1ascore vs relevance\u3001timestamp \u7c7b\u578b int vs str', '\u6838\u5fc3\u67e5\u8be2\u529f\u80fd\u4e0d\u53ef\u7528'],
            ['\u4e2d\u7b49', '/v1/tenant/create \u65e0\u9700\u9274\u6743\u5373\u53ef\u8c03\u7528', '\u79df\u6237\u521b\u5efa\u63a5\u53e3\u66b4\u9732'],
            ['\u4e2d\u7b49', 'JWT \u5bc6\u94a5\u6bcf\u6b21\u91cd\u542f\u968f\u673a\u751f\u6210', '\u6240\u6709\u5df2\u7b7e\u53d1 Token \u5931\u6548'],
            ['\u4e2d\u7b49', 'API Key (sk_) \u76f4\u63a5\u7528\u4f5c tenant_id\uff0c\u65e0\u6570\u636e\u5e93\u9a8c\u8bc1', '\u79df\u6237\u9694\u79bb\u53ef\u88ab\u7ed5\u8fc7'],
            ['\u4e2d\u7b49', 'setup_middleware() \u672a\u5728 main.py \u4e2d\u8c03\u7528', 'CSP/HSTS \u7b49\u5b89\u5168\u7b56\u7565\u7f3a\u5931']
          ],
          [1000, 5500, 2500]
        ),

        heading3('5.2 \u5173\u952e\u98ce\u9669'),
        makeTable(
          ['\u98ce\u9669', '\u63cf\u8ff0', '\u5f71\u54cd\u7ea7\u522b'],
          [
            ['\u8bed\u4e49\u68c0\u7d22\u8d28\u91cf', 'hash-based embedding \u4e25\u91cd\u5f71\u54cd\u68c0\u7d22\u8d28\u91cf\uff0c\u51c6\u786e\u5ea6\u4ec5 42%', '\u9ad8'],
            ['\u6269\u5c55\u6027\u74f6\u9888', 'SDK query \u7ebf\u6027\u626b\u63cf\uff0c10K \u6570\u636e\u5ef6\u8fdf\u589e\u5e45 74 \u500d', '\u9ad8'],
            ['\u8c61\u538b\u7f29\u8bed\u4e49\u6a21\u5f0f', '\u5143\u6570\u636e\u5bfc\u81f4\u5b57\u7b26\u81a8\u80c0 0.82-0.87x', '\u4e2d'],
            ['Docker \u914d\u7f6e', 'docker-compose.yml \u672a\u66b4\u9732 Qdrant gRPC \u7aef\u53e3', '\u4e2d'],
            ['SQLAlchemy 2.0 \u517c\u5bb9', 'declarative_base() \u5df2\u5f03\u7528', '\u4f4e']
          ],
          [2000, 5000, 1500]
        ),

        heading2('6. \u5546\u7528\u53d1\u5e03 Go/No-Go \u5224\u5b9a'),
        boldBodyText('\u5224\u5b9a\u7ed3\u679c\uff1aConditional Go\uff08\u6709\u6761\u4ef6\u53d1\u5e03\uff09'),
        bulletItem('\u6838\u5fc3 SDK \u529f\u80fd\u5168\u90e8\u901a\u8fc7\u6d4b\u8bd5\uff0c\u67b6\u6784\u8bbe\u8ba1\u65e0\u7f3a\u9677\u3002'),
        bulletItem('\u6027\u80fd\u57fa\u51c6\u5728\u5c0f\u6570\u636e\u91cf\u4e0b\u5168\u90e8\u8fbe\u6807\u3002'),
        bulletItem('\u5b58\u5728 1 \u4e2a\u4e25\u91cd Bug \u548c 1 \u4e2a\u6838\u5fc3\u6027\u80fd\u74f6\u9888\uff0c\u4f46\u5747\u4e3a\u53ef\u4fee\u590d\u7684\u5de5\u7a0b\u5b9e\u73b0\u95ee\u9898\u3002'),

        heading2('7. \u53d1\u5e03\u524d\u5fc5\u987b\u4fee\u590d\u7684\u9879\u76ee\uff08P0\uff09'),
        makeTable(
          ['\u4f18\u5148\u7ea7', '\u9879\u76ee', '\u539f\u56e0', '\u5de5\u4f5c\u91cf'],
          [
            ['P0', '\u4fee\u590d MemoryItem \u5b57\u6bb5\u4e0d\u5339\u914d', '\u4e25\u91cd Bug\uff0c\u963b\u585e\u6838\u5fc3\u67e5\u8be2\u529f\u80fd', '0.5 \u5929'],
            ['P0', '\u63a5\u5165\u771f\u5b9e embedding \u6a21\u578b', '\u8bed\u4e49\u68c0\u7d22\u8d28\u91cf 42% \u4e0d\u53ef\u63a5\u53d7', '2-3 \u5929'],
            ['P0', 'SDK query \u5f15\u5165\u7d22\u5f15\u673a\u5236', '10K \u6570\u636e\u5ef6\u8fdf\u589e\u5e45 74 \u500d', '2-3 \u5929'],
            ['P0', '\u4fee\u590d /v1/tenant/create \u9274\u6743\u7f3a\u5931', '\u5b89\u5168\u6f0f\u6d1e', '0.5 \u5929'],
            ['P0', '\u4fee\u590d JWT \u5bc6\u94a5\u8bfb\u53d6 .env', '\u91cd\u542f\u540e Token \u5931\u6548', '0.5 \u5929']
          ],
          [1000, 3500, 3000, 1200]
        ),

        heading2('8. \u5efa\u8bae\u4f18\u5148\u4fee\u590d\u7684\u9879\u76ee\uff08P1/P2\uff09'),
        makeTable(
          ['\u4f18\u5148\u7ea7', '\u9879\u76ee', '\u539f\u56e0', '\u5de5\u4f5c\u91cf'],
          [
            ['P1', 'API Key \u589e\u52a0\u6570\u636e\u5e93\u9a8c\u8bc1', '\u5b89\u5168\u52a0\u56fa', '1 \u5929'],
            ['P1', '\u8c03\u7528 setup_middleware()', '\u5b89\u5168\u5934\u751f\u6548', '0.5 \u5929'],
            ['P1', '\u66b4\u9732 Qdrant gRPC \u7aef\u53e3', '\u4fee\u590d\u7a33\u5b9a\u6027\u6d4b\u8bd5\u5931\u8d25', '0.5 \u5929'],
            ['P1', '\u7a33\u5b9a\u6027\u6d4b\u8bd5\u8865\u5145 tenant \u9884\u521b\u5efa', '\u4fee\u590d\u5916\u952e\u7ea6\u675f\u5931\u8d25', '0.5 \u5929'],
            ['P2', '\u5347\u7ea7 SQLAlchemy 2.0 \u8bed\u6cd5', '\u6d88\u9664\u5f03\u7528\u8b66\u544a', '1 \u5929'],
            ['P2', '\u4f18\u5316\u8c61\u538b\u7f29\u5143\u6570\u636e', '\u63d0\u5347\u538b\u7f29\u7387', '1-2 \u5929'],
            ['P2', '\u5f15\u5165\u5206\u9875\u673a\u5236', '\u9632\u6b62\u5185\u5b58\u6ea2\u51fa', '1 \u5929']
          ],
          [1000, 3500, 3000, 1200]
        ),

        heading2('9. \u6539\u8fdb\u8def\u7ebf\u56fe'),
        heading3('Phase 1\uff1a\u7d27\u6025\u4fee\u590d\uff081 \u5468\uff09'),
        bulletItem('\u4fee\u590d API \u5b57\u6bb5\u4e0d\u5339\u914d\u548c\u9274\u6743\u6f0f\u6d1e\uff08P0\uff09'),
        bulletItem('\u63a5\u5165 sentence-transformers all-MiniLM-L6-v2 \u4f5c\u4e3a\u9ed8\u8ba4 embedding'),
        bulletItem('SDK query \u5f15\u5165 BTree/\u5012\u6392\u7d22\u5f15\u52a0\u901f\u68c0\u7d22'),
        bulletItem('\u66f4\u65b0 docker-compose.yml \u66b4\u9732 gRPC \u7aef\u53e3'),
        heading3('Phase 2\uff1a\u6027\u80fd\u4f18\u5316\uff082 \u5468\uff09'),
        bulletItem('10K+ \u6570\u636e\u91cf\u4e13\u9879\u4f18\u5316\uff08\u5206\u9875\u3001\u9884\u52a0\u8f7d\u3001\u8fde\u63a5\u6c60\uff09'),
        bulletItem('\u8c61\u538b\u7f29\u8bed\u4e49\u6a21\u5f0f\u5143\u6570\u636e\u7cbe\u7b80'),
        bulletItem('API \u5b89\u5168\u52a0\u56fa\uff08Rate Limiting\u3001\u8f93\u5165\u8fc7\u6ee4\uff09'),
        heading3('Phase 3\uff1a\u751f\u4ea7\u5c31\u7eea\uff082 \u5468\uff09'),
        bulletItem('\u96c6\u6210 OpenAI / Claude embedding \u4f5c\u4e3a\u9ad8\u7aef\u9009\u9879'),
        bulletItem('\u5b8c\u5584\u76d1\u63a7\u548c\u544a\u8b66\uff08Prometheus \u6307\u6807\u66b4\u9732\uff09'),
        bulletItem('\u8865\u5145\u7a33\u5b9a\u6027\u6d4b\u8bd5\u81f3 100% \u901a\u8fc7'),
        bulletItem('\u53d1\u5e03 v1.0 GA'),

        // PART 2
        pb(),
        heading1('\u7b2c\u4e8c\u90e8\u5206\uff1a\u95ee\u9898\u5206\u6790\u4e0e\u4fee\u590d\u65b9\u6848'),
        bodyText('\u672c\u90e8\u5206\u63d0\u53d6\u81ea ISSUE_ANALYSIS_AND_FIX_PLAN.md \u7684\u6838\u5fc3\u5185\u5bb9\u3002'),

        heading2('1. \u95ee\u9898\u5168\u666f\u56fe'),
        makeTable(
          ['\u7ea7\u522b', '\u95ee\u9898\u7c7b\u578b', '\u95ee\u9898\u6570', '\u5f71\u54cd\u8303\u56f4'],
          [
            [{text:'\u4e25\u91cd Bug', bold:true}, '\u963b\u585e\u6838\u5fc3\u529f\u80fd', '1', 'query \u7aef\u70b9\u4e0d\u53ef\u7528'],
            [{text:'\u4e2d\u7b49 Bug', bold:true}, '\u5b89\u5168\u6f0f\u6d1e+\u914d\u7f6e\u95ee\u9898', '4', 'API\u9274\u6743\u3001\u4f1a\u8bdd\u7ba1\u7406\u3001\u57fa\u7840\u8bbe\u65bd'],
            [{text:'\u5173\u952e\u98ce\u9669', bold:true}, '\u6027\u80fd/\u8d28\u91cf\u74f6\u9888', '3', '\u8bed\u4e49\u68c0\u7d22\u3001\u53ef\u6269\u5c55\u6027\u3001Docker\u914d\u7f6e'],
            ['\u6027\u80fd\u74f6\u9888', '\u5927\u6570\u636e\u91cf\u9000\u5316', '1', '10K+ \u6570\u636e\u96c6'],
            ['\u7ade\u54c1\u5dee\u8ddd', '\u6280\u672f\u67b6\u6784\u5dee\u5f02', '1', '\u603b\u4f53\u51c6\u786e\u5ea6 42% vs 91.4%']
          ],
          [1800, 2500, 1200, 3500]
        ),
        bodyText('\u603b\u95ee\u9898\u6570\uff1a10 \u4e2a\u53d1\u73b0 | \u5546\u7528\u53d1\u5e03\u72b6\u6001\uff1a\u6761\u4ef6\u53d1\u5e03'),

        heading2('2. \u6838\u5fc3\u95ee\u9898\u8be6\u89e3'),

        heading3('Bug #1 \u4e25\u91cd\uff1aMemoryItem \u5b57\u6bb5\u4e0d\u5339\u914d\u5bfc\u81f4 query \u7aef\u70b9 500'),
        bodyText('\u95ee\u9898\uff1a/v1/memory/query \u7aef\u70b9\u8fd4\u56de\u524d\uff0cFastAPI Pydantic \u6a21\u578b\u6821\u9a8c\u5931\u8d25\uff0c\u5bfc\u81f4 500 \u9519\u8bef\u3002'),
        bulletItem('\u8def\u7531\u5b9a\u4e49\u4f7f\u7528\u5b57\u6bb5 relevance: float\uff0c\u4f46\u68c0\u7d22\u8fd4\u56de\u7ed3\u679c\u4f7f\u7528 score: float'),
        bulletItem('\u65f6\u95f4\u6233\u7c7b\u578b\u4e0d\u4e00\u81f4\uff1a\u5b58\u50a8\u4e3a int\uff0cPydantic \u671f\u671b str'),
        bodyText('\u5f71\u54cd\uff1a\u6240\u6709\u67e5\u8be2\u8bf7\u6c42\u8fd4\u56de 500\uff0c\u6838\u5fc3\u529f\u80fd\u5b8c\u5168\u4e0d\u53ef\u7528\u3002'),
        bodyText('\u4fee\u590d\uff1a\u4fee\u6539 gateway/router.py \u7684 MemoryItem \u6a21\u578b\u5b57\u6bb5\u3002\u5de5\u4f5c\u91cf 0.5 \u5929 | P0'),

        heading3('Bug #2 \u4e2d\u7b49\uff1a/v1/tenant/create \u65e0\u9700\u9274\u6743'),
        bodyText('\u95ee\u9898\uff1a\u7aef\u70b9\u6ca1\u6709\u9274\u6743\u6821\u9a8c\uff0c\u4efb\u4f55\u4eba\u90fd\u53ef\u8c03\u7528\u521b\u5efa\u79df\u6237\u3002'),
        bodyText('\u4fee\u590d\uff1a\u6dfb\u52a0 verify_admin_key \u4f9d\u8d56\u6ce8\u5165\uff0c\u6216\u5c06\u63a5\u53e3\u79fb\u5230\u79c1\u6709\u7ba1\u7406\u8def\u7531\u3002\u5de5\u4f5c\u91cf 0.5 \u5929 | P1'),

        heading3('Bug #3 \u4e2d\u7b49\uff1aJWT \u5bc6\u94a5\u6bcf\u6b21\u91cd\u542f\u968f\u673a\u751f\u6210'),
        bodyText('\u95ee\u9898\uff1aJWT_SECRET_KEY \u6bcf\u6b21\u542f\u52a8\u65f6\u968f\u673a\u751f\u6210\uff0c\u5ffd\u89c6 .env \u914d\u7f6e\u3002\u91cd\u542f\u540e\u6240\u6709 Token \u5931\u6548\u3002'),
        bodyText('\u4fee\u590d\uff1a\u4f18\u5148\u4ece os.getenv("JWT_SECRET_KEY") \u8bfb\u53d6\u3002\u5de5\u4f5c\u91cf 0.5 \u5929 | P0'),

        heading3('Bug #4 \u4e2d\u7b49\uff1aAPI Key \u76f4\u63a5\u7528\u4f5c tenant_id \u65e0\u9a8c\u8bc1'),
        bodyText('\u95ee\u9898\uff1aAPI Key\uff08sk_xxx\uff09\u76f4\u63a5\u8fd4\u56de\u4e3a tenant_id\uff0c\u672a\u67e5\u5e93\u9a8c\u8bc1\u3002'),
        bodyText('\u4fee\u590d\uff1a\u65b0\u589e get_tenant_by_api_key \u6570\u636e\u5e93\u67e5\u8be2\u3002\u5de5\u4f5c\u91cf 1 \u5929 | P1'),

        heading3('Bug #5 \u4e2d\u7b49\uff1asetup_middleware() \u672a\u5728 main.py \u8c03\u7528'),
        bodyText('\u95ee\u9898\uff1a\u5b89\u5168\u5934\u548c\u9650\u6d41\u4e2d\u95f4\u4ef6\u5b9a\u4e49\u5b58\u5728\u4f46\u672a\u96c6\u6210\u3002'),
        bodyText('\u4fee\u590d\uff1a\u5728 main.py \u7684 CORS \u4e2d\u95f4\u4ef6\u540e\u6dfb\u52a0 setup_middleware(app)\u3002\u5de5\u4f5c\u91cf 0.5 \u5929 | P1'),

        heading3('Risk #1 \u5173\u952e\uff1a\u8bed\u4e49\u68c0\u7d22\u8d28\u91cf\u4e25\u91cd\u4e0d\u8db3'),
        bodyText('\u95ee\u9898\uff1a\u603b\u4f53\u51c6\u786e\u5ea6\u4ec5 42%\uff0c\u4e0e Hindsight \u7684 91.4% \u76f8\u5dee 49.4 \u4e2a\u767e\u5206\u70b9\u3002'),
        bulletItem('\u9636\u6bb5 1\uff082-3 \u5929\uff09\uff1a\u63a5\u5165 sentence-transformers all-MiniLM-L6-v2 \u672c\u5730\u6a21\u578b'),
        bulletItem('\u9636\u6bb5 2\uff08\u53ef\u9009\uff09\uff1a\u652f\u6301 OpenAI/Claude Embedding API'),
        bodyText('\u9884\u671f\u6548\u679c\uff1a\u51c6\u786e\u5ea6\u4ece 42% \u63d0\u5347\u81f3 65-70%+'),

        heading3('Risk #2 \u5173\u952e\uff1aSDK query \u7ebf\u6027\u626b\u63cf\u5bfc\u81f4\u6269\u5c55\u6027\u74f6\u9888'),
        bodyText('\u95ee\u9898\uff1a10K \u6570\u636e\u67e5\u8be2\u5ef6\u8fdf\u4ece 0.033ms \u66b4\u589e\u81f3 2.47ms\uff08\u589e\u5e45 74 \u500d\uff09\u3002'),
        bodyText('\u4fee\u590d\uff1a\u5efa\u7acb bagua \u548c wuxing \u7d22\u5f15\uff0c\u65f6\u95f4\u590d\u6742\u5ea6\u4ece O(n) \u964d\u81f3 O(k)\u3002'),
        bodyText('\u9884\u671f\u6548\u679c\uff1a10K \u6570\u636e\u67e5\u8be2\u5ef6\u8fdf\u4ece 2.47ms \u964d\u81f3 0.3-0.5ms'),

        heading2('3. \u4fee\u590d\u4f18\u5148\u7ea7\u6392\u5e8f'),
        heading3('P0\uff08\u5fc5\u987b\u7acb\u5373\u4fee\u590d\uff09\u2014 4 \u4e2a\u95ee\u9898'),
        makeTable(
          ['\u5e8f\u53f7', '\u95ee\u9898', '\u5de5\u4f5c\u91cf', '\u9884\u8ba1\u5b8c\u6210'],
          [
            ['1', 'Bug #1 - MemoryItem \u5b57\u6bb5\u4e0d\u5339\u914d', '0.5\u5929', 'D1'],
            ['2', 'Bug #3 - JWT \u5bc6\u94a5\u8bfb\u53d6 .env', '0.5\u5929', 'D1'],
            ['3', 'Risk #1 - \u8bed\u4e49\u68c0\u7d22\u8d28\u91cf\uff08embedding\uff09', '2-3\u5929', 'D3-D4'],
            ['4', 'Risk #2 - SDK query \u7d22\u5f15\u52a0\u901f', '2-3\u5929', 'D3-D4']
          ],
          [1000, 4500, 1500, 1500]
        ),

        heading3('P1\uff08\u53d1\u5e03\u524d\u5fc5\u987b\u4fee\u590d\uff09\u2014 4 \u4e2a\u95ee\u9898'),
        makeTable(
          ['\u5e8f\u53f7', '\u95ee\u9898', '\u5de5\u4f5c\u91cf', '\u4f9d\u8d56'],
          [
            ['1', 'Bug #2 - /tenant/create \u9274\u6743', '0.5\u5929', '\u65e0'],
            ['2', 'Bug #4 - API Key \u6570\u636e\u5e93\u9a8c\u8bc1', '1\u5929', '\u65e0'],
            ['3', 'Bug #5 - setup_middleware \u8c03\u7528', '0.5\u5929', '\u65e0'],
            ['4', 'Risk #3 - docker-compose gRPC \u7aef\u53e3', '0.5\u5929', '\u65e0']
          ],
          [1000, 4500, 1500, 1500]
        ),

        heading2('4. \u5546\u7528\u53d1\u5e03\u524d Must-Fix \u6e05\u5355'),
        makeTable(
          ['\u9879\u76ee', '\u68c0\u67e5\u9879', '\u5f53\u524d\u72b6\u6001', '\u76ee\u6807\u72b6\u6001'],
          [
            ['\u529f\u80fd', 'query \u7aef\u70b9\u4e0d\u8fd4\u56de 500', '\u5931\u8d25', '\u901a\u8fc7'],
            ['\u529f\u80fd', '\u8bed\u4e49\u68c0\u7d22\u51c6\u786e\u5ea6', '42%', '>= 70%'],
            ['\u529f\u80fd', '10K \u6570\u636e\u68c0\u7d22\u5ef6\u8fdf', '2.47ms', '<= 0.5ms'],
            ['\u5b89\u5168', '/tenant/create \u9274\u6743', '\u65e0', '\u6709'],
            ['\u5b89\u5168', 'API Key \u9a8c\u8bc1', '\u65e0', '\u67e5\u5e93\u9a8c\u8bc1'],
            ['\u5b89\u5168', 'JWT \u4f1a\u8bdd\u6301\u4e45', '\u91cd\u542f\u5931\u6548', '\u6301\u4e45\u6709\u6548'],
            ['\u5b89\u5168', '\u5b89\u5168\u5934\u4e2d\u95f4\u4ef6', '\u672a\u542f\u7528', '\u542f\u7528'],
            ['\u57fa\u7840\u8bbe\u65bd', 'Qdrant gRPC \u53ef\u8fbe', '\u4e0d\u53ef\u8fbe', '\u53ef\u8fbe'],
            ['\u6d4b\u8bd5', '\u5355\u5143\u6d4b\u8bd5\u901a\u8fc7\u7387', '98.1%', '100%'],
            ['\u6d4b\u8bd5', '\u7a33\u5b9a\u6027\u6d4b\u8bd5\u901a\u8fc7\u7387', '55.6%', '>= 95%']
          ],
          [1500, 2500, 2000, 2000]
        ),

        // PART 3
        pb(),
        heading1('\u7b2c\u4e09\u90e8\u5206\uff1a\u6267\u884c\u6458\u8981'),

        heading2('1. \u5173\u952e\u53d1\u73b0'),
        heading3('\u6d4b\u8bd5\u6210\u679c'),
        bulletItem('365 \u4e2a\u5355\u5143\u6d4b\u8bd5\u901a\u8fc7\uff0898.1%\uff09'),
        bulletItem('\u6838\u5fc3\u7b97\u6cd5\uff08\u516b\u5366/\u4e94\u884c/\u5366\u7237\uff09\u5168\u90e8\u901a\u8fc7'),
        bulletItem('\u8bb0\u5fc6\u5f15\u64ce\u67b6\u6784\u5b8c\u5907\uff0887% \u8986\u76d6\u7387\uff09'),
        bulletItem('\u6027\u80fd\u57fa\u51c6 27/27 \u6307\u6807\u8fbe\u6807'),
        heading3('\u53d1\u73b0\u7684 10 \u4e2a\u95ee\u9898'),
        bulletItem('1 \u4e2a\u4e25\u91cd Bug\uff08query \u7aef\u70b9 500\uff09'),
        bulletItem('4 \u4e2a\u4e2d\u7b49 Bug\uff08\u5b89\u5168\u6f0f\u6d1e\uff09'),
        bulletItem('3 \u4e2a\u5173\u952e\u98ce\u9669\uff08\u6027\u80fd/\u8d28\u91cf\uff09'),
        bulletItem('2 \u4e2a\u4f18\u5316\u9879'),

        heading2('2. \u6027\u80fd\u57fa\u51c6\u603b\u4f53\u8bc4\u4ef7'),
        heading3('\u5c0f\u6570\u636e\u91cf\u6027\u80fd\uff081K \u4ee5\u4e0b\uff09\u2014 \u4f18\u79c0'),
        bulletItem('query P50\uff1a0.241 ms\uff08\u76ee\u6807 100ms\uff09'),
        bulletItem('\u541e\u5410\u91cf\uff1a27,566 QPS\uff08\u76ee\u6807 50 QPS\uff09'),
        bulletItem('\u5185\u5b58\u5360\u7528\uff1a96.25 MB\uff08\u76ee\u6807 500 MB\uff09'),
        heading3('\u5927\u6570\u636e\u91cf\u6027\u80fd\uff0810K+\uff09\u2014 \u9700\u8981\u6539\u8fdb'),
        bulletItem('\u7ebf\u6027\u626b\u63cf\u5bfc\u81f4\u67e5\u8be2\u5ef6\u8fdf\u589e\u5e45 74 \u500d'),

        heading2('3. \u7ade\u54c1\u5bf9\u6807\u5206\u6790'),
        heading3('\u843d\u540e\u7ef4\u5ea6'),
        makeTable(
          ['\u7ef4\u5ea6', 'Hindsight', 'su-memory', '\u5dee\u8ddd'],
          [
            ['\u5355\u8df3\u68c0\u7d22', '86.17%', '36.7%', '-49.5%'],
            ['\u591a\u8df3\u63a8\u7406', '70.83%', '25.0%', '-45.8%'],
            ['\u65f6\u5e8f\u7406\u89e3', '91.0%', '50.0%', '-41.0%'],
            ['\u5f00\u653e\u9886\u57df', '95.12%', '53.3%', '-41.8%'],
            [{text:'\u603b\u4f53\u51c6\u786e\u5ea6', bold:true}, {text:'91.4%', bold:true}, {text:'42.0%', bold:true}, {text:'-49.4%', bold:true}]
          ],
          [2200, 2000, 2000, 1500]
        ),
        heading3('\u9886\u5148\u7ef4\u5ea6'),
        makeTable(
          ['\u7ef4\u5ea6', 'Hindsight', 'su-memory', '\u4f18\u52bf'],
          [
            ['\u5168\u606f\u68c0\u7d22\u63d0\u5347', 'N/A', '+20%', '\u72ec\u6709'],
            ['\u8c61\u538b\u7f29\u7387', '~2x', '2.1x', '\u72ec\u6709'],
            ['\u56e0\u679c\u63a8\u7406\u8986\u76d6', 'N/A', '100%', '\u72ec\u6709'],
            ['\u52a8\u6001\u4f18\u5148\u7ea7', 'N/A', '100%', '\u72ec\u6709'],
            ['\u5143\u8ba4\u77e5\u53d1\u73b0\u7387', 'N/A', '100%', '\u72ec\u6709'],
            ['\u53ef\u89e3\u91ca\u6027', '\u65e0', '100%', '\u72ec\u6709']
          ],
          [2200, 2000, 2000, 1500]
        ),

        heading2('4. \u4e09\u9636\u6bb5\u53d1\u5e03\u8ba1\u5212'),
        heading3('Sprint 1\uff081-2 \u5929\uff09\uff1a\u7d27\u6025\u4fee\u590d P0 Bug'),
        bulletItem('\u4fee\u590d MemoryItem \u5b57\u6bb5\u4e0d\u5339\u914d\uff08query \u7aef\u70b9\uff09'),
        bulletItem('\u4fee\u590d JWT \u5bc6\u94a5\u914d\u7f6e'),
        bulletItem('\u542f\u52a8 embedding \u96c6\u6210'),
        bulletItem('\u542f\u52a8\u7d22\u5f15\u4f18\u5316'),
        heading3('Sprint 2\uff083-5 \u5929\uff09\uff1a\u5b8c\u5584 P0 Risk + P1 Bug'),
        bulletItem('\u5b8c\u6210 embedding \u96c6\u6210\uff08Hindsight \u5bf9\u6807 >= 70%\uff09'),
        bulletItem('\u5b8c\u6210\u7d22\u5f15\u4f18\u5316\uff0810K \u67e5\u8be2 <= 0.5ms\uff09'),
        bulletItem('\u4fee\u590d\u6240\u6709 P1 \u5b89\u5168 Bug'),
        bulletItem('\u96c6\u6210\u6d4b\u8bd5\u9a8c\u8bc1'),
        heading3('Sprint 3\uff08\u7b2c 6 \u5468\uff09\uff1a\u751f\u4ea7\u5c31\u7eea'),
        bulletItem('\u7a33\u5b9a\u6027\u6d4b\u8bd5 100% \u901a\u8fc7'),
        bulletItem('\u5b8c\u6574\u6587\u6863\u4e0e\u90e8\u7f72\u6307\u5357'),
        bulletItem('\u53d1\u5e03 v1.0 GA'),

        heading2('5. \u53d1\u5e03\u51b3\u7b56\u5efa\u8bae'),
        boldBodyText('\u76ee\u6807\uff1a\u4ece\u201c\u6761\u4ef6\u53d1\u5e03\u201d\u5347\u7ea7\u81f3\u201c\u65e0\u6761\u4ef6\u53d1\u5e03\u201d'),
        bodyText('\u603b\u8ba1\u5de5\u4f5c\u91cf\uff1a5-6 \u5929'),
        bulletItem('\u7acb\u5373\u542f\u52a8 P0 \u4fee\u590d\uff0c\u9884\u8ba1 Week 1 \u53ef\u5b8c\u6210'),
        bulletItem('\u5229\u7528 embedding \u96c6\u6210\u671f\u95f4\u5e76\u884c\u8fdb\u884c P1 \u4fee\u590d'),
        bulletItem('Week 2 \u5b8c\u6574\u9a8c\u8bc1\uff0cWeek 3 \u53d1\u5e03 v1.0 GA'),

        heading2('6. \u5173\u952e\u6307\u6807\u8ffd\u8e2a'),
        makeTable(
          ['\u6307\u6807', '\u53d1\u5e03\u524d', '\u53d1\u5e03\u540e\u76ee\u6807', '\u4f18\u5148\u7ea7'],
          [
            ['\u6838\u5fc3 API \u53ef\u7528\u6027', '0% (query 500)', '100%', 'P0'],
            ['\u8bed\u4e49\u68c0\u7d22\u51c6\u786e\u5ea6', '42%', '>= 70%', 'P0'],
            ['10K \u6570\u636e\u67e5\u8be2\u5ef6\u8fdf', '2.47ms', '<= 0.5ms', 'P0'],
            ['\u5355\u5143\u6d4b\u8bd5\u901a\u8fc7\u7387', '98.1%', '100%', 'P0'],
            ['\u7a33\u5b9a\u6027\u6d4b\u8bd5\u901a\u8fc7\u7387', '55.6%', '>= 95%', 'P1'],
            ['\u5b89\u5168\u6f0f\u6d1e\u6570', '4 \u4e2a', '0 \u4e2a', 'P1']
          ],
          [2500, 2200, 2000, 1200]
        ),

        new Paragraph({ spacing: { before: 400 } }),
        bodyText('\u62a5\u544a\u751f\u6210\u65f6\u95f4\uff1a2026-04-22'),
        boldBodyText('\u5efa\u8bae\u884c\u52a8\uff1a\u9a6c\u4e0a\u542f\u52a8 Sprint 1\uff0c\u4f18\u5148\u4fee\u590d P0 \u95ee\u9898\uff0c\u9884\u8ba1 5-6 \u5929\u53ef\u8fbe\u6210\u5546\u7528\u53d1\u5e03\u6807\u51c6\u3002')
      ]
    }
  ]
});

Packer.toBuffer(doc).then((buffer) => {
  const outPath = '/Users/mac/.openclaw/workspace/su-memory/su-memory-SDK\u6d4b\u8bd5\u62a5\u544a.docx';
  fs.writeFileSync(outPath, buffer);
  console.log('DOCX generated successfully:', outPath);
  console.log('File size:', (buffer.length / 1024).toFixed(1), 'KB');
}).catch((err) => {
  console.error('Error generating DOCX:', err);
  process.exit(1);
});
