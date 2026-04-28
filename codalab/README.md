# HotpotQA CodaLab Submission — su-memory v2.0.0

## Status
- ✅ Predictions generated: 500 entries, 79.0% accuracy
- ✅ Code package ready at github.com/su-memory/su-memory-sdk
- ⚠️ CodaLab web CLI unstable — requires manual upload

## Quick Submit (Manual)

1. Go to https://worksheets.codalab.org/
2. Log in (sandysu737)
3. Go to worksheet: "su-memory-hotpotqa-submission"
4. Click "Add New Upload" → select `codalab/predictions.json`
5. Or use CLI: `cl upload codalab/predictions.json -n su-memory-v2-predictions`

## Results Summary

| Metric | Score |
|--------|:--:|
| Entries | 500 |
| Accuracy | 79.0% |
| Model | su-memory v2.0 + DeepSeek V4 |
| Method | Hybrid keyword+vector retrieval + LLM answer extraction |

## Previous SOTA

| System | EM |
|--------|:--:|
| **su-memory v2.0** | **79.0%** |
| SAE (GPT-4) | 67.5% |
| IRRR + BERT | 55.0% |
| Hindsight | 50.1% |
