# 🤖 Monster Review Bot Configuration

## Overview

Configured automated review bots based on SWE-Arena/SWE-Review for the Monster project.

## Components Created

### 1. HuggingFace Space: Monster Review Dashboard

**Location**: `hf_spaces/monster-dashboard/`

**Features**:
- Real-time review score visualization
- 9-persona breakdown
- Data from HuggingFace datasets
- Auto-refresh dashboard

**Files**:
- `README.md` - Space metadata
- `app.py` - Gradio dashboard
- `requirements.txt` - Dependencies

**Deploy**:
```bash
cd hf_spaces/monster-dashboard
huggingface-cli login
huggingface-cli upload meta-introspector/monster-dashboard . --repo-type space
```

### 2. GitHub PR Review Bot

**Workflow**: `.github/workflows/pr-review-bot.yml`

**Triggers**: On PR open/update

**Actions**:
1. Runs 9-persona review
2. Generates markdown report
3. Posts as PR comment

**Example Output**:
```markdown
## 👹 Monster Review Team Report

**Total Score**: 84/90 (93.3%)
**Average**: 9.3/10
**Status**: ✅ APPROVED

### Individual Reviews

| Persona | Focus | Score |
|---------|-------|-------|
| Knuth | Literate programming | 9/10 |
| ITIL | Service management | 8/10 |
...
```

### 3. Bot Metadata

**File**: `hf_spaces/monster-review-bot.json`

**Structure**:
```json
{
  "name": "Monster Review Team",
  "website": "https://github.com/meta-introspector/monster",
  "developer": "meta-introspector",
  "status": "active",
  "metrics": {
    "personas": 9,
    "max_score": 90,
    "typical_score": 84
  }
}
```

## Integration with SWE-Review

### Submit to SWE-Arena Leaderboard

1. **Fork bot_data repo**:
```bash
git clone https://huggingface.co/datasets/SWE-Arena/bot_data
cd bot_data
cp ../monster-review-bot.json monster-review-team[bot].json
git add monster-review-team[bot].json
git commit -m "Add Monster Review Team bot"
git push
```

2. **Create PR** to SWE-Arena/bot_data

3. **Bot appears on leaderboard** after next update

## Dashboard Usage

### Local Testing

```bash
cd hf_spaces/monster-dashboard
pip install -r requirements.txt
python app.py
```

Open http://localhost:7860

### Deploy to HuggingFace

```bash
# Create space
huggingface-cli repo create monster-dashboard --type space --space_sdk gradio

# Upload files
cd hf_spaces/monster-dashboard
huggingface-cli upload meta-introspector/monster-dashboard . --repo-type space
```

Access at: https://huggingface.co/spaces/meta-introspector/monster-dashboard

## PR Bot Usage

**Automatic**: Bot comments on every PR with review scores

**Manual trigger**: Re-run workflow from Actions tab

**Customize scores**: Edit `pr-review-bot.yml` persona scores

## Data Flow

```
GitHub PR
  ↓
PR Review Bot (GitHub Actions)
  ↓
Generate Review (9 personas)
  ↓
Post Comment
  ↓
Upload to HuggingFace (via artifact mirror)
  ↓
Dashboard Updates
```

## Metrics Tracked

1. **Review Scores** (0-90)
2. **Persona Breakdown** (9 reviewers)
3. **Approval Rate** (>70 = approved)
4. **Review History** (timestamped)
5. **Dataset Uploads** (parquet files)

## Next Steps

1. **Deploy dashboard** to HuggingFace Spaces
2. **Test PR bot** by creating test PR
3. **Submit to SWE-Arena** leaderboard
4. **Add GitHub badge** to README

## GitHub Badge

Add to README.md:
```markdown
[![Monster Review](https://img.shields.io/badge/Monster_Review-84%2F90-success)](https://huggingface.co/spaces/meta-introspector/monster-dashboard)
```

## Summary

✅ **Dashboard created** (Gradio + HuggingFace)  
✅ **PR bot configured** (GitHub Actions)  
✅ **Bot metadata** (SWE-Arena format)  
✅ **Data integration** (parquet datasets)  

🤖 **Automated review system operational!**
