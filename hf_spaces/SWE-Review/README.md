---
title: SWE-Review
emoji: 👁️
colorFrom: red
colorTo: pink
sdk: gradio
sdk_version: 5.50.0
app_file: app.py
hf_oauth: true
pinned: false
short_description: Track GitHub review statistics for SWE assistants
---

# SWE Assistant Review Leaderboard

SWE-Review ranks software engineering assistants by their real-world GitHub review performance.

No benchmarks. No sandboxes. Just real PR reviews from actual repositories.

## Why This Exists

Most AI assistant benchmarks use synthetic tasks and simulated environments. This leaderboard measures real-world performance: how many PRs did the assistant review? What percentage were merged? Were the reviews valuable?

If an assistant can consistently provide valuable reviews across different projects, that tells you something no benchmark can.

## What We Track

Key metrics from the last 180 days:

**Leaderboard Table**
- **Assistant**: Display name of the assistant
- **Website**: Link to the assistant's homepage or documentation
- **Total Reviews**: PR reviews the assistant has made
- **Merged PRs**: PRs reviewed by the assistant that were merged
- **Acceptance Rate**: Percentage of reviewed PRs that were merged

**Monthly Trends**
- Acceptance rate trends (line plots)
- Review volume over time (bar charts)

We focus on 180 days to highlight current capabilities and active assistants.

## How It Works

**Data Collection**
We mine GitHub activity from [GHArchive](https://www.gharchive.org/), tracking:
- PR reviews by the assistant (`PullRequestReviewEvent`)
- PR review comments by the assistant (`PullRequestReviewCommentEvent`)

For each reviewed PR, we determine status: Merged, Rejected (closed without merge), or Pending (still open).

**Regular Updates**
Leaderboard refreshes weekly (Wednesday at 00:00 UTC).

**Community Submissions**
Anyone can submit an assistant. We store metadata in `SWE-Arena/bot_data` and results in `SWE-Arena/leaderboard_data`. All submissions are validated via GitHub API.

## Understanding the Metrics

**Acceptance Rate**
Percentage of reviewed PRs ultimately merged:

```
Acceptance Rate = Merged PRs ÷ (Merged PRs + Rejected PRs) × 100
```

Pending PRs (still open) are excluded to measure only completed reviews.

What this tells us:
- High rates = valuable reviews identifying quality PRs
- Balanced rates = thorough, critical review practices
- Very low rates = potentially harsh or inaccurate reviews

Context matters: 100 reviews at 70% acceptance differs from 10 reviews at 100%. Consider both rate and volume.

## What's Next

Planned improvements:
- Repository-based analysis
- Extended metrics (response time, depth, message quality)
- Review sentiment analysis
- Review patterns (security, code quality, architecture)
- PR characteristics (size, complexity, type)

## Questions or Issues?

[Open an issue](https://github.com/SE-Arena/SWE-Review/issues) for bugs, feature requests, or data concerns.