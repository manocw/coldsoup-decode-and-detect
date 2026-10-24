# Marat's session notes (2026-10-24, chat with Claude) - handoff for Claude Code

## Leaderboard results from this session (test split, site score = 100 x AP)

| Run | File | Score | Verdict |
|---|---|---|---|
| 8 | v2b_8fam_stack re-run (same code, cloud box) | 72.1 | reproduces the 72.2 run (tiny library-version noise) |
| 9 | rank blend 0.85 v2b + 0.15 Marat flow-episode model (site 66.8) | 71.7 | worse |
| 10 | rank blend 0.70 v2b + 0.30 Marat | 71.2 | worse: more Marat weight = lower score |
| - | src/pseudo.py: self-training, single stage, top 300 test as pseudo-positive, bottom 60% pseudo-negative, 4 time blocks cross-fitted, weight 0.5 | 71.5 | ~ single-stage v2a (71.3), no real gain |
| - | rank blend 0.5 v2b + 0.5 self-trained | 72.1 | no gain |

Conclusion: v2b_8fam_stack (72.2) stays the best. Blending with Marat's model and self-training on
the test week both land within +-0.5 of it, consistent with TEAM_BRIEF section 7 (medium and hard
near their information limit).

Rankings: v2b vs Marat Spearman 0.59 overall but top-1% overlap 276/401 - the disagreement is in
the bulk, where Marat's model is simply weaker. v2b vs self-trained: Spearman 0.85, top-1% 320/401.

## Files added (uncommitted, in this clone)
- `src/blend.py` - rank-average submission CSVs (`--compare` for correlation / top-1% overlap).
- `src/pseudo.py` - self-training on the test week (negative result above).
- `notes/marat_session.md` - this file.
Test CSVs are in `Downloads/lb_tests/` (not in the repo).

## Repo / git state
- Local setting `core.autocrlf=true` was set in this clone (Windows line endings showed every file
  as modified from the Linux side). Nothing was committed or pushed.
- Marat's instruction: commits must NOT add Claude as co-author / contributor.

## Still to do before 16:30
1. Final upload: v2b predictions + code ZIP (`python -m src.package`).
2. Slides (< 10) or report (<= 4 pages) from docs/TEAM_BRIEF.md; add the results above as tested
   and rejected ideas.
3. Team details: name coldsoup, captain, members, final commit hash.
