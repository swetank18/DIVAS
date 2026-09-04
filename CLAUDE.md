# Instructions for Claude — DIVAS repo

## Read this first, every session

**At the start of every new chat in this repo, read `CONTEXT.md` before
doing anything else.** It has the condensed project state: what's real
vs. stub, current test/ablation numbers, environment setup (`.venv/`
usage), known gaps, and what claims are and aren't backed by data. Do not
re-derive this from scratch or re-read all of `STATUS.md` /
`EXECUTION_PLAN.md` / `PROJECT_OVERVIEW.md` unless `CONTEXT.md` sends you
there for detail.

## Code navigation — use the graph, not grep

This repo has a graphify knowledge graph at `graphify-out/graph.json`
(gitignored, local-only — not on a fresh clone), plus the
`code-review-graph` MCP server. **The graph is stale as of a large
teammate merge (18 commits, 67 files: real IDD segmentation model, BEV
projection, conformal margin, Bengaluru CARLA maps) — run
`/graphify --update` before trusting it for anything in those areas.**

**Before grepping or reading files to find a function, class, caller, or
relationship, use one of these first:**

- `graphify query "<question>"` — ask a natural-language question about
  the codebase (e.g. "what calls CarlaWorld.step", "where is d_safe
  computed"). Reads `graphify-out/graph.json` directly.
- `graphify explain "<NodeName>"` — plain-language explanation of a
  specific node.
- `graphify path "<A>" "<B>"` — shortest path between two concepts/files.
- `mcp__code-review-graph__*` tools — `semantic_search_nodes`,
  `query_graph` (callers_of/callees_of/imports_of/tests_for),
  `get_impact_radius`, `get_architecture_overview`, `list_communities`.

Fall back to Grep/Glob/Read only when the graph doesn't cover what's
needed, or for reading actual implementation detail once the graph has
pointed at the right file/line.

If the repo changes significantly (new files, big refactor), re-run
`/graphify --update` to keep the graph current rather than trusting a
stale one.
