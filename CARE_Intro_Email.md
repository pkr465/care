**Subject:** Introducing CARE — Automated Design Health Scoring for Your RTL

---

Hi team,

I wanted to introduce CARE (Codebase Analysis & Repair Engine), a framework we've built to bring automated, reproducible quality analysis to our Verilog and SystemVerilog codebases.

**What it does:** CARE runs 9 purpose-built HDL analyzers across your RTL — covering synthesis safety, CDC violations, signal integrity, complexity, documentation gaps, and more — then produces a single weighted Design Health Score (A through F) so you can see exactly where a module stands before it hits synthesis.

**Why it matters:** Today, code review quality depends on who's reviewing and how much time they have. CDC issues and blocking-assign mistakes slip through to synthesis or, worse, silicon. CARE catches these patterns automatically and consistently, every run.

A few highlights worth noting:

- **EDA tool integration** — hooks into Verilator and Verible for deep lint and parsing, with regex fallback so it works without any tool setup
- **LLM-powered design review** — module-boundary-aware chunking means the AI never splits an always block mid-analysis; constraint rules let you permanently override false positives
- **Human-in-the-Loop feedback** — engineer decisions persist in PostgreSQL and feed back into future runs via RAG retrieval, so the system gets smarter over time
- **Auto-repair with auditable diffs** — every suggested fix comes as a reviewable patch, nothing changes silently

The output is an Excel report with per-file scores, a JSON health summary, and an optional Streamlit dashboard for interactive review and fix workflows.

I've attached a short slide deck that walks through the architecture and scoring model. Happy to set up a 30-minute walkthrough for anyone interested in piloting this on their next tapeout block.

Best,
Pavan
