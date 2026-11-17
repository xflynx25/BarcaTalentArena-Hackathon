"""Report pages for viewing results from a phone (backlog E1/E2, docs/viewing-plan.md).

`python -m src.report dashboard|timeline` writes self-contained HTML under reports/ (gitignored),
which agents publish as Claude Artifacts. External data only: BAR-ATH is refused by the loaders.
The exception is `local` (src/report/local.py, E3): BAR-ATH on the owner's laptop, written to
outputs/report/index.html only and never published.
"""
