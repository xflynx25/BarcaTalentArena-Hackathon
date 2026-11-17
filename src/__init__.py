"""Ball-possession pipeline for More than a Hack 2027, Phase 1.

    src/official.py      bridge to the read-only official repo (games.py, scorer)
    src/schema.py        the internal Match format every loader produces
    src/ingest/          one loader per data source -> Match
    src/models/          Match -> per-frame label codes
    src/submission.py    write, validate and score submission files
    src/run.py           the one command the owner runs on the real match data
    src/check_data.py    format checks on the real bundle, pass/fail only
    src/synthetic.py     fake BAR-ATH bundle in the official formats, for cloud agents and tests

Nothing in this package prints data rows. Anything run on the match data reports aggregates only.
"""
