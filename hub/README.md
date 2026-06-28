# Discovery Hub

Research funnel that scans a US equity universe, scores each name with seven
early-mover signals, ranks the top candidates, and explains the likely catalyst.

> ⚠️ Research tool, not buy signals or financial advice. It surfaces candidates
> with many false positives; it does not predict price moves.

## Run

    .venv/bin/pip install -r hub/requirements.txt
    .venv/bin/python -m hub scan                 # full scan (LLM explanations)
    .venv/bin/python -m hub scan --no-explain    # signals only, no LLM/API key
    .venv/bin/python -m hub backtest             # honest walk-forward validation

Output lands in `out/` (CSV + JSON + HTML). Needs `ANTHROPIC_API_KEY` for explanations.
