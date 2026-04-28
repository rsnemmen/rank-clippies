# Website deployment guide

The interactive site lives in `docs/` and is served as a static GitHub Pages site. Python is the source of truth for all ranking math; the site only reads pre-computed JSON.

## Test locally

```bash
python -m http.server 8000 --directory docs
```

Open `http://localhost:8000` in your browser. The four category tabs, filter checkboxes, and both charts should all work without an internet connection (Plotly.js is loaded from CDN, so you need a network connection for the first load).

## Deploy to GitHub Pages

1. Push the `docs/` directory to `main` on GitHub (already done if you committed).
2. In the repo on GitHub, go to **Settings → Pages**.
3. Under **Source**, select `Deploy from a branch`, choose `main`, and set the folder to `/docs`.
4. Click **Save**. The site will be live at `https://rsnemmen.github.io/rank-clippies` within a minute or two.

You only need to do steps 2–4 once. Future pushes to `main` that include changes under `docs/` are deployed automatically.

## Refresh ranking data

Run the exporter whenever you add or update benchmarks or model scores in `data/`:

```bash
make refresh-ranking-data                    # writes to docs/data/
python rank_models.py --export-json          # equivalent direct command
python rank_models.py --export-json path/to/dir   # custom output directory
```

This regenerates all four category JSON files (`general`, `coding`, `agentic`, `stem`) in one shot.

## Update the website manually

If you are updating the site locally, commit and push the regenerated files:

```bash
git add docs/data/
git commit -m "Refresh rankings data"
git push
```

GitHub Pages picks up the new JSON automatically on the next page load.

> **Note:** `--export-json` requires `pandas` (same dependency as `--plot`).
> Install with `pip install pandas` if not already present.

## Update the website via GitHub Actions

This repository also includes a workflow that watches `data/**`, `rank_models.py`, and `Makefile` on `main`. When one of those inputs changes, GitHub Actions runs `make refresh-ranking-data` and commits updated `docs/data/` files back to `main` automatically.

If the workflow can read the repository but cannot push the regenerated JSON, enable **Settings → Actions → General → Workflow permissions → Read and write permissions** for the repository.
