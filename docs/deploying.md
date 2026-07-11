# Deploying the dashboard as a website

The interactive dashboard has two deployment stories:

## 1. Static site — Cloudflare Pages (or any static host)

Cloudflare Pages serves static files; it cannot run the Python (Dash/Flask)
server. So sigmaflow exports a **static version** of the dashboard: the four
live open-data feeds (NOAA solar wind, NDBC buoy, GOES magnetometer, Elexon
GB grid frequency) are fetched at build time, detection scores for every
signal x detector pair are precomputed into JSON, and the threshold control,
metrics, anomaly table, and plain-language interpretation all run client-side
in the browser. Each signal shows its source and fetch timestamp. The one
thing a static site can't do is *re-fit* a detector with new parameters —
that needs Python.

### Build the site

```bash
sigmaflow dashboard --export site
# add your own signal to the browser alongside the live feeds:
sigmaflow dashboard my_data.h5 --export site
```

Re-running the export refreshes the data snapshot; redeploying publishes it.
(To automate daily refreshes later: any scheduler that runs these two
commands works — GitHub Actions cron is the usual choice once the repo is on
GitHub.)

This writes a fully self-contained `site/` directory (~1 MB): `index.html`
plus one JSON per signal under `data/`.

### Deploy to Cloudflare Pages

One-time setup (needs a free Cloudflare account and Node.js):

```bash
npx wrangler login        # opens the browser to authorize
```

Deploy:

```bash
npx wrangler pages deploy site --project-name sigmaflow
```

Wrangler creates the project on first deploy and prints the public URL
(`https://sigmaflow.pages.dev` or similar). Re-run the same two commands
(`--export`, `deploy`) whenever you want to publish an update.

**No terminal? Use the dashboard instead:** Cloudflare dashboard →
*Workers & Pages* → *Create* → *Pages* → *Upload assets* → drag the
`site/` folder in.

Notes:
- The page loads Plotly.js from `cdn.plot.ly`; to be fully self-hosted,
  download `plotly-2.35.2.min.js` into `site/` and change the `<script src>`
  in `index.html` to the local file.
- The same `site/` directory works unchanged on GitHub Pages, Netlify,
  S3 + CloudFront, or any web server.

## 2. Live server — full interactivity

For live detector re-fitting (parameter tuning against arbitrary uploaded
data), the Python app itself must run somewhere. Cloudflare Pages can't host
it, but any Python-friendly platform can:

```python
# app.py — what a hosting platform runs
from sigmaflow.dashboard import create_app

app = create_app()          # Dash app
server = app.server         # the underlying Flask app for gunicorn
```

```
# Procfile / start command
gunicorn app:server
```

Works on Render, Fly.io, Railway, or a VPS. You can still put Cloudflare in
front of it as DNS/proxy — just not Pages.
