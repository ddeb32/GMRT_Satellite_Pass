# GMRT Satellite Pass / Source Overlap Tool

A web application that replicates the GMRT internal satellite-pass prediction
tool, built with Python + Streamlit.  The Band 3 (250–500 MHz) satellite list
is verified against the GMRT web tool.

**Live demo:** *(your Streamlit Cloud URL goes here after deployment)*

---

## What it does

For a given observation window it predicts which pre-selected satellites will
pass within a specified angular separation of each target source, and for how
long — matching the output format of the internal GMRT tool at
`ww1.gmrt.ncra.tifr.res.in/~astrosupp/sat/pre/sat.html`.

## Features

- All 6 GMRT frequency bands (dropdown selector)
- Automatic TLE download from CelesTrak with catalog-file fallback for
  satellites that fail the individual query
- Polar sky-path plot (downloadable PNG)
- Report download as plain text
- Built-in example source file

---

## Deploy to Streamlit Cloud (free, < 5 minutes)

### 1 — Fork / push to GitHub

```bash
# Clone or fork this repository, then:
git add .
git commit -m "initial commit"
git push origin main
```

### 2 — Connect to Streamlit Cloud

1. Go to **[share.streamlit.io](https://share.streamlit.io)** and sign in with GitHub.
2. Click **New app**.
3. Select your repository and branch (`main`).
4. Set **Main file path** → `app.py`.
5. Click **Deploy**.

Streamlit Cloud installs `requirements.txt` automatically and gives you a
public URL like `https://yourname-gmrt-sat-overlap-app-xxxx.streamlit.app`.

No server, no Docker, no cost for public repositories.

---

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

---

## Source file format

```
# name        RA                Dec               Epoch
3C48          01h37m41.30s      +33d09'35.13"     2000.0
0521-207      05h21m38.90s      -20d47'39.40"     2000.0
J0437-4715    04h37m15.80s      -47d15'09.10"     2000.0   12
```

Lines beginning with `#` or `{` are skipped.
Extra columns after the epoch are accepted and ignored.
An example file is bundled: `example_sources.txt`.

---

## Repository structure

```
gmrt-sat-overlap/
├── app.py                  ← Streamlit application (single file)
├── requirements.txt        ← Python dependencies
├── example_sources.txt     ← Example pulsar timing source list
├── .streamlit/
│   └── config.toml         ← Theme and upload-size settings
└── README.md
```

## Notes on satellite lists

| Band | Status |
|------|--------|
| Band 3 (250–500 MHz) | ✅ Verified against GMRT web tool |
| Band 2 (125–250 MHz) | ⚠ Best-effort (NOAA weather satellites) |
| Band 4 (550–900 MHz) | ⚠ Best-effort (UFO UHF series) |
| Band 5 (1000–1450 MHz) | ⚠ Best-effort (Inmarsat L-band) |
| Band 1 (50–80 MHz) | ❌ Empty — add NORAD IDs after verifying with GMRT |
| Band 6 (All) | Union of bands 1–5 |

Cross-check non-Band-3 lists with the GMRT web tool and update
`BAND_SATELLITES` in `app.py` as needed.
