"""
GMRT Satellite Pass / Source Overlap Tool — Streamlit web application
Replicates the GMRT internal satellite-pass prediction tool.

Deploy to Streamlit Cloud:
  1. Push this repo to GitHub (see README.md)
  2. Go to share.streamlit.io → New app → select repo → Main file: app.py
"""

import re
import io
import requests
import urllib3
import streamlit as st
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from skyfield.api import wgs84, EarthSatellite, load
from skyfield.framelib import true_equator_and_equinox_of_date

# Suppress SSL warnings when verify=False fallback is used
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# ─── Observatory & time constants ─────────────────────────────────────────────
GMRT   = wgs84.latlon(19.0964, 74.0497, elevation_m=650)
IST    = timedelta(hours=5, minutes=30)
MIN_EL = 17.0

# ─── Satellite lists per GMRT frequency band ──────────────────────────────────
# Band 3 verified against ww1.gmrt.ncra.tifr.res.in/~astrosupp/sat/pre/sat.html
BAND_SATELLITES = {
    '1': {},   # 50–80 MHz — populate after verifying with GMRT web tool
    '2': {     # 125–250 MHz
        25338: 'NOAA 15',     28654: 'NOAA 18',     33591: 'NOAA 19',
        43013: 'NOAA 20',     47438: 'NOAA 21',
        39444: 'METEOR-M 2',  44387: 'METEOR-M2-2',
    },
    '3': {     # 250–500 MHz ← verified
        37212: 'MERIDIAN 3',  38995: 'MERIDIAN 6',  44453: 'MERIDIAN 8',
        22563: 'UFO 1 (USA 98)', 22787: 'UFO 2 (USA 95)', 22921: 'NATO 4B',
        32294: 'SKYNET 5B',   35943: 'COMSATBW-1',  40887: 'MUOS 4',
    },
    '4': {     # 550–900 MHz
        22076: 'UFO 3 (USA 108)', 23170: 'UFO 4 (USA 113)',
        23467: 'UFO 5 (USA 118)', 24315: 'UFO 6 (USA 130)',
        25272: 'UFO 7 (USA 143)', 26063: 'UFO 8 (USA 164)',
        27597: 'UFO 10 (USA 183)', 28897: 'UFO 11 (USA 201)',
    },
    '5': {     # 1000–1450 MHz
        23839: 'INMARSAT 3-F4', 25153: 'INMARSAT 3-F2',
        24958: 'INMARSAT 3-F3', 24307: 'INMARSAT 3-F5',
        28628: 'INMARSAT 4-F2', 26624: 'INMARSAT 4-F1',
        32791: 'INMARSAT 4-F3', 39476: 'INMARSAT-5 F1',
        40732: 'INMARSAT-5 F2', 41838: 'INMARSAT-5 F3',
        43712: 'INMARSAT-5 F4',
    },
}

BAND_LABELS = {
    '1': 'Band 1  (50 – 80 MHz)',
    '2': 'Band 2  (125 – 250 MHz)',
    '3': 'Band 3  (250 – 500 MHz)',
    '4': 'Band 4  (550 – 900 MHz)',
    '5': 'Band 5  (1000 – 1450 MHz)',
    '6': 'Band 6  All  (1 – 1500 MHz)',
}

_ALL_SAT_NAMES: dict = {}
for _b in BAND_SATELLITES.values():
    _ALL_SAT_NAMES.update(_b)

# CelesTrak URLs tried in order for each satellite (individual query first,
# then bulk catalog fallbacks for those that fail)
_CATALOG_FALLBACKS = [
    ('geo',      'https://celestrak.org/NORAD/elements/geo.txt'),
    ('military', 'https://celestrak.org/NORAD/elements/military.txt'),
    ('active',   'https://celestrak.org/NORAD/elements/active.txt'),
]

# Headers that CelesTrak and most TLE servers accept from automated tools
_HEADERS = {
    'User-Agent': (
        'Mozilla/5.0 (compatible; GMRT-SatPassTool/2.0; '
        'astronomy research; +https://github.com)'
    ),
    'Accept': 'text/plain, */*',
}

RA_RE  = re.compile(r'(\d+)h(\d+)m([\d.]+)s')
DEC_RE = re.compile(r'([+-])(\d+)d(\d+)\'([\d.]+)"')


# ─── Timescale (cached once per process) ──────────────────────────────────────

@st.cache_resource
def get_ts():
    return load.timescale(builtin=True)


# ─── Network helpers ──────────────────────────────────────────────────────────

def _http_get(url: str, timeout: int = 30) -> str | None:
    """
    GET a URL with proper headers.
    Returns the response text on success, None on any failure.
    Falls back to verify=False if SSL handshake fails (some cloud egress
    proxies present different certificates).
    Does NOT cache the result — callers handle caching via session_state
    so failures are never persisted between button clicks.
    """
    for verify in (True, False):
        try:
            r = requests.get(url, headers=_HEADERS, timeout=timeout, verify=verify)
            if r.ok and len(r.text.strip()) > 20:
                return r.text
        except requests.exceptions.SSLError:
            continue          # try without SSL verification
        except Exception:
            return None
    return None


# ─── TLE parsing ──────────────────────────────────────────────────────────────

def parse_tle_block(text: str) -> dict:
    """Parse a TLE text block → {norad_int: EarthSatellite}."""
    ts    = get_ts()
    sats  = {}
    lines = [l for l in text.strip().splitlines() if l.strip()]
    i = 0
    while i < len(lines):
        if (i + 2 < len(lines)
                and not lines[i].startswith('1 ')
                and not lines[i].startswith('2 ')
                and lines[i+1].startswith('1 ')
                and lines[i+2].startswith('2 ')):
            try:
                s = EarthSatellite(lines[i+1], lines[i+2], lines[i], ts)
                sats[s.model.satnum] = s
            except Exception:
                pass
            i += 3
        elif (i + 1 < len(lines)
              and lines[i].startswith('1 ')
              and lines[i+1].startswith('2 ')):
            try:
                s = EarthSatellite(lines[i], lines[i+1], '', ts)
                sats[s.model.satnum] = s
            except Exception:
                pass
            i += 2
        else:
            i += 1
    return sats


# ─── Satellite loading ────────────────────────────────────────────────────────

def load_satellites(norads: list, tle_upload_text: str | None,
                    log: callable) -> dict:
    """
    Load EarthSatellite objects for the given NORAD IDs.

    Caching: successful results are stored in st.session_state keyed by
    the sorted NORAD list so they are reused within a session without
    re-downloading.  Failures are NEVER cached — the user can click
    'Generate passes' again after a transient network hiccup.

    Strategy:
      1. Session-state hit  → return immediately (no network)
      2. Uploaded TLE file  → parse and return
      3. Per-satellite CelesTrak query
      4. Catalog fallback (geo → military → active) for any failures
    """
    sats: dict = {}

    # ── uploaded TLE file ────────────────────────────────────────────────
    if tle_upload_text:
        parsed = parse_tle_block(tle_upload_text)
        for n in norads:
            if n in parsed:
                sats[n] = parsed[n]
                log(f"  ✓ NORAD {n:6d}  {_ALL_SAT_NAMES.get(n,'')} [uploaded TLE]")
            else:
                log(f"  ✗ NORAD {n:6d}  {_ALL_SAT_NAMES.get(n,'')} — not in uploaded TLE")
        return sats

    # ── session-state cache ───────────────────────────────────────────────
    cache_key = 'tles_' + '_'.join(str(n) for n in sorted(norads))
    if cache_key in st.session_state:
        cached = st.session_state[cache_key]
        log(f"  ✓ Using {len(cached)} TLE(s) cached in this session "
            f"(click 'Clear TLE cache' to force a fresh download)")
        return cached

    # ── Step 1: per-satellite CelesTrak queries ───────────────────────────
    failed = []
    for n in norads:
        url  = f"https://celestrak.org/NORAD/elements/gp.php?CATNR={n}&FORMAT=tle"
        text = _http_get(url)
        if text:
            parsed = parse_tle_block(text)
            if n in parsed:
                sats[n] = parsed[n]
                log(f"  ✓ NORAD {n:6d}  {_ALL_SAT_NAMES.get(n,'')}")
                continue
        failed.append(n)
        log(f"  ✗ NORAD {n:6d}  {_ALL_SAT_NAMES.get(n,'')} "
            f"— direct query failed, trying catalog fallback")

    # ── Step 2: catalog fallback for any that failed ──────────────────────
    if failed:
        log(f"\n  Catalog fallback for NORAD {failed} ...")
        for cat_name, cat_url in _CATALOG_FALLBACKS:
            if not failed:
                break
            log(f"  ↻ Downloading {cat_name} catalog ... "
                f"(this may take 10–30 s)")
            cat_text = _http_get(cat_url, timeout=60)
            if cat_text:
                by_norad  = parse_tle_block(cat_text)
                still_bad = []
                for n in failed:
                    if n in by_norad:
                        sats[n] = by_norad[n]
                        log(f"  ✓ NORAD {n:6d}  {_ALL_SAT_NAMES.get(n,'')} "
                            f"[found in {cat_name}]")
                    else:
                        still_bad.append(n)
                failed = still_bad
            else:
                log(f"  ✗ {cat_name} catalog download failed")

        if failed:
            log(f"\n  ✗ WARNING: NORAD {failed} not found anywhere — skipped.")

    # Cache only if at least one satellite was loaded successfully
    if sats:
        st.session_state[cache_key] = sats
    else:
        log("\n  ✗ All TLE fetches failed.  Possible causes:\n"
            "     • CelesTrak is temporarily down\n"
            "     • Streamlit Cloud's outbound IP is blocked by CelesTrak\n"
            "  → Upload a local TLE file via the 'Advanced' expander and try again.\n"
            "  → Or click 'Clear TLE cache' and retry in a few minutes.")

    return sats


# ─── Source parsing ───────────────────────────────────────────────────────────

def parse_sources_text(text: str) -> list:
    out = []
    for line in text.splitlines():
        line = line.strip()
        if not line or line[0] in '#{}':
            continue
        tok = line.split()
        if len(tok) < 4:
            continue
        m1, m2 = RA_RE.match(tok[1]), DEC_RE.match(tok[2])
        if not m1 or not m2:
            continue
        h, mn, s = map(float, m1.groups())
        sg, d, am, asec = m2.groups()
        dec = int(d) + int(am)/60 + float(asec)/3600
        out.append(dict(name=tok[0], ra_txt=tok[1], dec_txt=tok[2], epoch=tok[3],
                        ra_h=h + mn/60 + s/3600,
                        dec_d=(-dec if sg == '-' else dec)))
    return out


# ─── Coordinate maths ─────────────────────────────────────────────────────────

def source_azel(ra_h, dec_d, t):
    ra  = np.radians(ra_h * 15.0)
    dec = np.radians(dec_d)
    u   = np.array([np.cos(dec)*np.cos(ra), np.cos(dec)*np.sin(ra), np.sin(dec)])
    R      = true_equator_and_equinox_of_date.rotation_at(t)
    ud     = np.einsum('ijn,j->in', R, u)
    ra_d   = np.arctan2(ud[1], ud[0])
    dec_d2 = np.arcsin(ud[2])
    ha  = np.radians(t.gast * 15.0 + GMRT.longitude.degrees) - ra_d
    lat = GMRT.latitude.radians
    alt = np.arcsin(np.sin(lat)*np.sin(dec_d2)
                    + np.cos(lat)*np.cos(dec_d2)*np.cos(ha))
    az  = np.arctan2(-np.cos(dec_d2)*np.sin(ha),
                      np.sin(dec_d2)*np.cos(lat)
                      - np.cos(dec_d2)*np.sin(lat)*np.cos(ha))
    return np.degrees(az) % 360.0, np.degrees(alt)


def azel_sep(az1, el1, az2, el2):
    a1, e1, a2, e2 = map(np.radians, (az1, el1, az2, el2))
    c = np.sin(e1)*np.sin(e2) + np.cos(e1)*np.cos(e2)*np.cos(a1-a2)
    return float(np.degrees(np.arccos(np.clip(c, -1, 1))))


def incl_deg(sat):
    return float(np.degrees(sat.model.inclo))


def get_band_satellites(band: str) -> dict:
    if band == '6':
        merged: dict = {}
        for b in ['1', '2', '3', '4', '5']:
            merged.update(BAND_SATELLITES.get(b, {}))
        return merged
    return dict(BAND_SATELLITES.get(band, {}))


# ─── Main computation ─────────────────────────────────────────────────────────

def run_overlap(source_text, date_str, start_hhmm, obs_hours, sep_deg,
                band='3', tle_upload_text=None, log=print):
    names      = get_band_satellites(band)
    band_label = BAND_LABELS.get(band, f'Band {band}')

    start_ist = datetime.strptime(f'{date_str} {start_hhmm}', '%d%b%Y %H:%M')
    nmin      = int(round(obs_hours * 60))
    nmin_grid = nmin + 1      # include end-point minute (e.g. 08:00)
    mins = [start_ist + timedelta(minutes=i) for i in range(nmin_grid)]
    utc  = [m - IST for m in mins]
    ts   = get_ts()
    t    = ts.utc([u.year for u in utc], [u.month for u in utc],
                  [u.day  for u in utc], [u.hour  for u in utc],
                  [u.minute for u in utc])

    log(f'\nFetching TLEs for {band_label} ...')
    sats = load_satellites(list(names), tle_upload_text, log)

    # Compute source paths regardless of satellite loading (for the sky plot)
    srcs  = parse_sources_text(source_text)
    paths = []
    for sc in srcs:
        saz, sel = source_azel(sc['ra_h'], sc['dec_d'], t)
        paths.append((sc['name'], saz, sel))

    if not sats:
        return ('ERROR: no satellites could be loaded (see log above).\n'
                'Tip: upload a local TLE file or retry after a few minutes.'),\
               paths, {}, names, mins

    # Satellite az/el
    sat_azel: dict = {}
    for n, s in sats.items():
        alt, az, _ = (s - GMRT).at(t).altaz()
        sat_azel[n] = (az.degrees, alt.degrees)
        age = abs(t[0].tt - s.epoch.tt)
        if age > 30:
            log(f'  ⚠ TLE for {names[n]} is {age:.0f} days old — '
                f'predictions may drift slightly.')

    # Build report
    lines = [
        'Summary: source(s) and satellite(s) passes overlap during the observing run', '',
        f'Observing band : {band_label}',
        f'Start (IST)    : {start_ist:%d%b%Y %H:%M}',
        f'Duration       : {obs_hours} h  ({nmin} min)',
        f'Elevation limit: {MIN_EL}°',
        f'Separation     : {sep_deg}°', '',
        'Satellites checked:',
    ]
    for n, s in sats.items():
        lines.append(f'  {names[n]:<28} NORAD {n:6d}  i={incl_deg(s):.1f}°')

    for sc, (saz, sel) in zip(srcs, (p[1:] for p in paths)):
        lines += [
            f"\n{'─'*95}",
            f"{sc['name']}  {sc['ra_txt']}  {sc['dec_txt']}  Epoch {sc['epoch']}",
            f"{'─'*95}",
            (f"{'Date-time (IST)':<17} {'Source':<13} {'Src-Az':>6} {'Src-El':>6}   "
             f"{'Satellite':<28} {'Sat-Az':>6} {'Sat-El':>6}  {'Sep°':>5}"),
            '-'*95,
        ]
        hit_count = 0
        for i in range(nmin_grid):
            if sel[i] < MIN_EL:
                continue
            for n, (az, el) in sat_azel.items():
                if el[i] < MIN_EL:
                    continue
                off = azel_sep(saz[i], sel[i], az[i], el[i])
                if off <= sep_deg:
                    hit_count += 1
                    lines.append(
                        f'{mins[i]:%d%b%Y %H:%M}  {sc["name"]:<13} '
                        f'{saz[i]:6.1f} {sel[i]:6.1f}   '
                        f'{names[n]:<28} {az[i]:6.1f} {el[i]:6.1f}  {off:5.1f}')
        if not hit_count:
            lines.append('  (no satellite passes within separation limit)')
        lines += ['', f'Total overlap time for {sc["name"]} = {hit_count} minute(s)']

    return '\n'.join(lines), paths, sat_azel, names, mins


# ─── Plot ─────────────────────────────────────────────────────────────────────

def make_sky_plot(paths, sat_azel, names, mins, sep_deg, band_label):
    fig = plt.figure(figsize=(8, 8))
    ax  = fig.add_subplot(111, projection='polar')
    ax.set_theta_zero_location('N')
    ax.set_theta_direction(-1)
    ax.set_rlim(0, 90)
    ax.set_rticks([90-MIN_EL, 60, 30])
    ax.set_yticklabels([f'EL {MIN_EL:.0f}', 'EL 30', 'EL 60'], fontsize=8)
    ax.set_xlabel('Azimuth', labelpad=10)
    colors = plt.cm.tab10.colors

    for i, (name, az, el) in enumerate(paths):
        ok = el >= MIN_EL
        if ok.any():
            ax.plot(np.radians(az[ok]), 90-el[ok], lw=2.5,
                    color=colors[i % 10], label=f'◉ {name}')

    for n, (az, el) in sat_azel.items():
        ok = el >= MIN_EL
        if ok.any():
            ax.plot(np.radians(az[ok]), 90-el[ok], lw=1, ls='--',
                    color='grey', alpha=0.6, label=f'-- {names[n]}')

    ax.set_title(
        f'{mins[0]:%d%b%Y %H:%M} – {mins[-1]:%H:%M} IST\n'
        f'{band_label}  |  sep ≤ {sep_deg}°  |  el ≥ {MIN_EL}°',
        pad=14, fontsize=10)
    ax.legend(loc='center left', bbox_to_anchor=(1.08, 0.5), fontsize=7)
    plt.tight_layout()
    return fig


# ─── Streamlit UI ─────────────────────────────────────────────────────────────

def main():
    st.set_page_config(
        page_title="GMRT Satellite Pass Checker",
        page_icon="📡",
        layout="wide",
        initial_sidebar_state="collapsed",
    )

    st.title("📡 Satellite Passes Prediction over the GMRT Antennas")
    st.markdown(
        "Searches for artificial satellites that will pass within a specified "
        "angular radius of your target sources during an observing run. "
        "TLEs are fetched live from [CelesTrak](https://celestrak.org).")
    st.divider()

    # ── Input form ─────────────────────────────────────────────────────────────
    with st.form("params"):
        col_l, col_r = st.columns(2)

        with col_l:
            st.subheader("Observation")
            src_file   = st.file_uploader(
                "Source list file  *(required)*",
                type=['txt', 'list', 'src'],
                help="Plain-text, one source per line. Download the example below.")
            date_str   = st.text_input("Observing date (IST)", value="21Oct2026",
                                        placeholder="e.g. 21Oct2026")
            start_time = st.text_input("Start time HH:MM (IST)", value="03:00",
                                        placeholder="e.g. 03:00")
            obs_hours  = st.number_input("Total observing time (hours)",
                                          min_value=0.5, max_value=24.0,
                                          value=5.0, step=0.5)

        with col_r:
            st.subheader("Band & Geometry")
            band = st.selectbox(
                "Observing Frequency Band (MHz)",
                options=['1', '2', '3', '4', '5', '6'],
                format_func=lambda x: BAND_LABELS[x],
                index=2)
            st.markdown(
                "**Lower elevation limit:** 17°  "
                "<span style='color:grey'>(GMRT fixed)</span>",
                unsafe_allow_html=True)
            sep_deg   = st.number_input(
                "Separation angle / radius (degree)",
                min_value=1.0, max_value=90.0, value=9.0, step=1.0)
            show_plot = st.checkbox("Generate sky-path plot", value=True)

        with st.expander("Advanced — local TLE file (use if CelesTrak download fails)"):
            st.markdown(
                "Download TLEs manually from "
                "[celestrak.org/NORAD/elements](https://celestrak.org/NORAD/elements/) "
                "and upload the `.txt` file here.")
            tle_file = st.file_uploader(
                "TLE file  (leave empty to download automatically)",
                type=['txt', 'tle'])

        col_btn1, col_btn2 = st.columns([3, 1])
        with col_btn1:
            run_btn = st.form_submit_button(
                "🔍  Generate passes", type="primary", use_container_width=True)
        with col_btn2:
            clear_btn = st.form_submit_button(
                "🔄  Clear TLE cache", use_container_width=True,
                help="Forces a fresh TLE download on the next run.")

    # ── Cache clear ────────────────────────────────────────────────────────────
    if clear_btn:
        keys_removed = [k for k in list(st.session_state) if k.startswith('tles_')]
        for k in keys_removed:
            del st.session_state[k]
        st.success(f"TLE cache cleared ({len(keys_removed)} entry/entries). "
                   "Click 'Generate passes' to re-download.")

    # ── Validation ─────────────────────────────────────────────────────────────
    if run_btn:
        errors = []
        if src_file is None:
            errors.append("Please upload a source list file.")
        try:
            datetime.strptime(f"{date_str.strip()} {start_time.strip()}",
                              "%d%b%Y %H:%M")
        except ValueError:
            errors.append("Date must be like `21Oct2026` and time like `03:00`.")
        if not get_band_satellites(band):
            errors.append(
                f"No satellites defined for {BAND_LABELS[band]}. "
                "Add NORAD IDs to BAND_SATELLITES in app.py.")
        if errors:
            for e in errors:
                st.error(e)
            st.stop()

        source_text = src_file.read().decode('utf-8', errors='replace')
        tle_text    = (tle_file.read().decode('utf-8', errors='replace')
                       if tle_file else None)

        if not parse_sources_text(source_text):
            st.error("No valid sources parsed. Check the file format (see help below).")
            st.stop()

        # ── Run ────────────────────────────────────────────────────────────────
        log_msgs: list[str] = []

        def log(msg: str):
            log_msgs.append(msg)

        with st.status("Computing satellite passes …", expanded=True) as status:
            st.write(f"**{BAND_LABELS[band]}** | sep ≤ **{sep_deg}°** | "
                     f"duration **{obs_hours} h**")
            try:
                report, paths, sat_azel, names_map, mins = run_overlap(
                    source_text, date_str.strip(), start_time.strip(),
                    obs_hours, sep_deg, band=band,
                    tle_upload_text=tle_text, log=log)
                for m in log_msgs:
                    st.write(m)
            except Exception as exc:
                for m in log_msgs:
                    st.write(m)
                status.update(label="Error!", state="error")
                st.exception(exc)
                st.stop()

        has_sats = bool(sat_azel)
        status.update(
            label="Done ✓" if has_sats else "Done (no satellites loaded — see log)",
            state="complete" if has_sats else "error",
            expanded=not has_sats)      # leave log open when satellites failed

        # Persist results in session state
        st.session_state.update(dict(
            report=report, paths=paths, sat_azel=sat_azel,
            names_map=names_map, mins=mins,
            band=band, band_label=BAND_LABELS[band],
            sep_deg=sep_deg, show_plot=show_plot,
            date_str=date_str.strip(), start_time=start_time.strip()))

    # ── Results ────────────────────────────────────────────────────────────────
    if 'report' in st.session_state:
        ss  = st.session_state
        tab_rep, tab_plot = st.tabs(["📋 Report", "🌐 Sky-path plot"])

        with tab_rep:
            st.code(ss['report'], language=None)
            fname = (f"gmrt_overlap_{ss['date_str']}_"
                     f"{ss['start_time'].replace(':','')}_band{ss['band']}.txt")
            st.download_button("⬇️ Download report (.txt)",
                               data=ss['report'], file_name=fname,
                               mime='text/plain')

        with tab_plot:
            if ss.get('show_plot') and ss.get('paths'):
                fig = make_sky_plot(
                    ss['paths'], ss['sat_azel'], ss['names_map'],
                    ss['mins'], ss['sep_deg'], ss['band_label'])
                st.pyplot(fig)

                # Note: even when satellite loading failed, source paths still
                # render so the observer can see source sky coverage.
                if not ss['sat_azel']:
                    st.warning("No satellite tracks shown — TLE loading failed. "
                               "Source paths are still visible.")

                buf = io.BytesIO()
                fig.savefig(buf, format='png', dpi=150, bbox_inches='tight')
                buf.seek(0)
                plt.close(fig)
                st.download_button("⬇️ Download plot (.png)",
                                   data=buf,
                                   file_name=fname.replace('.txt', '.png'),
                                   mime='image/png')
            else:
                st.info("Tick 'Generate sky-path plot' and click Generate passes.")

    # ── Help ───────────────────────────────────────────────────────────────────
    st.divider()
    with st.expander("📖  Source file format & example"):
        st.markdown("""
**One source per line:**  `name  RA  Dec  Epoch  [optional extra columns]`

```
# name        RA                Dec               Epoch
3C48          01h37m41.30s      +33d09'35.13"     2000.0
0521-207      05h21m38.90s      -20d47'39.40"     2000.0
J0437-4715    04h37m15.80s      -47d15'09.10"     2000.0   12   2.64016
3C147OFF      05h42m36.14s      +54d51'07.20"     2000.0    1
```

Lines beginning with `#` or `{` are skipped. Extra columns are ignored.
        """)
        example = (
            "3C48            01h37m41.30s  +33d09'35.13\"  2000.0   5\n"
            "0521-207        05h21m38.90s  -20d47'39.40\"  2000.0   5\n"
            "J0437-4715      04h37m15.80s  -47d15'09.10\"  2000.0  12\n"
            "0834+555        08h34m54.90s  +55d34'21.10\"  2000.0   5\n"
            "J0740+6620      07h40m45.70s  +66d20'33.50\"  2000.0  27\n"
            "J0613-0200      06h13m43.90s  -02d00'47.20\"  2000.0  22\n"
            "J1022+1001      10h22m57.90s  +10d01'52.70\"  2000.0  12\n"
            "3C147           05h42m36.14s  +49d51'07.20\"  2000.0   5\n"
            "3C147OFF        05h42m36.14s  +54d51'07.20\"  2000.0   1\n"
        )
        st.download_button("⬇️ Download example source file",
                           data=example, file_name="example_sources.txt",
                           mime='text/plain')

    with st.expander("🛠  Troubleshooting — TLE download failures"):
        st.markdown("""
**'ERROR: no satellites loaded'** means all CelesTrak requests failed.  Try:

1. **Wait and retry** — CelesTrak occasionally has short outages.
2. **Click 'Clear TLE cache'** then click 'Generate passes' again.
3. **Upload a local TLE file** — download manually:
   - [geo.txt](https://celestrak.org/NORAD/elements/geo.txt)
   - [military.txt](https://celestrak.org/NORAD/elements/military.txt)
   - [active.txt](https://celestrak.org/NORAD/elements/active.txt)  
   Then upload it via the *Advanced* expander.
4. **Run locally** (`streamlit run app.py`) — your own machine's IP is
   not subject to cloud-hosting rate limits.
        """)


if __name__ == '__main__':
    main()
