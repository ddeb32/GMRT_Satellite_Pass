"""
GMRT Satellite Pass / Source Overlap Tool — Streamlit web application
Replicates the GMRT internal satellite-pass prediction tool.

Author: built with the GMRT pulsar-timing team
Deploy: https://share.streamlit.io  (connect your GitHub repo, point to app.py)
"""

import re
import io
import requests
import streamlit as st
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from skyfield.api import wgs84, EarthSatellite, load
from skyfield.framelib import true_equator_and_equinox_of_date

# ─── Observatory & time constants ─────────────────────────────────────────────
GMRT   = wgs84.latlon(19.0964, 74.0497, elevation_m=650)   # Khodad
IST    = timedelta(hours=5, minutes=30)
MIN_EL = 17.0   # GMRT hard lower elevation limit (degrees)

# ─── Satellite lists per GMRT frequency band ──────────────────────────────────
# Band 3 is verified against ww1.gmrt.ncra.tifr.res.in/~astrosupp/sat/pre/sat.html
# Other bands are best-effort; cross-check with the GMRT web tool.
BAND_SATELLITES = {
    '1': {   # 50 – 80 MHz  (populate after verifying with GMRT web tool)
    },
    '2': {   # 125 – 250 MHz
        25338: 'NOAA 15',       # APT 137.500 MHz
        28654: 'NOAA 18',       # APT 137.9125 MHz
        33591: 'NOAA 19',       # APT 137.100 MHz
        43013: 'NOAA 20',       # APT 137.350 MHz
        47438: 'NOAA 21',       # APT 137.350 MHz
        39444: 'METEOR-M 2',    # APT 137.100 MHz
        44387: 'METEOR-M2-2',   # APT 137.900 MHz
    },
    '3': {   # 250 – 500 MHz  ← verified from GMRT website
        37212: 'MERIDIAN 3',
        38995: 'MERIDIAN 6',
        44453: 'MERIDIAN 8',
        22563: 'UFO 1 (USA 98)',
        22787: 'UFO 2 (USA 95)',
        22921: 'NATO 4B',
        32294: 'SKYNET 5B',
        35943: 'COMSATBW-1',
        40887: 'MUOS 4',
    },
    '4': {   # 550 – 900 MHz
        22076: 'UFO 3 (USA 108)',
        23170: 'UFO 4 (USA 113)',
        23467: 'UFO 5 (USA 118)',
        24315: 'UFO 6 (USA 130)',
        25272: 'UFO 7 (USA 143)',
        26063: 'UFO 8 (USA 164)',
        27597: 'UFO 10 (USA 183)',
        28897: 'UFO 11 (USA 201)',
    },
    '5': {   # 1000 – 1450 MHz
        23839: 'INMARSAT 3-F4',
        25153: 'INMARSAT 3-F2',
        24958: 'INMARSAT 3-F3',
        24307: 'INMARSAT 3-F5',
        28628: 'INMARSAT 4-F2',
        26624: 'INMARSAT 4-F1',
        32791: 'INMARSAT 4-F3',
        39476: 'INMARSAT-5 F1',
        40732: 'INMARSAT-5 F2',
        41838: 'INMARSAT-5 F3',
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

FALLBACK_CATALOGS = [
    ('geo',      'https://celestrak.org/NORAD/elements/geo.txt'),
    ('military', 'https://celestrak.org/NORAD/elements/military.txt'),
    ('active',   'https://celestrak.org/NORAD/elements/active.txt'),
]

RA_RE  = re.compile(r'(\d+)h(\d+)m([\d.]+)s')
DEC_RE = re.compile(r'([+-])(\d+)d(\d+)\'([\d.]+)"')


# ─── Cached resources (shared across users / sessions) ────────────────────────

@st.cache_resource
def get_timescale():
    """Skyfield timescale — built-in so no network call needed."""
    return load.timescale(builtin=True)


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_tle_text(norad: int) -> str | None:
    """Fetch TLE text for one NORAD ID from CelesTrak. Cached 1 h."""
    url = (f"https://celestrak.org/NORAD/elements/"
           f"gp.php?CATNR={norad}&FORMAT=tle")
    try:
        r = requests.get(url, timeout=15)
        if r.ok and r.text.strip():
            return r.text
    except Exception:
        pass
    return None


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_catalog_text(cat_url: str) -> str | None:
    """Download a bulk CelesTrak catalog file. Cached 1 h."""
    try:
        r = requests.get(cat_url, timeout=45)
        if r.ok:
            return r.text
    except Exception:
        pass
    return None


# ─── Core helpers ─────────────────────────────────────────────────────────────

def get_band_satellites(band: str) -> dict:
    if band == '6':
        merged: dict = {}
        for b in ['1', '2', '3', '4', '5']:
            merged.update(BAND_SATELLITES.get(b, {}))
        return merged
    return dict(BAND_SATELLITES.get(band, {}))


def parse_tle_block(text: str) -> dict:
    """Parse a TLE text block into {norad_int: EarthSatellite}."""
    ts    = get_timescale()
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


def source_azel(ra_h: float, dec_d: float, t) -> tuple:
    ra  = np.radians(ra_h * 15.0)
    dec = np.radians(dec_d)
    u   = np.array([np.cos(dec)*np.cos(ra),
                    np.cos(dec)*np.sin(ra),
                    np.sin(dec)])
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


def azel_sep(az1, el1, az2, el2) -> float:
    a1, e1, a2, e2 = map(np.radians, (az1, el1, az2, el2))
    c = np.sin(e1)*np.sin(e2) + np.cos(e1)*np.cos(e2)*np.cos(a1-a2)
    return float(np.degrees(np.arccos(np.clip(c, -1, 1))))


def incl_deg(sat) -> float:
    return float(np.degrees(sat.model.inclo))


def load_satellites(norads: list, tle_upload_text: str | None,
                    log: callable) -> dict:
    """
    Load EarthSatellite objects for the given NORAD IDs.
    Strategy: per-satellite CelesTrak query → catalog fallback.
    """
    sats: dict = {}

    if tle_upload_text:
        parsed = parse_tle_block(tle_upload_text)
        for n in norads:
            if n in parsed:
                sats[n] = parsed[n]
                log(f"  ✓ NORAD {n:6d}  {_ALL_SAT_NAMES.get(n,'')} [uploaded TLE]")
            else:
                log(f"  ✗ NORAD {n:6d}  {_ALL_SAT_NAMES.get(n,'')} — not in uploaded TLE")
        return sats

    # Step 1 — per-satellite queries
    failed = []
    for n in norads:
        text = fetch_tle_text(n)
        if text:
            parsed = parse_tle_block(text)
            if n in parsed:
                sats[n] = parsed[n]
                log(f"  ✓ NORAD {n:6d}  {_ALL_SAT_NAMES.get(n,'')}")
                continue
        failed.append(n)
        log(f"  ✗ NORAD {n:6d}  {_ALL_SAT_NAMES.get(n,'')} — query failed, trying catalog fallback")

    # Step 2 — catalog fallback
    if failed:
        log(f"\n  Catalog fallback for NORAD {failed} ...")
        for cat_name, cat_url in FALLBACK_CATALOGS:
            if not failed:
                break
            log(f"  ↻ Downloading {cat_name} catalog ...")
            cat_text = fetch_catalog_text(cat_url)
            if cat_text:
                by_norad  = parse_tle_block(cat_text)
                still_bad = []
                for n in failed:
                    if n in by_norad:
                        sats[n] = by_norad[n]
                        log(f"  ✓ NORAD {n:6d}  {_ALL_SAT_NAMES.get(n,'')} [found in {cat_name}]")
                    else:
                        still_bad.append(n)
                failed = still_bad
            else:
                log(f"  ✗ {cat_name} catalog download failed")
        if failed:
            log(f"\n  ✗ WARNING: NORAD {failed} not found — skipped.")
    return sats


def run_overlap(source_text: str, date_str: str, start_hhmm: str,
                obs_hours: float, sep_deg: float, band: str = '3',
                tle_upload_text: str | None = None,
                log: callable = print) -> tuple:
    """
    Main computation.
    Returns (report_str, paths, sat_azel, names, mins).
    """
    names      = get_band_satellites(band)
    band_label = BAND_LABELS.get(band, f'Band {band}')

    start_ist = datetime.strptime(f'{date_str} {start_hhmm}', '%d%b%Y %H:%M')
    nmin      = int(round(obs_hours * 60))
    nmin_grid = nmin + 1            # include window end-point (e.g. 08:00)
    mins = [start_ist + timedelta(minutes=i) for i in range(nmin_grid)]
    utc  = [m - IST for m in mins]
    ts   = get_timescale()
    t    = ts.utc([u.year for u in utc], [u.month for u in utc],
                  [u.day  for u in utc], [u.hour  for u in utc],
                  [u.minute for u in utc])

    log(f'\nFetching TLEs from CelesTrak for {band_label} ...')
    sats = load_satellites(list(names), tle_upload_text, log)
    if not sats:
        return 'ERROR: no satellites loaded.', [], {}, names, mins

    sat_azel: dict = {}
    for n, s in sats.items():
        alt, az, _ = (s - GMRT).at(t).altaz()
        sat_azel[n] = (az.degrees, alt.degrees)
        age = abs(t[0].tt - s.epoch.tt)
        if age > 30:
            log(f'  ⚠ TLE for {names[n]} is {age:.0f} days old — predictions may drift.')

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

    srcs  = parse_sources_text(source_text)
    paths = []

    for sc in srcs:
        saz, sel = source_azel(sc['ra_h'], sc['dec_d'], t)
        paths.append((sc['name'], saz, sel))
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


# ─── Streamlit page ───────────────────────────────────────────────────────────

def main():
    st.set_page_config(
        page_title="GMRT Satellite Pass Checker",
        page_icon="📡",
        layout="wide",
        initial_sidebar_state="collapsed",
    )

    st.title("📡 Satellite Passes Prediction over the GMRT Antennas")
    st.markdown(
        "This tool searches for artificial satellites which will pass over the GMRT "
        "antennas/source during the observing run in the region specified by radius "
        "and time window. TLEs are fetched from [CelesTrak](https://celestrak.org).")
    st.divider()

    # ── Input form ─────────────────────────────────────────────────────────────
    with st.form("params"):
        col_left, col_right = st.columns(2)

        with col_left:
            st.subheader("Observation")
            src_file = st.file_uploader(
                "Source list file  *(required)*",
                type=['txt', 'list', 'src'],
                help="Plain-text file, one source per line.")
            date_str   = st.text_input("Observing date (IST)", value="21Oct2026",
                                        placeholder="e.g. 21Oct2026")
            start_time = st.text_input("Start time HH:MM (IST)", value="03:00",
                                        placeholder="e.g. 03:00")
            obs_hours  = st.number_input("Total observing time (hours)",
                                          min_value=0.5, max_value=24.0,
                                          value=5.0, step=0.5)

        with col_right:
            st.subheader("Band & Geometry")
            band = st.selectbox(
                "Observing Frequency Band (MHz)",
                options=['1', '2', '3', '4', '5', '6'],
                format_func=lambda x: BAND_LABELS[x],
                index=2,
                help="Selects the pre-defined satellite list for that band.")
            st.markdown(
                f"**Lower elevation limit:** 17°  "
                f"<span style='color:grey'>(GMRT fixed)</span>",
                unsafe_allow_html=True)
            sep_deg = st.number_input(
                "Separation angle / radius (degree)",
                min_value=1.0, max_value=90.0, value=9.0, step=1.0,
                help="Minimum angular distance between source and satellite.")
            show_plot = st.checkbox("Generate sky-path plot", value=True)

        with st.expander("Advanced — local TLE file (optional)"):
            tle_file = st.file_uploader(
                "TLE file (leave empty to download from CelesTrak)",
                type=['txt', 'tle'])

        run_btn = st.form_submit_button(
            "🔍  Generate passes", type="primary", use_container_width=True)

    # ── Validation & run ───────────────────────────────────────────────────────
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
            st.error("No valid sources parsed. Check the file format (see below).")
            st.stop()

        # Live log shown inside the status widget
        log_msgs: list = []

        def log(msg):
            log_msgs.append(msg)

        with st.status("Computing satellite passes …", expanded=True) as status:
            st.write(f"Band: **{BAND_LABELS[band]}** | "
                     f"Sep: **{sep_deg}°** | "
                     f"Duration: **{obs_hours} h**")
            try:
                report, paths, sat_azel, names_map, mins = run_overlap(
                    source_text, date_str.strip(), start_time.strip(),
                    obs_hours, sep_deg, band=band,
                    tle_upload_text=tle_text, log=log)
                for m in log_msgs:
                    st.write(m)
            except Exception as exc:
                status.update(label="Error!", state="error")
                st.exception(exc)
                st.stop()

        status.update(label="Done ✓", state="complete", expanded=False)

        # Persist results across Streamlit reruns
        st.session_state.update(dict(
            report=report, paths=paths, sat_azel=sat_azel,
            names_map=names_map, mins=mins,
            band=band, band_label=BAND_LABELS[band],
            sep_deg=sep_deg, show_plot=show_plot,
            date_str=date_str.strip(), start_time=start_time.strip(),
        ))

    # ── Results ────────────────────────────────────────────────────────────────
    if 'report' in st.session_state:
        ss = st.session_state
        st.divider()
        tab_rep, tab_plot = st.tabs(["📋 Report", "🌐 Sky-path plot"])

        with tab_rep:
            st.code(ss['report'], language=None)
            fname = (f"gmrt_overlap_{ss['date_str']}_"
                     f"{ss['start_time'].replace(':','')}_band{ss['band']}.txt")
            st.download_button("⬇️ Download report (.txt)",
                               data=ss['report'],
                               file_name=fname,
                               mime='text/plain')

        with tab_plot:
            if ss.get('show_plot') and ss['paths']:
                fig = make_sky_plot(
                    ss['paths'], ss['sat_azel'], ss['names_map'],
                    ss['mins'], ss['sep_deg'], ss['band_label'])
                st.pyplot(fig)
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
    with st.expander("📖  Source file format"):
        st.markdown("""
Each non-comment line must contain four fields separated by whitespace:

```
# name        RA                Dec               Epoch
3C48          01h37m41.30s      +33d09'35.13"     2000.0
0521-207      05h21m38.90s      -20d47'39.40"     2000.0
J0437-4715    04h37m15.80s      -47d15'09.10"     2000.0
```

Lines beginning with `#` or `{` are skipped.
Coordinates are always treated as **J2000**.
Extra columns after the epoch (scan duration, period, …) are accepted and ignored.
        """)
        example = (
            "3C48            01h37m41.30s  +33d09'35.13\"  2000.0\n"
            "0521-207        05h21m38.90s  -20d47'39.40\"  2000.0\n"
            "J0437-4715      04h37m15.80s  -47d15'09.10\"  2000.0  12\n"
            "0834+555        08h34m54.90s  +55d34'21.10\"  2000.0\n"
        )
        st.download_button("⬇️ Download example source file",
                           data=example,
                           file_name="example_sources.txt",
                           mime='text/plain')


if __name__ == '__main__':
    main()
