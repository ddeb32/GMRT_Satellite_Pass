# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  GMRT Satellite Pass / Source Overlap Tool                                  ║
# ║  Google Colab version                                                        ║
# ║                                                                              ║
# ║  HOW TO USE                                                                  ║
# ║  ─────────                                                                   ║
# ║  1. Open in Colab (File → Open notebook → paste the GitHub raw URL, or      ║
# ║     File → Upload notebook if you have the .ipynb)                           ║
# ║  2. Runtime → Run all  (or Ctrl+F9)                                          ║
# ║  3. The form appears at the bottom — upload your source file and fill in     ║
# ║     the observation details, then click "Generate passes".                   ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

# ── Cell 1: install once per session ──────────────────────────────────────────
# In Colab, run this cell first (Runtime → Run all does it automatically).
# Packages persist for the life of the runtime (~12 h free tier).

import subprocess, sys
subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q',
                        'skyfield', 'numpy', 'matplotlib', 'ipywidgets'])
print('All dependencies ready. ✓')


# ── Cell 2: core computation helpers ──────────────────────────────────────────
# Identical to the local Jupyter version — no changes needed here.

import re, io
from datetime import datetime, timedelta
import numpy as np
from skyfield.api import load, wgs84
from skyfield.framelib import true_equator_and_equinox_of_date

GMRT   = wgs84.latlon(19.0964, 74.0497, elevation_m=650)
IST    = timedelta(hours=5, minutes=30)
MIN_EL = 17.0

BAND_SATELLITES = {
    '1': {},
    '2': {
        25338: 'NOAA 15',     28654: 'NOAA 18',     33591: 'NOAA 19',
        43013: 'NOAA 20',     47438: 'NOAA 21',
        39444: 'METEOR-M 2',  44387: 'METEOR-M2-2',
    },
    '3': {   # verified from GMRT website
        37212: 'MERIDIAN 3',  38995: 'MERIDIAN 6',  44453: 'MERIDIAN 8',
        22563: 'UFO 1 (USA 98)', 22787: 'UFO 2 (USA 95)', 22921: 'NATO 4B',
        32294: 'SKYNET 5B',   35943: 'COMSATBW-1',  40887: 'MUOS 4',
    },
    '4': {
        22076: 'UFO 3 (USA 108)', 23170: 'UFO 4 (USA 113)',
        23467: 'UFO 5 (USA 118)', 24315: 'UFO 6 (USA 130)',
        25272: 'UFO 7 (USA 143)', 26063: 'UFO 8 (USA 164)',
        27597: 'UFO 10 (USA 183)', 28897: 'UFO 11 (USA 201)',
    },
    '5': {
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

_FALLBACK_CATALOGS = [
    ('geo',      'https://celestrak.org/NORAD/elements/geo.txt'),
    ('military', 'https://celestrak.org/NORAD/elements/military.txt'),
    ('active',   'https://celestrak.org/NORAD/elements/active.txt'),
]

RA_RE  = re.compile(r'(\d+)h(\d+)m([\d.]+)s')
DEC_RE = re.compile(r'([+-])(\d+)d(\d+)\'([\d.]+)"')


def get_band_satellites(band):
    if band == '6':
        merged = {}
        for b in ['1', '2', '3', '4', '5']:
            merged.update(BAND_SATELLITES.get(b, {}))
        return merged
    return dict(BAND_SATELLITES.get(band, {}))


def parse_sources_text(text):
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


def load_satellites(norads, tle_path=None, log=print):
    sats = {}
    if tle_path:
        for s in load.tle_file(tle_path):
            if s.model.satnum in norads:
                sats[s.model.satnum] = s
        missing = [n for n in norads if n not in sats]
        if missing:
            log(f'  ✗ WARNING: NORAD {missing} not found in {tle_path}')
        return sats

    failed = []
    for n in norads:
        url = f'https://celestrak.org/NORAD/elements/gp.php?CATNR={n}&FORMAT=tle'
        try:
            result = load.tle_file(url, filename=f'celestrak_{n}.tle', reload=True)
            if result:
                sats[n] = result[0]
                log(f'  ✓ NORAD {n:6d}  {_ALL_SAT_NAMES.get(n, "")}')
            else:
                failed.append(n)
                log(f'  ✗ NORAD {n:6d} — empty, trying catalog fallback')
        except Exception as e:
            failed.append(n)
            log(f'  ✗ NORAD {n:6d} — failed ({e}), trying catalog fallback')

    if failed:
        log(f'\n  Catalog fallback for NORAD {failed} ...')
        for cat_name, cat_url in _FALLBACK_CATALOGS:
            if not failed:
                break
            try:
                log(f'  ↻ Downloading {cat_name} catalog ...')
                all_sats  = load.tle_file(cat_url,
                                           filename=f'celestrak_{cat_name}.txt',
                                           reload=True)
                by_norad  = {s.model.satnum: s for s in all_sats}
                still_bad = []
                for n in failed:
                    if n in by_norad:
                        sats[n] = by_norad[n]
                        log(f'  ✓ NORAD {n:6d}  {_ALL_SAT_NAMES.get(n, "")} '
                            f'[found in {cat_name}]')
                    else:
                        still_bad.append(n)
                failed = still_bad
            except Exception as e:
                log(f'  ✗ {cat_name} catalog download failed: {e}')
        if failed:
            log(f'\n  ✗ WARNING: NORAD {failed} not found — skipped.')
    return sats


def run_overlap(source_text, date_str, start_hhmm, obs_hours, sep_deg,
                band='3', tle_path=None, log=print):
    names      = get_band_satellites(band)
    band_label = BAND_LABELS.get(band, f'Band {band}')

    start_ist = datetime.strptime(f'{date_str} {start_hhmm}', '%d%b%Y %H:%M')
    nmin      = int(round(obs_hours * 60))
    nmin_grid = nmin + 1
    mins = [start_ist + timedelta(minutes=i) for i in range(nmin_grid)]
    utc  = [m - IST for m in mins]
    ts   = load.timescale()
    t    = ts.utc([u.year for u in utc], [u.month for u in utc],
                  [u.day  for u in utc], [u.hour  for u in utc],
                  [u.minute for u in utc])

    log(f'\nFetching TLEs for {band_label} ...')
    sats = load_satellites(list(names), tle_path, log)
    if not sats:
        return 'ERROR: no satellites loaded.', [], {}, names, mins

    sat_azel = {}
    for n, s in sats.items():
        alt, az, _ = (s - GMRT).at(t).altaz()
        sat_azel[n] = (az.degrees, alt.degrees)
        age = abs(t[0].tt - s.epoch.tt)
        if age > 30:
            log(f'  ⚠ TLE for {names[n]} is {age:.0f} days old.')

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


print('Core helpers loaded. ✓')


# ── Cell 3: interactive form ───────────────────────────────────────────────────
#
# KEY DIFFERENCE vs local Jupyter:
#   Source file is uploaded via Colab's native file picker (google.colab.files)
#   instead of typing a file path. Everything else is identical.

import ipywidgets as widgets
from IPython.display import display, Image as IPyImage
import os

# ── detect whether we are in Colab ──────────────────────────────────────────
try:
    from google.colab import files as colab_files
    IN_COLAB = True
except ImportError:
    IN_COLAB = False


_label_style = {'description_width': '220px'}
_box_layout  = widgets.Layout(width='540px')

# ── Band dropdown ────────────────────────────────────────────────────────────
def _band_hint(band):
    n     = len(get_band_satellites(band))
    label = BAND_LABELS.get(band, '')
    note  = ('' if band == '3'
             else '  <span style="color:orange">⚠ list approximate</span>')
    if n == 0:
        note = '  <span style="color:red">⚠ no satellites defined</span>'
    return (f'<div style="font-size:0.85em;color:#555;padding:2px 0 4px 225px">'
            f'{label} · {n} satellite(s){note}</div>')

def _band_hint(band):
    n     = len(get_band_satellites(band))
    label = BAND_LABELS.get(band, '')
    note  = ('' if band == '3'
             else '  ⚠ list approximate')
    if n == 0:
        note = '  ⚠ no satellites defined'
    return f'{label} · {n} satellite(s){note}'

w_band = widgets.Dropdown(
    description='Observing band (MHz):',
    options=[
        ('1 : 50 – 80 MHz',         '1'),
        ('2 : 125 – 250 MHz',       '2'),
        ('3 : 250 – 500 MHz',       '3'),
        ('4 : 550 – 900 MHz',       '4'),
        ('5 : 1000 – 1450 MHz',     '5'),
        ('6 : All  (1 – 1500 MHz)', '6'),
    ],
    value='3',
    style=_label_style,
    layout=widgets.Layout(width='420px'))

w_band_hint = widgets.Label(value=_band_hint('3'))

def _on_band_change(change):
    w_band_hint.value = _band_hint(change['new'])

w_band.observe(_on_band_change, names='value')

# ── Other input widgets ──────────────────────────────────────────────────────
w_date = widgets.Text(
    description='Observing date (IST):',
    placeholder='e.g. 21Oct2026', value='21Oct2026',
    style=_label_style, layout=_box_layout)

w_start = widgets.Text(
    description='Start time HH:MM (IST):',
    placeholder='e.g. 03:00', value='03:00',
    style=_label_style, layout=_box_layout)

w_hours = widgets.BoundedFloatText(
    description='Total observing time (h):',
    value=8, min=0.5, max=24, step=0.5,
    style=_label_style, layout=widgets.Layout(width='360px'))

w_el = widgets.HTML(
    value=('<div style="padding:3px 0;font-family:monospace">'
           '<b>Lower elevation limit:</b> 17° '
           '<span style="color:grey">[GMRT fixed]</span></div>'))

w_sep = widgets.BoundedFloatText(
    description='Separation angle (°):',
    value=15, min=1, max=90, step=1,
    style=_label_style, layout=widgets.Layout(width='360px'))

w_tle = widgets.Text(
    description='Local TLE file (optional):',
    placeholder='leave blank to download from CelesTrak',
    style=_label_style, layout=_box_layout)

w_plot = widgets.Checkbox(
    description='Generate sky-path plot', value=True, style=_label_style)

# ── Source file section — adapts to Colab vs local ─────────────────────────
if IN_COLAB:
    # In Colab: a button triggers google.colab.files.upload()
    w_upload_btn = widgets.Button(
        description='📂 Upload source file',
        button_style='info',
        layout=widgets.Layout(width='220px', height='36px'))
    w_src_status = widgets.Label(value='No file uploaded yet')
    _src_text = ['']

    def on_upload_click(b):
        uploaded = colab_files.upload()   # opens the native Colab file picker
        if not uploaded:
            return
        fname   = list(uploaded.keys())[0]
        content = list(uploaded.values())[0]
        _src_text[0] = content.decode('utf-8', errors='replace')
        n_src = len(parse_sources_text(_src_text[0]))
        w_src_status.value = f'✓  {fname}  —  {n_src} source(s) found'

    w_upload_btn.on_click(on_upload_click)
    src_section = widgets.VBox([
        widgets.HTML('<b>Source list file</b>'),
        widgets.HBox([w_upload_btn, w_src_status]),
    ])
else:
    # Local Jupyter: classic FileUpload widget
    w_upload = widgets.FileUpload(
        description='Upload source file',
        accept='.txt,.list,.src,',
        multiple=False,
        layout=widgets.Layout(width='260px'))
    w_src_status = widgets.HTML(
        value='<span style="color:grey">No file uploaded yet</span>')
    _src_text = ['']

    def on_upload_change(change):
        if not w_upload.value:
            return
        v = w_upload.value
        if isinstance(v, dict):
            fname   = list(v.keys())[0]
            content = v[fname]['content']
        else:
            fname   = v[0]['name']
            content = v[0]['content']
        _src_text[0] = content.decode('utf-8', errors='replace')
        n_src = len(parse_sources_text(_src_text[0]))
        w_src_status.value = (f'<span style="color:green">✓ <b>{fname}</b> — '
                               f'{n_src} source(s) found</span>')

    w_upload.observe(on_upload_change, names='value')
    src_section = widgets.VBox([
        widgets.HTML('<b>Source list file</b>'),
        widgets.HBox([w_upload, widgets.HTML('&nbsp;&nbsp;'), w_src_status]),
    ])

# ── Run / save buttons ───────────────────────────────────────────────────────
btn_run  = widgets.Button(description='Generate passes', button_style='primary',
                           layout=widgets.Layout(width='180px', height='36px'))
btn_save = widgets.Button(description='Save report (.txt)', button_style='',
                           layout=widgets.Layout(width='180px', height='36px'),
                           disabled=True)

w_out_log  = widgets.Output()
w_out_rep  = widgets.Output()
w_out_plot = widgets.Output()
_last_report = ['']


def on_run(b):
    w_out_log.clear_output()
    w_out_rep.clear_output()
    w_out_plot.clear_output()
    btn_save.disabled = True

    src_text = _src_text[0]
    date     = w_date.value.strip()
    start    = w_start.value.strip()
    hours    = float(w_hours.value)
    sep      = float(w_sep.value)
    band     = w_band.value
    tle      = w_tle.value.strip() or None

    with w_out_log:
        if not src_text.strip():
            print('⛔ No source file uploaded. Click "Upload source file" first.')
            return
        try:
            datetime.strptime(f'{date} {start}', '%d%b%Y %H:%M')
        except ValueError:
            print('⛔ Date must be like 21Oct2026 and time like 03:00')
            return
        if not get_band_satellites(band):
            print(f'⛔ No satellites defined for {BAND_LABELS[band]}.')
            return
        print(f'Running overlap check ...')
        print(f'  Date/time : {date}  {start}  IST')
        print(f'  Duration  : {hours} h')
        print(f'  Band      : {BAND_LABELS[band]}')
        print(f'  Sep limit : {sep}°')

    try:
        report, paths, sat_azel, names, mins = run_overlap(
            src_text, date, start, hours, sep, band=band, tle_path=tle,
            log=lambda msg: _log(msg))
    except Exception as e:
        with w_out_log:
            import traceback
            print(f'\n⛔ Error: {e}')
            traceback.print_exc()
        return

    _last_report[0] = report
    btn_save.disabled = False

    with w_out_rep:
        print(report)

    if w_plot.value and paths:
        with w_out_plot:
            _draw_plot(paths, sat_azel, names, mins, sep, BAND_LABELS.get(band))


def _log(msg):
    with w_out_log:
        print(msg)


def _draw_plot(paths, sat_azel, names, mins, sep, band_label):
    import matplotlib.pyplot as plt
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
        f'{band_label}  |  sep ≤ {sep}°  |  el ≥ {MIN_EL}°',
        pad=14, fontsize=10)
    ax.legend(loc='center left', bbox_to_anchor=(1.08, 0.5), fontsize=7)
    plt.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=130, bbox_inches='tight')
    buf.seek(0)
    plt.close(fig)
    display(IPyImage(data=buf.read()))


def on_save(b):
    fname = (f'gmrt_overlap_{w_date.value}_'
             f'{w_start.value.replace(":","")}_band{w_band.value}.txt')
    with open(fname, 'w') as f:
        f.write(_last_report[0] + '\n')
    with w_out_log:
        print(f'\n✅ Report saved → {fname}')
    # In Colab, also offer download to local machine
    if IN_COLAB:
        colab_files.download(fname)


btn_run.on_click(on_run)
btn_save.on_click(on_save)

hr = lambda: widgets.HTML('<hr style="border:1px solid #ccc;margin:6px 0">')

form = widgets.VBox([
    widgets.HTML(
        '<h3 style="margin-bottom:2px">GMRT Satellite Pass / Source Overlap Checker</h3>'
        '<p style="color:grey;margin-top:0">Khodad, India</p>'),
    hr(),
    src_section,
    hr(),
    widgets.HTML('<b>Observation window</b>'),
    w_date, w_start, w_hours,
    hr(),
    widgets.HTML('<b>Band & geometry</b>'),
    w_band, w_band_hint, w_el, w_sep,
    hr(),
    w_tle, w_plot,
    hr(),
    widgets.HBox([btn_run, widgets.HTML('&nbsp;&nbsp;'), btn_save]),
    hr(),
    widgets.HTML('<b>Log</b>'), w_out_log,
    widgets.HTML('<b>Report</b>'), w_out_rep,
    widgets.HTML('<b>Sky-path plot</b>'), w_out_plot,
], layout=widgets.Layout(padding='14px', width='800px',
                          border='1px solid #ddd', border_radius='6px'))

display(form)
