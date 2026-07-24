"""
Tiny READ-ONLY metrics dashboard for the fpc3 DE search, so you don't have to SSH in to
watch progress. Reads fpc3_gain_de_opt<TAG>/{optimization_log.csv, optimized_params.json,
driver.out, live_s11.png} and serves an auto-refreshing HTML page:
  * best-so-far card (score / realized gain / worst-in-band S11, with a matched/unmatched badge)
  * convergence chart: worst-in-band S11 and score per eval + running best, with -8/-10 dB lines
  * the best design's live S11 / realized-gain plot
  * a recent-eval table (NEW BEST rows highlighted)

Stdlib http.server + matplotlib (Agg) only -- no Flask, no pip. It never writes anything.

Run on the instance:
    FPC_TAG=_v3 nohup python3 fpc3_dashboard.py > dashboard.out 2>&1 &
Then open  http://<INSTANCE_PUBLIC_IP>:8080/

SECURITY: this binds 0.0.0.0 so it's reachable. Open the port in the EC2 security group to
YOUR IP only. It's read-only, non-sensitive data, but don't expose it to 0.0.0.0/0.
"""

import os
import io
import re
import csv
import glob
import json
import base64
import time
import html
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

TAG     = os.environ.get('FPC_TAG', '_v3')
PORT    = int(os.environ.get('FPC_DASH_PORT', '8080'))
REFRESH = int(os.environ.get('FPC_DASH_REFRESH', '30'))
NRTS    = int(os.environ.get('FPC_DASH_NRTS', '300000'))   # match the optimizer's EVAL_NRTS cap
DIR     = os.path.join(os.getcwd(), 'fpc3_gain_de_opt' + TAG)


def read_log():
    rows = []
    try:
        with open(os.path.join(DIR, 'optimization_log.csv')) as fh:
            for row in csv.DictReader(fh):
                rows.append(row)
    except Exception:
        pass
    return rows


def read_json():
    try:
        return json.load(open(os.path.join(DIR, 'optimized_params.json')))
    except Exception:
        return {}


def b64_file(path):
    try:
        return base64.b64encode(open(path, 'rb').read()).decode()
    except Exception:
        return None


def tail(path, n=25):
    try:
        with open(path, encoding='utf-8', errors='ignore') as fh:
            return ''.join(fh.readlines()[-n:])
    except Exception:
        return ''


def instance_uptime():
    """Seconds since the instance booted (Linux /proc/uptime); None if unavailable."""
    try:
        with open('/proc/uptime') as fh:
            return float(fh.read().split()[0])
    except Exception:
        return None


def fmt_dur(s):
    if s is None:
        return 'n/a'
    d, r = divmod(int(s), 86400)
    h, r = divmod(r, 3600)
    m, _ = divmod(r, 60)
    out = (['%dd' % d] if d else []) + (['%dh' % h] if d or h else []) + ['%dm' % m]
    return ' '.join(out)


_TS_RE = re.compile(r'Timestep:\s*(\d+).*?Energy:.*?\(-\s*([0-9.]+)dB\)')


def worker_progress():
    """Per-worker eval progress from the openEMS logs: current timestep + energy-decay dB.
    An eval finishes at the -30 dB EndCriteria or the NRTS cap, so % done = how far it is
    toward whichever it hits first."""
    out = []
    now = time.time()
    for lf in glob.glob(os.path.join(DIR, 'openems_logs', 'worker_*.log')):
        try:
            age = now - os.path.getmtime(lf)
            with open(lf, 'rb') as fh:              # only the tail holds the latest line
                fh.seek(0, 2); size = fh.tell()
                fh.seek(max(0, size - 6000))
                chunk = fh.read().decode('utf-8', 'ignore')
            m = None
            for mm in _TS_RE.finditer(chunk):
                m = mm
            if not m:
                continue
            ts, db = int(m.group(1)), float(m.group(2))
            pid = re.search(r'worker_(\d+)', lf).group(1)
            pct = int(100 * min(1.0, max(min(db / 30.0, 1.0), ts / NRTS)))
            out.append(dict(pid=pid, ts=ts, db=db, age=age, pct=pct))
        except Exception:
            continue
    return out


def workers_card():
    ws = [w for w in worker_progress() if w['age'] < 180]     # "active" = log touched recently
    if not ws:
        return ('<div class="card"><h2>workers</h2>'
                '<i>no active worker logs (idle / between evals / post-processing)</i></div>')
    ws.sort(key=lambda w: -w['pct'])
    rows = []
    for w in ws:
        col = '#2ea043' if w['pct'] >= 90 else ('#1f6feb' if w['pct'] >= 40 else '#8957e5')
        bar = ('<div style="background:#30363d;border-radius:4px;height:14px;width:200px;'
               'display:inline-block;vertical-align:middle"><div style="background:%s;height:14px;'
               'border-radius:4px;width:%d%%"></div></div>') % (col, w['pct'])
        rows.append('<tr><td>%s</td><td>%d</td><td>-%.1f dB</td><td>%s&nbsp;%d%%</td><td>%.0fs ago</td></tr>'
                    % (w['pid'], w['ts'], w['db'], bar, w['pct'], w['age']))
    return ('<div class="card"><h2>workers &mdash; ring-down progress toward finishing (%d active)</h2>'
            '<table><tr><th>worker</th><th>timestep</th><th>energy decay</th>'
            '<th>progress (to -30 dB / %dk cap)</th><th>updated</th></tr>%s</table></div>'
            % (len(ws), NRTS // 1000, ''.join(rows)))


def convergence_png(rows):
    ev, s11, score = [], [], []
    for r in rows:
        try:
            ev.append(int(r['eval']))
            s11.append(float(r['worst_S11_dB']))
            score.append(float(r['score']))
        except Exception:
            continue
    if not ev:
        return None
    bsc, b_s11, rb_sc, rb_s11 = -1e18, float('nan'), [], []
    for sc, si in zip(score, s11):
        if sc > bsc and abs(sc) < 1e17:
            bsc, b_s11 = sc, si
        rb_sc.append(bsc if abs(bsc) < 1e17 else float('nan'))
        rb_s11.append(b_s11)
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    fin = [(e, v) for e, v in zip(ev, s11) if abs(v) < 1e17]
    if fin:
        ax[0].plot([e for e, _ in fin], [v for _, v in fin], '.', ms=4, alpha=0.35, label='per eval')
    ax[0].plot(ev, rb_s11, '-', lw=2, color='C0', label='best-so-far')
    ax[0].axhline(-10, color='g', ls='--', lw=1, label='-10 dB target')
    ax[0].axhline(-8, color='orange', ls=':', lw=1, label='-8 dB plateau')
    ax[0].set(title='worst-in-band |S11|', xlabel='eval', ylabel='dB')
    ax[0].legend(fontsize=7); ax[0].grid(True, alpha=0.3)
    fsc = [(e, v) for e, v in zip(ev, score) if abs(v) < 1e17]
    if fsc:
        ax[1].plot([e for e, _ in fsc], [v for _, v in fsc], '.', ms=4, alpha=0.35, label='score')
    ax[1].plot(ev, rb_sc, '-', lw=2, color='C2', label='best score')
    ax[1].set(title='score (gain - match penalty)', xlabel='eval', ylabel='dBi')
    ax[1].legend(fontsize=7); ax[1].grid(True, alpha=0.3)
    buf = io.BytesIO(); fig.tight_layout(); fig.savefig(buf, format='png', dpi=100); plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()


def build_page():
    rows = read_log()
    j = read_json()
    conv = convergence_png(rows)
    live = b64_file(os.path.join(DIR, 'live_s11.png'))
    drv = tail(os.path.join(DIR, 'driver.out'), 18)

    score = j.get('score')
    gain = j.get('worst_realized_gain_dBi')
    s11b = j.get('worst_in_band_S11_dB')
    nev = j.get('evals_so_far', len(rows))
    matched = (s11b is not None and s11b <= -10.0)
    badge_c = '#1a7f37' if matched else '#9a6700'
    badge_t = 'MATCHED (S11 &le; -10 dB)' if matched else 'not matched yet'

    def fnum(v, u=''):
        return ('%.2f%s' % (v, u)) if isinstance(v, (int, float)) else '&mdash;'

    params = j.get('params_mm', {})
    ptbl = ''.join('<td>%s<br><b>%.2f</b></td>' % (html.escape(k), v) for k, v in params.items())

    # recent evals table (last 25), highlight running-best rows
    head = ['eval', 'worst_gain_dBi', 'worst_S11_dB', 'score']
    bsc = -1e18
    trows = []
    for r in rows:
        try:
            sc = float(r['score'])
        except Exception:
            sc = float('nan')
        newbest = abs(sc) < 1e17 and sc > bsc
        if newbest:
            bsc = sc
        cells = ''.join('<td>%s</td>' % html.escape(r.get(h, '')) for h in head)
        trows.append('<tr%s>%s</tr>' % (' class="nb"' if newbest else '', cells))
    trows = trows[-25:]

    img = lambda b: ('<img src="data:image/png;base64,%s">' % b) if b else '<i>(no plot yet)</i>'
    return """<!doctype html><html><head><meta charset="utf-8">
<meta http-equiv="refresh" content="%d"><title>fpc3 %s</title>
<style>
body{font-family:system-ui,Arial;margin:18px;background:#0d1117;color:#c9d1d9}
.card{background:#161b22;border:1px solid #30363d;border-radius:8px;padding:14px 18px;margin:12px 0}
h1{font-size:20px;margin:0 0 4px} h2{font-size:15px;color:#8b949e;margin:0 0 10px}
.big{font-size:30px;font-weight:700} .row{display:flex;gap:26px;flex-wrap:wrap;align-items:center}
.badge{padding:4px 10px;border-radius:12px;color:#fff;font-weight:600;font-size:13px}
table{border-collapse:collapse;font-size:12px;width:100%%} td,th{border:1px solid #30363d;padding:3px 7px;text-align:center}
tr.nb{background:#12361f} img{max-width:100%%;border-radius:6px;background:#fff}
pre{background:#010409;padding:10px;border-radius:6px;overflow-x:auto;font-size:12px;color:#8b949e}
.lab{color:#8b949e;font-size:12px}
</style></head><body>
<h1>fpc3 DE search &mdash; %s</h1>
<h2>%d evals &middot; auto-refresh %ds &middot; instance up %s &middot; %s</h2>
<div class="card"><div class="row">
  <div><div class="lab">best score</div><div class="big">%s</div></div>
  <div><div class="lab">realized gain</div><div class="big">%s</div></div>
  <div><div class="lab">worst-in-band S11</div><div class="big">%s</div></div>
  <div><span class="badge" style="background:%s">%s</span></div>
</div></div>
%s
<div class="card"><h2>convergence (is S11 breaking -8 &rarr; -10 dB?)</h2>%s</div>
<div class="card"><h2>best design</h2><table><tr>%s</tr></table><br>%s</div>
<div class="card"><h2>recent evals (green = new best)</h2><table>
<tr>%s</tr>%s</table></div>
<div class="card"><h2>driver.out (tail)</h2><pre>%s</pre></div>
</body></html>""" % (
        REFRESH, TAG, TAG, nev, REFRESH, fmt_dur(instance_uptime()),
        time.strftime('%Y-%m-%d %H:%M:%S'),
        fnum(score), fnum(gain, ' dBi'), fnum(s11b, ' dB'), badge_c, badge_t,
        workers_card(),
        img(conv), ptbl or '<td>(none yet)</td>', img(live),
        ''.join('<th>%s</th>' % h for h in head), ''.join(trows),
        html.escape(drv) or '(no driver.out yet)')


class H(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path not in ('/', '/index.html'):
            self.send_response(404); self.end_headers(); return
        try:
            body = build_page().encode('utf-8')
        except Exception as e:
            body = ('<pre>dashboard error: %s</pre>' % html.escape(repr(e))).encode()
        self.send_response(200)
        self.send_header('Content-Type', 'text/html; charset=utf-8')
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *a):
        pass


if __name__ == '__main__':
    print('serving fpc3 dashboard for %s on 0.0.0.0:%d  (refresh %ds)' % (DIR, PORT, REFRESH))
    ThreadingHTTPServer(('0.0.0.0', PORT), H).serve_forever()
