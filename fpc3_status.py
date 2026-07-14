"""One-shot status of the fpc3 antenna optimization. Run: python fpc3_status.py"""
import os
import json

D = 'fpc3_gain_de_opt'


def tail(path, n=6):
    try:
        with open(path, errors='ignore') as f:
            return ''.join(f.readlines()[-n:]).rstrip()
    except FileNotFoundError:
        return '(none)'


print('==================== fpc3 optimization status ====================')
try:
    with open(os.path.join(D, 'optimized_params.json')) as f:
        b = json.load(f)
    print('BEST worst-in-band realized gain: %.2f dBi' % b['worst_realized_gain_dBi'])
    print('  params : %s' % b['params_mm'])
    print('  fixed  : %s' % b.get('fixed_mm'))
    gb = b.get('gain_across_band')
    if gb:
        print('  across band (3.10..3.40): %s dBi' % [round(v, 2) for v in gb])
    print('  evals at last best: %s' % b.get('evals_so_far'))
except FileNotFoundError:
    print('BEST: no optimized_params.json yet (initial population still evaluating)')

try:
    with open(os.path.join(D, 'optimization_log.csv')) as f:
        rows = f.readlines()
    print('total evals logged: %d' % (len(rows) - 1))
except FileNotFoundError:
    print('total evals logged: 0')

for f in ('driver.pid', 'watchdog.pid'):
    p = os.path.join(D, f)
    if os.path.exists(p):
        print('%s: %s' % (f, open(p).read().strip()))
print('\n--- recent driver.out ---')
print(tail(os.path.join(D, 'driver.out'), 6))
print('\n--- watchdog.log tail ---')
print(tail(os.path.join(D, 'watchdog.log'), 4))
print('==================================================================')
