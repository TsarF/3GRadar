# Self-healing watchdog for the fpc3 optimizer. Runs detached; survives VSCode
# closing. Restarts the optimizer if it dies (it warm-starts from optimized_params.json,
# so no progress is lost) and stops once the run reports convergence/completion.
$repo    = "c:\Users\Nikita\Documents\Repos\3GRadar"
$dir     = Join-Path $repo "fpc3_gain_de_opt"
$out     = Join-Path $dir "driver.out"
$errf    = Join-Path $dir "driver.err"
$pidfile = Join-Path $dir "driver.pid"
$hb      = Join-Path $dir "watchdog.hb"
$wlog    = Join-Path $dir "watchdog.log"

"watchdog started $(Get-Date -Format o)" | Add-Content $wlog
while ($true) {
    "watchdog alive $(Get-Date -Format o)" | Set-Content $hb

    # finished? (optimizer prints one of these at the end of a clean run)
    if (Test-Path $out) {
        $tail = (Get-Content $out -Tail 8 -ErrorAction SilentlyContinue) -join "`n"
        if ($tail -match "BEST 3-layer DESIGN|Converged \(population|generation cap") {
            "DONE $(Get-Date -Format o)" | Set-Content (Join-Path $dir "WATCHDOG_DONE")
            "watchdog stopping (optimizer finished) $(Get-Date -Format o)" | Add-Content $wlog
            break
        }
    }

    # optimizer alive?
    $alive = $false
    if (Test-Path $pidfile) {
        $procid = (Get-Content $pidfile -ErrorAction SilentlyContinue | Select-Object -First 1)
        if ($procid) {
            try { Get-Process -Id ([int]$procid) -ErrorAction Stop | Out-Null; $alive = $true } catch {}
        }
    }
    if (-not $alive) {
        $p = Start-Process -FilePath "python" -ArgumentList "fpc3_optimize_de_gain.py" `
             -WorkingDirectory $repo -RedirectStandardOutput $out -RedirectStandardError $errf `
             -WindowStyle Hidden -PassThru
        $p.Id | Set-Content $pidfile
        "relaunched optimizer $(Get-Date -Format o) pid $($p.Id)" | Add-Content $wlog
    }

    Start-Sleep -Seconds 300
}
