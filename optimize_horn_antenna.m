%% optimize_horn_antenna.m
% Parallel optimization of a pyramidal (flared) horn antenna fed by a
% rectangular waveguide.
%
% Toolboxes required:
%   * Antenna Toolbox             - horn, pattern, impedance, sparameters, design, show
%   * Global Optimization Toolbox - ga (genetic algorithm)
%   * Parallel Computing Toolbox  - parpool / 'UseParallel'
%   * RF Toolbox                  - rfplot, rfparam (S-parameter handling / matching)
%
% Configuration here: S-band, ~3.25 GHz, fed by WR-284 waveguide.
%
% WHY THE VSWR WAS ~1e4 BEFORE, AND WHAT CHANGED:
%   1. The horn is fed by a monopole/delta-gap probe in the back of the
%      waveguide. Its position (FeedOffset) sets the impedance match, and
%      the default position is sized for the default WR-75 horn. Rescaling
%      the guide by hand without re-tuning the probe left it almost on the
%      back short -> near-total reflection. FeedOffset is now an optimized
%      variable (x(5)).
%   2. The old cost only weakly penalised mismatch, so the optimizer kept a
%      reflective feed to chase directivity. The objective is now REALIZED
%      GAIN, which folds |Gamma| in directly (see hornObjective.m).
%   3. The flare lower bounds were below the waveguide cross-section, so
%      part of the search space was auto-rejected. Bounds are fixed below.
%
% Keep this file and hornObjective.m in the SAME folder.

clear; clc; close all;

%% 1. Design specification ------------------------------------------------
freq   = 3.25e9;                     % operating frequency [Hz] (S-band)
Z0     = 50;                         % coax/system reference impedance [ohm]
c      = physconst('LightSpeed');
lambda = c/freq;

% Fixed feed-waveguide cross-section: WR-284 (standard S-band, 2.6-3.95 GHz)
wg.Width  = 72.136e-3;   % broad wall  a [m]
wg.Height = 34.036e-3;   % narrow wall b [m]

% TE10 cutoff and guide wavelength (used to set sane bounds for Length/feed)
fc      = c/(2*wg.Width);
lambda_g = lambda / sqrt(1 - (fc/freq)^2);
fprintf('Design freq %.2f GHz | lambda0 = %.1f mm | TE10 fc = %.2f GHz | lambda_g = %.1f mm\n', ...
        freq/1e9, lambda*1e3, fc/1e9, lambda_g*1e3);

%% 1b. Sanity-check baseline ---------------------------------------------
% design() returns a standard-gain horn auto-scaled to freq WITH a feed
% positioned for a good match (its own auto-scaled guide, not WR-284). If
% its |S11| is sensible (e.g. better than -10 dB), the toolbox and the
% 50-ohm reference are fine -> the old VSWR was purely an un-tuned feed.
antRef = design(horn, freq);
sRef   = sparameters(antRef, freq, Z0);
fprintf('Baseline design(horn): |S11| = %.1f dB  (expect a sane, matched value)\n', ...
        20*log10(abs(rfparam(sRef, 1, 1))));

%% 2. Optimization variables and bounds ----------------------------------
% x = [FlareWidth, FlareHeight, FlareLength, Length, FeedOffsetX]  (metres)
%        E-plane      H-plane      axial flare   guide len  probe position
%
% Flare lower bounds sit just above the waveguide so every candidate is a
% real horn. Length is held near a guide wavelength. FeedOffset spans about
% +/- a quarter guide wavelength (the objective rejects probes that fall
% outside the guide for short Length values).
lb = [1.05*wg.Width, 1.05*wg.Height, 0.030, 0.030, -0.060];
ub = [0.400,         0.400,          0.300, 0.150,  0.060];
nVars = numel(lb);

%% 3. Start a parallel pool ----------------------------------------------
pool = gcp('nocreate');
if isempty(pool)
    pool = parpool(16);
end
fprintf('Parallel pool active with %d workers.\n', pool.NumWorkers);

%% 4. Configure and run the genetic algorithm ----------------------------
% Each evaluation is one MoM solve; ga spreads a generation's evaluations
% across the workers via 'UseParallel'.
objFcn = @(x) hornObjective(x, freq, Z0, wg);

opts = optimoptions('ga', ...
    'UseParallel',       true, ...
    'PopulationSize',    30, ...
    'MaxGenerations',    25, ...
    'EliteCount',        4, ...
    'FunctionTolerance', 1e-3, ...
    'Display',           'iter', ...
    'PlotFcn',           {@gaplotbestf, @gaplotstopping});

tStart = tic;
[xopt, fval, exitflag, output] = ga(objFcn, nVars, ...
    [], [], [], [], lb, ub, [], opts);
fprintf('\nOptimization finished in %.1f s (exitflag = %d, %d generations).\n', ...
        toc(tStart), exitflag, output.generations);

% ---- Alternatives (also support 'UseParallel', usually fewer EM solves):
%   opts = optimoptions('surrogateopt','UseParallel',true,'MaxFunctionEvaluations',200);
%   [xopt,fval] = surrogateopt(objFcn, lb, ub, opts);

%% 5. Build and report the optimized antenna -----------------------------
antOpt             = horn;
antOpt.Width       = wg.Width;
antOpt.Height      = wg.Height;
antOpt.FlareWidth  = xopt(1);
antOpt.FlareHeight = xopt(2);
antOpt.FlareLength = xopt(3);
antOpt.Length      = xopt(4);
antOpt.FeedOffset  = [xopt(5) 0];

fprintf('\n--- Optimized horn dimensions ------------------------------\n');
fprintf('  FlareWidth  (E-plane aperture): %7.2f mm\n', xopt(1)*1e3);
fprintf('  FlareHeight (H-plane aperture): %7.2f mm\n', xopt(2)*1e3);
fprintf('  FlareLength (axial flare)     : %7.2f mm\n', xopt(3)*1e3);
fprintf('  Length      (feed waveguide)  : %7.2f mm\n', xopt(4)*1e3);
fprintf('  FeedOffset  (probe position)  : %7.2f mm\n', xopt(5)*1e3);

% Performance of the final design (finer angular grid for the report)
Dpeak = max(pattern(antOpt, freq, 0:5:355, -90:5:90), [], 'all');
Zin   = impedance(antOpt, freq);
gamma = (Zin - Z0)/(Zin + Z0);
magG  = min(abs(gamma), 0.999999);            % clamp for safe RL/VSWR
RL    = -20*log10(magG);
VSWR  = (1 + magG)/(1 - magG);
realizedGain = Dpeak + 10*log10(1 - magG^2);
fprintf('\n--- Optimized performance @ %.2f GHz -----------------------\n', freq/1e9);
fprintf('  Peak directivity : %6.2f dBi\n', Dpeak);
fprintf('  Realized gain    : %6.2f dBi\n', realizedGain);
fprintf('  Input impedance  : %6.1f %+6.1fj ohm\n', real(Zin), imag(Zin));
fprintf('  Return loss      : %6.2f dB\n', RL);
fprintf('  VSWR             : %6.2f\n', VSWR);
if VSWR > 5
    fprintf('  NOTE: VSWR still high. The MATLAB probe length is not a free\n');
    fprintf('        parameter, so a hand-set WR-284 guide may not reach 50 ohm.\n');
    fprintf('        Try widening the FeedOffset/Length bounds, or start from\n');
    fprintf('        design(horn,freq) and optimize the flare only.\n');
end

%% 6. Visualize -----------------------------------------------------------
figure; show(antOpt);          title('Optimized flared horn geometry');
figure; pattern(antOpt, freq); title('Optimized 3-D radiation pattern');

% S-parameters across the band -> RF Toolbox hand-off (rfplot / matching).
bw = linspace(freq*0.95, freq*1.05, 5);
S  = sparameters(antOpt, bw, Z0);
figure; rfplot(S);             title('S_{11} vs frequency');
