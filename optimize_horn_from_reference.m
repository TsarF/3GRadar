 %% optimize_horn_from_reference.m
% Start from the matched reference horn (design(horn,freq), which gives a
% good S11, e.g. -24 dB) and maximize directivity WITHOUT losing the match.
%
% The flare is the waveguide-to-free-space impedance transition, so changing
% it detunes the match design() set up. We therefore maximize directivity
% subject to a return-loss penalty (soft constraint), seed the GA with the
% reference flare (a known-matched, feasible point), and edit a COPY of the
% reference antenna so every tuned property is preserved.
%
% Toolboxes: Antenna Toolbox, Global Optimization Toolbox,
%            Parallel Computing Toolbox, RF Toolbox.
% Keep this file and hornObjectiveFlare.m in the SAME folder.

clear; clc; close a ll;

%% 1. Reference (matched) horn -------------------------------------------
freq = 3.25e9;                       % operating frequency [Hz]
Z0   = 50;                           % reference impedance [ohm]

antRef = design(horn, freq);         % standard-gain horn, matched at freq

%% 1b. Self-check: confirm the seed really is matched --------------------
% Evaluate the reference flare THROUGH the objective. If this prints a good
% return loss (e.g. ~24 dB) and a sensible cost, the seed is feasible and
% the optimizer cannot end up worse than this.
xRef = [antRef.FlareWidth, antRef.FlareHeight, antRef.FlareLength];
[cRef, Dref, RLref] = hornObjectiveFlare(xRef, freq, Z0, antRef);
fprintf('--- Reference design(horn,%.2fGHz) -------------------------\n', freq/1e9);
fprintf('  Flare (W x H x L): %.1f x %.1f x %.1f mm\n', xRef*1e3);
fprintf('  Directivity      : %6.2f dBi\n', Dref);
fprintf('  Return loss      : %6.2f dB   (S11 = %.1f dB)\n', RLref, -RLref);
fprintf('  Objective (cost) : %6.2f\n', cRef);
if RLref < 10
    warning(['Reference return loss is low (%.1f dB). If you expected ~24 dB, ', ...
             'design() may not be matched at this freq on your install.'], RLref);
end

%% 2. Flare-only variables and bounds ------------------------------------
% x = [FlareWidth, FlareHeight, FlareLength]   (metres)
lb = [1.05*antRef.Width, 1.05*antRef.Height, 0.030];
ub = [0.400,            0.400,              0.300];
nVars = numel(lb);

%% 3. Parallel pool -------------------------------------------------------
pool = gcp('nocreate');
if isempty(pool), pool = parpool(16); end
fprintf('Parallel pool active with %d workers.\n', pool.NumWorkers);

%% 4. Run the genetic algorithm ------------------------------------------
objFcn = @(x) hornObjectiveFlare(x, freq, Z0, antRef);   % ga uses 1st output

opts = optimoptions('ga', ...
    'UseParallel',             true, ...
    'PopulationSize',          30, ...
    'MaxGenerations',          25, ...
    'EliteCount',              4, ...
    'FunctionTolerance',       1e-3, ...
    'InitialPopulationMatrix', xRef, ...    % seed the matched reference
    'Display',                 'iter', ...
    'PlotFcn',                 {@gaplotbestf, @gaplotstopping});

tStart = tic;
[xopt, fval, exitflag, output] = ga(objFcn, nVars, ...
    [], [], [], [], lb, ub, [], opts);
fprintf('\nOptimization finished in %.1f s (exitflag = %d, %d generations).\n', ...
        toc(tStart), exitflag, output.generations);

%% 5. Build and report the optimized antenna -----------------------------
antOpt             = copy(antRef);
antOpt.FlareWidth  = xopt(1);
antOpt.FlareHeight = xopt(2);
antOpt.FlareLength = xopt(3);

[~, Dopt, RLopt] = hornObjectiveFlare(xopt, freq, Z0, antRef);
magG = 10^(-RLopt/20);
VSWR = (1 + magG)/(1 - magG);

fprintf('\n--- Optimized horn (directivity maximized, match kept) -----\n');
fprintf('  FlareWidth       : %7.2f mm   (ref %.1f)\n', xopt(1)*1e3, xRef(1)*1e3);
fprintf('  FlareHeight      : %7.2f mm   (ref %.1f)\n', xopt(2)*1e3, xRef(2)*1e3);
fprintf('  FlareLength      : %7.2f mm   (ref %.1f)\n', xopt(3)*1e3, xRef(3)*1e3);
fprintf('  Peak directivity : %6.2f dBi   (ref %.2f)\n', Dopt, Dref);
fprintf('  Return loss      : %6.2f dB    (S11 = %.1f dB, VSWR %.2f)\n', ...
        RLopt, -RLopt, VSWR);

%% 6. Visualize -----------------------------------------------------------
figure; show(antOpt);          title('Optimized flared horn geometry');
figure; pattern(antOpt, freq); title('Optimized 3-D radiation pattern');

bw = linspace(freq*0.9, freq*1.1, 11);
S  = sparameters(antOpt, bw, Z0);
figure; rfplot(S);             title('S_{11} vs frequency');
