%% Prime-focus parabolic reflector  -  fc = 3.25 GHz  (band 3.1 - 3.4 GHz)
%  Requires: MATLAB + Antenna Toolbox
%  API verified against MathWorks docs (reflectorParabolic, design,
%  hornConical, memoryEstimate, beamwidth, sparameters).
clear; clc; close all;

%% --- 1. Specification --------------------------------------------------
fc   = 3.25e9;                 % centre frequency  [Hz]
fLo  = 3.10e9;  fHi = 3.40e9;  % band edges        [Hz]
c    = physconst('lightspeed');
lam  = c/fc;                   % wavelength ~ 0.0923 m

%% --- 2. Geometry (edit these two lines to change the design) -----------
D      = 1.0;        % dish DIAMETER [m]  (= 10.8*lambda; must be >= 10*lambda)
FtoD   = 0.40;       % focal-length / diameter ratio (prime focus: 0.3 - 0.6)
F      = FtoD*D;     % focal length [m]

% First-order hand calculations (printed for reference) -----------------
DLam       = D/lam;
etaAp      = 0.60;                          % assumed aperture efficiency
G_dBi      = 10*log10(etaAp*(pi*DLam)^2);   % gain estimate
HPBW_deg   = 70*lam/D;                       % 3 dB beamwidth estimate
rimHalfAng = 2*atand(1/(4*FtoD));            % rim half-angle seen by the feed
fprintf('\n--- Hand estimates ---------------------------------\n');
fprintf(' D/lambda           : %.1f\n', DLam);
fprintf(' Gain (eta = %.2f)   : %.1f dBi\n', etaAp, G_dBi);
fprintf(' HPBW               : %.1f deg\n', HPBW_deg);
fprintf(' Feed rim half-angle: %.1f deg  (feed should light this to ~ -11 dB)\n', rimHalfAng);

%% --- 3. Build the reflector --------------------------------------------
p = design(reflectorParabolic, fc);   % default dipole-fed dish sized at fc
p.Radius      = D/2;
p.FocalLength = F;

%% --- 4. Choose a feed (exciter) ----------------------------------------
% --- Option A (recommended start): conical horn -------------------------
feed = design(hornConical, fc);
feed.Tilt = 90;                 % aim the horn aperture back at the dish
p.Exciter = feed;

% --- Option B: keep the simple half-wave dipole feed --------------------
%   (comment out Option A; design() already supplied a dipole exciter)

% --- Option C: pyramidal horn (fed from WR-284 S-band waveguide) --------
%   feed = design(horn, fc); feed.Tilt = 90; p.Exciter = feed;

% --- Option D: circular polarization -> axial-mode helix ----------------
%   feed = design(helix, fc); feed.Tilt = 90; p.Exciter = feed;

figure; show(p); title('Parabolic reflector @ 3.25 GHz'); view(40,20);

%% --- 5. Memory check (this is an electrically large structure!) --------
fprintf('\n--- Solver memory estimate -------------------------\n');
memoryEstimate(p, fc)

%% --- 6. Pattern & gain at the centre frequency -------------------------
figure; pattern(p, fc); title('3-D directivity @ 3.25 GHz');

[pat,~,~] = pattern(p, fc);             % full az/el grid
Gpeak = max(pat(:));                     % robust peak (boresight) value
fprintf('\nPeak directivity @ %.2f GHz : %.1f dBi\n', fc/1e9, Gpeak);

% Principal-plane cut + half-power beamwidth
figure; patternElevation(p, fc, 0); title('Elevation cut (az = 0)');
bw = beamwidth(p, fc, 0, 0:0.2:180);
fprintf('Elevation-plane HPBW        : %.2f deg\n', bw);

%% --- 7. Input match (S11) across the operating band --------------------
%  NOTE: this loop solves the full structure at each frequency and is the
%  slow part. Reduce the point count while iterating on the design.
freq = linspace(fLo, fHi, 31);
s = sparameters(p, freq);
figure; rfplot(s); grid on; title('S_{11}, 3.1 - 3.4 GHz');