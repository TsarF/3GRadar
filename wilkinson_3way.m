%% wilkinson_3way.m
% 3-way equal-split Wilkinson power divider in microstrip on FR-4
% Band: 3.1 - 3.4 GHz  (f0 = 3.25 GHz)
% Stack: er = 4.4, tan(d) = 0.02, h = 0.21 mm above ground, 1/2 oz copper
%
% Topology (Wilkinson 1960, N-way): three quarter-wave branches of
% Zq = sqrt(3)*50 = 86.6 ohm from the input node to each output port,
% plus a 50-ohm resistor from each output to a common FLOATING star
% node (equivalent to 150 ohm between any two outputs as a delta).
% At f0 this gives ideal match at all 4 ports and ideal isolation.
%
% Requires: RF Toolbox R2021a or newer (txlineMicrostrip as a circuit
% element). The circuit-level model includes conductor + dielectric
% loss and dispersion of the lines, but NOT junction discontinuities
% or resistor parasitics - EM-verify the final layout before fab.

clear; clc; close all;

%% ---------------- 1. Specifications ----------------
Z0   = 50;                  % system impedance [ohm]
N    = 3;                   % number of output ports
fL   = 3.1e9;  fH = 3.4e9;  % band edges [Hz]
f0   = (fL + fH)/2;         % design frequency = 3.25 GHz

% FR-4 stack-up -- edit to match your laminate datasheet
er    = 2.94;                % relative permittivity near 3 GHz
tand  = 0.0016;               % loss tangent
h     = 0.21e-3;            % dielectric height above ground [m]
t     = 17.5e-6;            % copper thickness [m] (1/2 oz foil)
sigma = 5.88e7;             % copper conductivity [S/m]

% txlineMicrostrip's quasi-static model requires t/h <= 0.1, so with
% h = 0.21 mm the modeled copper is capped at 21 um. Copper is many
% skin depths thick at 3.25 GHz (skin depth ~1.2 um), so clamping t
% leaves the simulated loss essentially unchanged; it only slightly
% affects the thickness correction to the line width.
if t > 0.1*h
    warning('t = %.1f um exceeds the t/h <= 0.1 model limit; clamping to %.1f um.', ...
            t*1e6, 0.1*h*1e6);
    t = 0.1*h;
end

freq = linspace(2.0e9, 4.5e9, 451);    % analysis grid

%% ---------------- 2. Electrical design values ----------------
Zq   = Z0*sqrt(N);          % 86.60 ohm quarter-wave branch lines
Riso = Z0;                  % 50 ohm from each output to floating star node

%% ---------------- 3. Microstrip synthesis (toolbox model) ----------------
% Helper: microstrip line on this stack-up
mk = @(w,L) txlineMicrostrip('Width',w, 'Height',h, 'EpsilonR',er, ...
        'LossTangent',tand, 'Thickness',t, 'SigmaCond',sigma, ...
        'LineLength',L);

% Width for the 86.6-ohm branches: solve with the toolbox's own model
wq  = fzero(@(w) getZ0(mk(w, 1e-3)) - Zq, [0.05e-3 1.5e-3]);

% 50-ohm reference width (for the feed lines in your layout)
w50 = fzero(@(w) getZ0(mk(w, 1e-3)) - Z0, [0.05e-3 1.5e-3]);

% First-cut quarter-wave length from quasi-static eps_eff (Hammerstad)
u    = wq/h;
eeff = (er+1)/2 + (er-1)/2 / sqrt(1 + 12/u);
c0   = 299792458;
L0   = c0/(4*f0*sqrt(eeff));

% Refine so the branch is exactly 90 deg at f0 per the toolbox model.
% Referencing the S-parameters to Zq makes angle(S21) = -beta*L exactly.
phs = @(L) angle(rfparam(sparameters(mk(wq, L), f0, Zq), 2, 1));
Lq  = fzero(@(L) phs(L) + pi/2, [0.6 1.4]*L0);

fprintf('--- 3-way Wilkinson @ %.3f GHz on FR-4 (er = %.2f, h = %.2f mm) ---\n', ...
        f0/1e9, er, h*1e3);
fprintf('Branch lines : Zq = %.2f ohm -> W = %.4f mm, L(90 deg) = %.3f mm\n', ...
        Zq, wq*1e3, Lq*1e3);
fprintf('50-ohm feed  : W = %.4f mm\n', w50*1e3);
fprintf('Isolation R  : %d ohm from each output to one floating star node\n', Riso);
fprintf('               (equivalent to %d ohm between any output pair)\n\n', 3*Riso);

%% ---------------- 4. Circuit simulation (4-port) ----------------
S  = simWilk3(wq, Lq, er, tand, h, t, sigma, Riso, freq);

dB  = @(x) 20*log10(abs(x));
fG  = freq/1e9;
S11 = dB(rfparam(S,1,1));
S21 = dB(rfparam(S,2,1));    % = S31 = S41 by symmetry
S22 = dB(rfparam(S,2,2));    % = S33 = S44
S23 = dB(rfparam(S,2,3));    % = every output-output pair

% rfwrite(S, 'wilkinson_3way.s4p');   % uncomment to export Touchstone

%% ---------------- 5. Plots ----------------
figure('Name','Split and input match');
plot(fG, S21, 'LineWidth', 1.5); hold on; grid on;
plot(fG, S11, 'LineWidth', 1.5);
yline(-10*log10(N), ':', 'ideal -4.77 dB');
xline(fL/1e9, '--'); xline(fH/1e9, '--');
xlabel('Frequency (GHz)'); ylabel('Magnitude (dB)');
legend('S_{21} = S_{31} = S_{41}', 'S_{11}', 'Location', 'southeast');
title('3-way Wilkinson on FR-4: transmission and input match');
ylim([-50 0]);

figure('Name','Output match and isolation');
plot(fG, S22, 'LineWidth', 1.5); hold on; grid on;
plot(fG, S23, 'LineWidth', 1.5);
xline(fL/1e9, '--'); xline(fH/1e9, '--');
xlabel('Frequency (GHz)'); ylabel('Magnitude (dB)');
legend('S_{22} = S_{33} = S_{44}', 'Isolation S_{23} (all pairs)', ...
       'Location', 'southeast');
title('Output return loss and port-to-port isolation');
ylim([-60 0]);

%% ---------------- 6. In-band summary ----------------
inb = (freq >= fL) & (freq <= fH);
fprintf('In-band performance, %.1f - %.1f GHz (worst case):\n', fL/1e9, fH/1e9);
fprintf('  Input return loss   : %6.1f dB\n', -max(S11(inb)));
fprintf('  Transmission        : %6.2f to %.2f dB (ideal -4.77 dB)\n', ...
        min(S21(inb)), max(S21(inb)));
fprintf('  Excess (FR-4) loss  : %6.2f dB\n', -10*log10(N) - max(S21(inb)));
fprintf('  Output return loss  : %6.1f dB\n', -max(S22(inb)));
fprintf('  Output isolation    : %6.1f dB\n\n', -max(S23(inb)));

%% ---------------- 7. FR-4 permittivity sensitivity ----------------
% FR-4 er is poorly controlled at S-band. Geometry stays fixed (as
% fabricated); er of the laminate is swept to see the band shift.
ers = [4.0 4.2 4.4 4.6 4.8];
co  = lines(numel(ers));
figure('Name','FR-4 er sensitivity'); hold on; grid on;
for k = 1:numel(ers)
    Sk = simWilk3(wq, Lq, ers(k), tand, h, t, sigma, Riso, freq);
    plot(fG, dB(rfparam(Sk,1,1)), 'Color', co(k,:), 'LineWidth', 1.2, ...
         'DisplayName', sprintf('\\epsilon_r = %.1f', ers(k)));
end
xline(fL/1e9, '--', 'HandleVisibility', 'off');
xline(fH/1e9, '--', 'HandleVisibility', 'off');
xlabel('Frequency (GHz)'); ylabel('|S_{11}| (dB)');
title('Input match vs FR-4 permittivity (fixed geometry)');
legend('Location', 'southeast'); ylim([-50 0]);

%% ---------------- Local function ----------------
function S = simWilk3(w, L, er, tand, h, t, sigma, Riso, freq)
% Builds the 4-port 3-way Wilkinson:
%   node 1 = input, nodes 2/3/4 = outputs, node 5 = floating star node
    ckt = circuit('wilk3');
    for k = 1:3
        TL = txlineMicrostrip('Width',w, 'Height',h, 'EpsilonR',er, ...
                'LossTangent',tand, 'Thickness',t, 'SigmaCond',sigma, ...
                'LineLength',L, 'Name', ['TL' num2str(k)]);
        add(ckt, [1 k+1], TL);                                % input -> out k
        add(ckt, [k+1 5], resistor(Riso, ['R' num2str(k)]));  % out k -> star
    end
    setports(ckt, [1 0], [2 0], [3 0], [4 0]);
    S = sparameters(ckt, freq);
end