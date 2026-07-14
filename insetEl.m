% insetEl.m - single native inset patch at the chosen operating point,
% L = 21.0 mm, notch = 2.5 mm, swept over the band. Confirms the element is
% sensible (real part near 100, modest reactance) before the 4-port solve.

freq = 3.25e9; freqRange = linspace(3.1e9,3.4e9,7);
d = dielectric("FR4"); d.EpsilonR = 4.3; d.Thickness = 1.6e-3;
wStrip = traceThickness(100,d);
L = 21.0e-3; notch = 2.5e-3;

p = patchMicrostripInsetfed(Length=L, Width=L, Height=d.Thickness, Substrate=d, ...
    GroundPlaneLength=0.06, GroundPlaneWidth=0.06, ...
    StripLineWidth=wStrip, NotchWidth=2*wStrip, NotchLength=notch);

fprintf('Single inset patch  L=%.1fmm notch=%.1fmm strip=%.2fmm:\n', L*1e3, notch*1e3, wStrip*1e3);
Z = impedance(p, freqRange);
for k=1:numel(freqRange)
    g = abs((Z(k)-100)/(Z(k)+100));   % vs 100 ohm reference
    fprintf('  %.3f GHz: Zin = %7.2f %+7.2fj  (|G vs100|=%.3f)\n', ...
        freqRange(k)/1e9, real(Z(k)), imag(Z(k)), g);
end
[~,i0]=min(abs(freqRange-freq));
fprintf('=> @3.25: %.2f%+.2fj ohm\n', real(Z(i0)), imag(Z(i0)));
