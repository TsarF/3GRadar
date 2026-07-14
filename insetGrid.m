% insetGrid.m - isolated native inset patch: grid (L, NotchLength) to land
% ~100 + j0 ohm at 3.25 GHz. Element was 92+38j at (20.9mm, 2.0mm): inductive,
% so lengthen L to drop resonance; notch trims the real part.

freq = 3.25e9;
d = dielectric("FR4"); d.EpsilonR = 4.3; d.Thickness = 1.6e-3;
wStrip = traceThickness(100,d);

Lv = (21.0:0.4:22.6)*1e-3;
Nv = (1.2:0.5:2.7)*1e-3;
fprintf('isolated inset patch, |Z-100| grid (strip=%.2fmm):\n', wStrip*1e3);
best = inf; bL=Lv(1); bN=Nv(1);
fprintf('        ');
fprintf('N=%4.1f      ', Nv*1e3); fprintf('\n');
for i=1:numel(Lv)
    fprintf('L=%4.1f  ', Lv(i)*1e3);
    for j=1:numel(Nv)
        p = patchMicrostripInsetfed(Length=Lv(i), Width=Lv(i), Height=d.Thickness, ...
            Substrate=d, GroundPlaneLength=0.06, GroundPlaneWidth=0.06, ...
            StripLineWidth=wStrip, NotchWidth=2*wStrip, NotchLength=Nv(j));
        Z = impedance(p, freq);
        fprintf('%5.0f%+5.0fj ', real(Z), imag(Z));
        if abs(Z-100) < best, best=abs(Z-100); bL=Lv(i); bN=Nv(j); end
    end
    fprintf('\n');
end
fprintf('=> best ~100ohm at L=%.2fmm notch=%.2fmm (|Z-100|=%.1f)\n', bL*1e3, bN*1e3, best);
