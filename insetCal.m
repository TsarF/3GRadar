function insetCal()
% insetCal - root-find (L, NotchLength) that makes the inset-fed square patch
% present 100 + j0 ohm at 3.25 GHz. The board-edge read equals the notch
% impedance only when matched (=100 ohm), so |Zin - 100| -> 0 pins both the
% resonant length and the inset depth at once.

freq = 3.25e9;
d = dielectric("FR4"); d.EpsilonR = 4.3; d.Thickness = 1.6e-3;
wStrip = traceThickness(100,d);
wNotch = 2*wStrip;

obj = @(p) costZ(p);
opts = optimset('Display','iter','MaxFunEvals',45,'TolX',1e-5,'TolFun',1e-1);
p0 = [20.7e-3, 2.0e-3];          % seed near the observed 100-ohm region
[pB,fB] = fminsearch(obj, p0, opts);
L = pB(1); nl = max(pB(2),0);
Z = impedance(mkP(L,nl), freq);
fprintf('\n==== INSET 100-OHM POINT ====\n');
fprintf('L = %.3f mm | NotchLength = %.3f mm | NotchWidth = %.3f mm | strip = %.3f mm\n', ...
    L*1e3, nl*1e3, wNotch*1e3, wStrip*1e3);
fprintf('Zin = %.2f %+.2fj ohm | |Zin-100| = %.2f\n', real(Z), imag(Z), fB);

    function p = mkP(L,nl)
        p = patchMicrostripInsetfed(Length=L, Width=L, Height=d.Thickness, ...
            Substrate=d, GroundPlaneLength=0.06, GroundPlaneWidth=0.06, ...
            StripLineWidth=wStrip, NotchWidth=wNotch, NotchLength=max(nl,0));
    end
    function c = costZ(p)
        L = min(max(p(1),19e-3),23e-3); nl = min(max(p(2),0),8e-3);
        Z = impedance(mkP(L,nl), freq);
        c = abs(Z - 100);
        fprintf('   L=%6.3fmm nl=%5.3fmm -> %7.2f%+7.2fj  cost=%6.2f\n', ...
            L*1e3, nl*1e3, real(Z), imag(Z), c);
    end
end
