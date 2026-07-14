% stackEl.m - single stacked (driven + parasitic) patch on 2x1.52mm Teflon.
% Goal: two merged resonances spanning 3.1-3.4 GHz (VSWR<2, ~9%).
% Probe feed to the driven (lower) patch; parasitic (upper) patch is coupled.

f0 = 3.25e9; band = linspace(3.0e9,3.5e9,21);
c = physconst("lightspeed");
er = 2.94; h = 1.52e-3;
d = dielectric('Name','PTFE','EpsilonR',er,'LossTangent',0.0016,'Thickness',h);

% initial guesses (driven resonates a touch high, parasitic a touch low)
Ld = 24.5e-3;   % driven patch length
Lp = 26.5e-3;   % parasitic patch length
W  = 33.0e-3;   % common width
feedOff = 6.5e-3;   % probe offset from centre on driven patch

drivenPatch = antenna.Rectangle(Length=W,Width=Ld,Center=[0 0]);
paraPatch   = antenna.Rectangle(Length=W,Width=Lp,Center=[0 0]);
gnd         = antenna.Rectangle(Length=2*W,Width=2*W,Center=[0 0]);

p = pcbStack;
p.BoardShape = antenna.Rectangle(Length=2*W,Width=2*W);
p.BoardThickness = 2*h;
p.Layers = {paraPatch, d, drivenPatch, d, gnd};
p.FeedLocations = [0 feedOff 3 5];   % probe: driven(3) -> ground(5)
p.FeedDiameter  = 1.0e-3;
p.ViaLocations  = [0 feedOff 3 5];
p.ViaDiameter   = 1.0e-3;

fprintf('stacked patch: Ld=%.1f Lp=%.1f W=%.1f feedOff=%.1fmm\n', Ld*1e3,Lp*1e3,W*1e3,feedOff*1e3);
Z = impedance(p, band);
fprintf('band response (ref 50 ohm):\n');
for k=1:numel(band)
    g=abs((Z(k)-50)/(Z(k)+50));
    fprintf('  %.3fGHz: Z=%7.2f%+7.2fj  VSWR=%5.2f\n', band(k)/1e9, real(Z(k)),imag(Z(k)),(1+g)/(1-g));
end
