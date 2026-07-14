% capProbe.m - capability check before committing to the native-element route.
%  1) which toolboxes are present (RF Toolbox needed for circuit combine)
%  2) can a conformalArray of native inset patches return multiport S-params
%  3) does an inset patch expose a usable feed reference

fprintf('=== installed toolboxes ===\n');
v = ver;
for k=1:numel(v)
    if contains(v(k).Name,'RF') || contains(v(k).Name,'Antenna')
        fprintf('  %s %s\n', v(k).Name, v(k).Version);
    end
end

freq = 3.25e9;
d = dielectric("FR4"); d.EpsilonR = 4.3; d.Thickness = 1.6e-3;
wStrip = traceThickness(100,d);

mkEl = @() patchMicrostripInsetfed(Length=20.9e-3, Width=20.9e-3, Height=d.Thickness, ...
    Substrate=d, GroundPlaneLength=0.06, GroundPlaneWidth=0.06, ...
    StripLineWidth=wStrip, NotchWidth=2*wStrip, NotchLength=2e-3);

fprintf('\n=== single inset patch impedance @ 3.25 (sanity) ===\n');
try
    Z1 = impedance(mkEl(), freq);
    fprintf('  Zin = %.2f %+.2fj ohm\n', real(Z1), imag(Z1));
catch e, fprintf('  single FAILED: %s\n', e.message); end

fprintf('\n=== four-element conformalArray S-params test ===\n');
try
    X=14e-3; Y=14e-3;
    e1=mkEl(); e2=mkEl(); e3=mkEl(); e4=mkEl();
    e3.Tilt=180; e3.TiltAxis=[0 0 1];      % bottom patches: strips point +y
    e4.Tilt=180; e4.TiltAxis=[0 0 1];
    ca = conformalArray;
    ca.Element = {e1,e2,e3,e4};
    ca.ElementPosition = [ X Y 0; -X Y 0; X -Y 0; -X -Y 0 ];
    S = sparameters(ca, freq);
    fprintf('  sparameters OK, size=%s, Z0=%g\n', mat2str(size(S.Parameters)), S.Impedance);
    disp(round(S.Parameters,3));
catch e
    fprintf('  conformalArray/sparameters FAILED: %s\n', e.message);
end
