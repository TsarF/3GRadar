% substrateBW.m - bandwidth vs substrate thickness on JLCPCB Teflon
% (Dk=2.94, Df=0.0016). Square probe-fed patch, re-resonated and feed-matched
% near 50 ohm at each thickness, swept to read the VSWR<2 fractional bandwidth.
% Answers: how thick must the Teflon stack be to span 3.1-3.4 GHz (9%)?

f0 = 3.25e9; c = physconst("lightspeed");
band = linspace(2.9e9, 3.6e9, 29);
er = 2.94;

ths = [1.52 3.04 4.56]*1e-3;   % 1, 2, 3 laminated layers
fprintf('Teflon Dk=%.2f Df=0.0016 | VSWR<2 bandwidth vs thickness:\n', er);

for h = ths
    d = dielectric('Name','PTFE','EpsilonR',er,'LossTangent',0.0016,'Thickness',h);
    % square patch sized for f0
    W = c/(2*f0)*sqrt(2/(er+1));
    eeff = (er+1)/2 + (er-1)/2/sqrt(1+12*h/W);
    dL = 0.412*h*(eeff+0.3)*(W/h+0.264)/((eeff-0.258)*(W/h+0.8));
    L = c/(2*f0*sqrt(eeff)) - 2*dL;

    % pick probe offset (from centre) giving ~50 ohm real at f0
    offs = (0.20:0.04:0.40)*L; bestc=inf; bo=offs(1);
    for o = offs
        p = patchMicrostrip(Length=L,Width=L,Height=h,Substrate=d, ...
            GroundPlaneLength=2*L,GroundPlaneWidth=2*L,FeedOffset=[o 0]);
        Z = impedance(p,f0); cst = abs(real(Z)-50)+abs(imag(Z));
        if cst<bestc, bestc=cst; bo=o; end
    end
    p = patchMicrostrip(Length=L,Width=L,Height=h,Substrate=d, ...
        GroundPlaneLength=2*L,GroundPlaneWidth=2*L,FeedOffset=[bo 0]);
    Zb = impedance(p,band);
    g = abs((Zb-50)./(Zb+50)); vs = (1+g)./(1-g);
    in = band(vs<2);
    if isempty(in), bw=0; fl=NaN; fh=NaN; else, fl=min(in); fh=max(in); bw=(fh-fl)/f0*100; end
    Z0=impedance(p,f0);
    fprintf('  h=%.2fmm L=%.1fmm W=%.1fmm off=%.1fmm | Z@f0=%5.1f%+5.1fj | VSWR<2: %.3f-%.3fGHz = %.1f%%\n', ...
        h*1e3, L*1e3, W*1e3, bo*1e3, real(Z0),imag(Z0), fl/1e9, fh/1e9, bw);
end
fprintf('(target span 3.1-3.4 GHz = 9.2%%)\n');
