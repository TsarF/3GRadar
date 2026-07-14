function insetProbe()
% insetProbe - diagnose the inset array directly.
%  (A) one inset patch + its 100ohm strip, fed at the strip end -> true Zpatch
%  (B) all 4 patches + strips + 50ohm lines, fed at the centre node -> Zcentre
% Tells us what the patch really presents and what the combiner produces,
% instead of inferring it through the long rotating feed line.

freq = 3.25e9;
d = dielectric("FR4"); d.EpsilonR = 4.3; d.Thickness = 1.6e-3;
c = physconst("lightspeed");
W0=c/(2*freq*sqrt((d.EpsilonR+1)/2));
ee0=(d.EpsilonR+1)/2+(d.EpsilonR-1)/2/sqrt(1+12*d.Thickness/W0);
lambdaEff=c/(freq*sqrt(ee0)); spacing=lambdaEff*0.6;
GPL=0.12; GPW=0.12; X=spacing/2; Y=spacing/2;
wStrip=traceThickness(100,d); w50=traceThickness(50,d); wNotch=2*wStrip;
NOTCH=2.0e-3;

for L = [20.9e-3 21.8e-3 22.5e-3]
    innerEdge=Y-L/2; nb=innerEdge+NOTCH; sLen=nb;
    mkPatch=@(cx,cy) antenna.Rectangle(Length=L,Width=L,Center=[cx cy]);
    mkSlot=@(cx,sgn) antenna.Rectangle(Length=wNotch,Width=NOTCH,Center=[cx, sgn*(Y-L/2+NOTCH/2)]);
    mkSt=@(cx,sgn) antenna.Rectangle(Length=wStrip,Width=sLen,Center=[cx, sgn*sLen/2]);

    % (A) single patch + strip extended past y=0 to a feed tab; feed mid-strip
    ftA = (mkPatch(X,Y)-mkSlot(X,1)) ...
        + antenna.Rectangle(Length=wStrip,Width=nb+4e-3,Center=[X (nb-4e-3)/2]);
    Za = NaN;
    try, Za = probe(ftA,[X -2e-3]); catch e, fprintf('  (A fail: %s)\n',e.message); end

    % (B) full combiner, fed at centre node (0,0)
    patches=(mkPatch(X,Y)-mkSlot(X,1))+(mkPatch(-X,Y)-mkSlot(-X,1)) ...
           +(mkPatch(X,-Y)-mkSlot(X,-1))+(mkPatch(-X,-Y)-mkSlot(-X,-1));
    strips=mkSt(X,1)+mkSt(-X,1)+mkSt(X,-1)+mkSt(-X,-1);
    lineR=antenna.Rectangle(Length=X+wStrip,Width=w50,Center=[X/2 0]);
    lineL=antenna.Rectangle(Length=X+wStrip,Width=w50,Center=[-X/2 0]);
    ftB = patches+strips+lineR+lineL;
    Zb = NaN;
    try, Zb = probe(ftB,[0 0]); catch e, fprintf('  (B fail: %s)\n',e.message); end

    fprintf('L=%.2fmm | Zpatch(@strip)=%7.2f%+7.2fj | Zcentre=%7.2f%+7.2fj\n', ...
        L*1e3, real(Za),imag(Za), real(Zb),imag(Zb));
end

    function Z = probe(ft, loc)
        pcb=pcbStack;
        pcb.BoardShape=antenna.Rectangle(Length=GPL,Width=GPW);
        pcb.BoardThickness=d.Thickness;
        pcb.Layers={ft, d, antenna.Rectangle(Length=GPL,Width=GPW)};
        pcb.FeedLocations=[loc 1 3]; pcb.FeedDiameter=w50/2;
        pcb.ViaLocations=[loc 1 3]; pcb.ViaDiameter=w50/2;
        Z=impedance(pcb,freq);
    end
end
