function insetTune()
% insetTune - in-array tuning of the inset 2x2. Stage 1: sweep patch L to
% resonate the array (port imag -> 0) at fixed notch & final xfmr. Stage 2:
% from the resonant real part, resize the final lambdaG/4 to land 50 ohm.

freq = 3.25e9;
d = dielectric("FR4"); d.EpsilonR = 4.3; d.Thickness = 1.6e-3;
c = physconst("lightspeed");
W0 = c/(2*freq*sqrt((d.EpsilonR+1)/2));
ee0 = (d.EpsilonR+1)/2 + (d.EpsilonR-1)/2/sqrt(1+12*d.Thickness/W0);
lambdaEff = c/(freq*sqrt(ee0));
spacing = lambdaEff*0.6;
GPL=0.12; GPW=0.12; X=spacing/2; Y=spacing/2; portY=-GPW/2+3e-3;
wStrip=traceThickness(100,d); w50=traceThickness(50,d); wNotch=2*wStrip;

% ---- Stage 1: resonance vs L ----
Lv = (20.0:0.3:21.8)*1e-3;
fprintf('Stage 1: L sweep (notch=2.0mm, Z_FINAL=35.4):\n');
bImag=inf; Lres=Lv(1); Rres=NaN;
for k=1:numel(Lv)
    Z = portZ(Lv(k), 2.0e-3, 35.4);
    fprintf('  L=%5.2fmm  Zin=%7.2f%+7.2fj  VSWR=%.2f\n', Lv(k)*1e3, real(Z),imag(Z), vswrOf(Z));
    if abs(imag(Z))<bImag, bImag=abs(imag(Z)); Lres=Lv(k); Rres=real(Z); end
end
fprintf('=> resonant L=%.2fmm, Rport=%.2f ohm (with 35.4 xfmr)\n', Lres*1e3, Rres);

% center node behind the 35.4 xfmr (approx, treats feed as matched once tuned)
Zcenter = 35.4^2 / Rres;
ZfNew = sqrt(real(Zcenter)*50);
fprintf('   implied centre node ~%.1f ohm -> new Z_FINAL=%.1f ohm\n', Zcenter, ZfNew);

% ---- Stage 2: apply new final xfmr, small L touch-up ----
fprintf('\nStage 2: retune Z_FINAL=%.1f around L=%.2fmm:\n', ZfNew, Lres*1e3);
for L = Lres + (-0.3:0.15:0.3)*1e-3
    Z = portZ(L, 2.0e-3, ZfNew);
    fprintf('  L=%5.2fmm  Zin=%7.2f%+7.2fj  RL=%6.2fdB VSWR=%.2f\n', ...
        L*1e3, real(Z),imag(Z), 20*log10(abs((Z-50)/(Z+50))), vswrOf(Z));
end

    function v = vswrOf(Z), g=abs((Z-50)/(Z+50)); v=(1+g)/(1-g); end

    function Zin = portZ(L, NOTCH_LEN, Z_FINAL)
        wF=traceThickness(Z_FINAL,d); [LqF,~,~]=quarterWave(wF,d,freq);
        innerEdge=Y-L/2; notchBottom=innerEdge+NOTCH_LEN; sLen=notchBottom;
        mkPatch=@(cx,cy) antenna.Rectangle(Length=L,Width=L,Center=[cx cy]);
        mkSlot=@(cx,sgn) antenna.Rectangle(Length=wNotch,Width=NOTCH_LEN,Center=[cx, sgn*(Y-L/2+NOTCH_LEN/2)]);
        mkSt=@(cx,sgn) antenna.Rectangle(Length=wStrip,Width=sLen,Center=[cx, sgn*sLen/2]);
        patches=(mkPatch(X,Y)-mkSlot(X,1))+(mkPatch(-X,Y)-mkSlot(-X,1)) ...
               +(mkPatch(X,-Y)-mkSlot(X,-1))+(mkPatch(-X,-Y)-mkSlot(-X,-1));
        strips=mkSt(X,1)+mkSt(-X,1)+mkSt(X,-1)+mkSt(-X,-1);
        lineR=antenna.Rectangle(Length=X+wStrip,Width=w50,Center=[X/2 0]);
        lineL=antenna.Rectangle(Length=X+wStrip,Width=w50,Center=[-X/2 0]);
        xfmr=antenna.Rectangle(Length=wF,Width=LqF,Center=[0 -LqF/2]);
        fl=-(LqF)-portY;
        feed=antenna.Rectangle(Length=w50,Width=fl,Center=[0 (-(LqF)+portY)/2]);
        ft=patches+strips+lineR+lineL+xfmr+feed;
        pcb=pcbStack;
        pcb.BoardShape=antenna.Rectangle(Length=GPL,Width=GPW);
        pcb.BoardThickness=d.Thickness;
        pcb.Layers={ft, d, antenna.Rectangle(Length=GPL,Width=GPW)};
        pcb.FeedLocations=[0 portY 1 3]; pcb.FeedDiameter=w50/2;
        pcb.ViaLocations=[0 portY 1 3]; pcb.ViaDiameter=w50/2;
        Zin=impedance(pcb,freq);
    end
end
