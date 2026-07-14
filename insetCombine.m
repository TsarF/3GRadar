function insetCombine()
% insetCombine - native inset 2x2: one EM 4-port solve (captures coupling),
% then combine through the corporate feed in the circuit domain (free to tune).
%
% Active impedance under uniform in-phase excitation: Gamma_i = sum_j S_ij,
% Z_active = Z0(1+G)/(1-G). By symmetry all 4 elements are identical, so the
% corporate input is a simple symmetric cascade:
%   Z_active --100ohm line(l1)--> ||2 --50ohm line(l2)--> ||2 --xfmr(Zf,lf)--> 50
% Tune (l1,l2,Zf,lf) for |Gamma_in|->0 at 3.25 GHz. Only the EM solve is slow.

freq = 3.25e9; freqRange = linspace(3.1e9,3.4e9,7);
d = dielectric("FR4"); d.EpsilonR = 4.3; d.Thickness = 1.6e-3;
wStrip = traceThickness(100,d);
L = 21.0e-3; notch = 2.5e-3;
c = physconst("lightspeed");
W0=c/(2*freq*sqrt((d.EpsilonR+1)/2));
ee0=(d.EpsilonR+1)/2+(d.EpsilonR-1)/2/sqrt(1+12*d.Thickness/W0);
lambdaEff=c/(freq*sqrt(ee0)); spacing=lambdaEff*0.6; X=spacing/2; Y=spacing/2;

mkEl=@() patchMicrostripInsetfed(Length=L,Width=L,Height=d.Thickness,Substrate=d, ...
    GroundPlaneLength=0.06,GroundPlaneWidth=0.06,StripLineWidth=wStrip, ...
    NotchWidth=2*wStrip,NotchLength=notch);

fprintf('EM: solving 4-port (coupled) over %d freqs...\n', numel(freqRange));
ca = conformalArray('Element',{mkEl(),mkEl(),mkEl(),mkEl()}, ...
    'ElementPosition',[X Y 0; -X Y 0; X -Y 0; -X -Y 0]);
S = sparameters(ca, freqRange);
save('insetS.mat','S');
P=S.Parameters; f=S.Frequencies; Z0=S.Impedance;

Zact = zeros(1,numel(f));
for k=1:numel(f), Sk=P(:,:,k); G=sum(Sk(1,:)); Zact(k)=Z0*(1+G)/(1-G); end
fprintf('Z_active (with coupling):\n');
for k=1:numel(f), fprintf('  %.3fGHz: %7.2f%+7.2fj\n', f(k)/1e9, real(Zact(k)),imag(Zact(k))); end
[~,i0]=min(abs(f-freq)); Za0=Zact(i0);

% ---- circuit combine ----
tlT=@(Z0l,bl,ZL) Z0l*(ZL+1i*Z0l*tan(bl))./(Z0l+1i*ZL*tan(bl));
lgOf=@(Zc,fk) 4*qlen(traceThickness(Zc,d),d,fk);
    function Gin = inputG(prm, ZaF, fk)
        l1=prm(1); l2=prm(2); Zf=min(max(prm(3),15),120); lf=prm(4);
        b1=2*pi/lgOf(100,fk); b2=2*pi/lgOf(50,fk); bf=2*pi/lgOf(Zf,fk);
        Za=tlT(100,b1*l1,ZaF); Zb=Za/2;
        Zc=tlT(50,b2*l2,Zb);   Zd=Zc/2;
        Ze=tlT(Zf,bf*lf,Zd);
        Gin=(Ze-50)/(Ze+50);
    end

obj=@(prm) abs(inputG(prm, Za0, freq));
lgf0=lgOf(35.4,freq);
p0=[5e-3, 8e-3, 35.4, lgf0/4];
opts=optimset('Display','off','MaxFunEvals',4000,'MaxIter',4000,'TolX',1e-6,'TolFun',1e-6);
[pB,gB]=fminsearch(obj,p0,opts);
pB(3)=min(max(pB(3),15),120);
fprintf('\nBest feed: l1=%.2fmm l2=%.2fmm Zf=%.2fohm lf=%.2fmm (lf/lgf=%.3f)\n', ...
    pB(1)*1e3,pB(2)*1e3,pB(3),pB(4)*1e3, pB(4)/lgOf(pB(3),freq));
fprintf('|Gin|@3.25 = %.4f  -> RL=%.2fdB VSWR=%.3f\n', gB, 20*log10(gB),(1+gB)/(1-gB));

fprintf('\nBand response with this feed:\n');
for k=1:numel(f)
    g=abs(inputG(pB,Zact(k),f(k)));
    fprintf('  %.3fGHz: RL=%6.2fdB VSWR=%5.2f\n', f(k)/1e9, 20*log10(g),(1+g)/(1-g));
end
end

function lq = qlen(W,d,freq)
    [lq,~,~]=quarterWave(W,d,freq);
end
