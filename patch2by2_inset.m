% patch2by2_inset.m - 2x2 inset-fed microstrip patch array, 3.25 GHz, FR4.
% Clean binary corporate feed built entirely from MATCHED lines, so routing
% length never transforms impedance:
%
%   each patch  -> 100 ohm (inset) --100ohm line--> T(100||100=50ohm)
%   T(50) --50ohm line--> centre T(50||50=25ohm) --lambdaG/4 @ 35.4ohm--> 50ohm
%
% Only the final lambdaG/4 transforms; everything else is matched. The inset
% notch (calibrated in insetCal.m) sets each patch to 100 ohm real, which is
% what removes the reactive-load problem of the edge-fed version.
%
% Knobs to tune in-array (mutual coupling shifts the isolated calibration):
%   PATCH_L     - resonant square side
%   NOTCH_LEN   - inset depth (sets patch R)
%   Z_FINAL     - final transformer impedance (cleanup)

freq = 3.25e9;
freqRange = linspace(3.1e9,3.4e9,5);
c = physconst("lightspeed");
d = dielectric("FR4"); d.EpsilonR = 4.3; d.Thickness = 1.6e-3;

if ~exist('SHOW','var');  SHOW = true;  end
if ~exist('QUICK','var'); QUICK = false; end

% ---- calibrated patch (filled from insetCal.m) ----
if ~exist('PATCH_L','var');   PATCH_L   = 20.9e-3;  end
if ~exist('NOTCH_LEN','var'); NOTCH_LEN = 2.0e-3;   end
if ~exist('Z_FINAL','var');   Z_FINAL   = 35.4;     end
L = PATCH_L;

% ---- effective wavelength (for element spacing only) ----
W0 = c/(2*freq*sqrt((d.EpsilonR+1)/2));
ee0 = (d.EpsilonR+1)/2 + (d.EpsilonR-1)/2/sqrt(1+12*d.Thickness/W0);
lambdaEff = c/(freq*sqrt(ee0));

% ---- line widths ----
wStrip = traceThickness(100,d);     % 100 ohm patch lines
w50    = traceThickness(50,d);      % 50 ohm lines
wF     = traceThickness(Z_FINAL,d); % final transformer
wNotch = 2*wStrip;
[LqF,~,~] = quarterWave(wF,d,freq); % final lambdaG/4 length

% ---- layout ----
GroundPlaneLength = 0.12; GroundPlaneWidth = 0.12;
spacing = lambdaEff*0.6;
X = spacing/2; Y = spacing/2;
portY = -GroundPlaneWidth/2 + 3e-3;

innerEdge = Y - L/2;                 % y of inner radiating edge (top patches)
notchBottom = innerEdge + NOTCH_LEN; % strip reaches this y inside the patch

% ---- patches (square) ----
mkPatch = @(cx,cy) antenna.Rectangle(Length=L,Width=L,Center=[cx cy]);
pTR = mkPatch( X, Y); pTL = mkPatch(-X, Y);
pBR = mkPatch( X,-Y); pBL = mkPatch(-X,-Y);

% ---- inset notch slots (cleared from each inner edge) ----
mkSlot = @(cx,signY) antenna.Rectangle(Length=wNotch,Width=NOTCH_LEN, ...
    Center=[cx, signY*(Y - L/2 + NOTCH_LEN/2)]);
slotTR = mkSlot( X,+1); slotTL = mkSlot(-X,+1);
slotBR = mkSlot( X,-1); slotBL = mkSlot(-X,-1);

% ---- 100 ohm patch strips: notch bottom -> T at (+/-X, 0) ----
% top strip spans y in [0, notchBottom]; bottom strip spans y in [-notchBottom,0]
stripLenTop = notchBottom;           % from y=0 up into the notch
mkStripTop = @(cx) antenna.Rectangle(Length=wStrip,Width=stripLenTop, ...
    Center=[cx, stripLenTop/2]);
mkStripBot = @(cx) antenna.Rectangle(Length=wStrip,Width=stripLenTop, ...
    Center=[cx,-stripLenTop/2]);
sTR = mkStripTop( X); sTL = mkStripTop(-X);
sBR = mkStripBot( X); sBL = mkStripBot(-X);

% ---- 50 ohm lines: (+/-X,0) -> centre (0,0) ----
lineR = antenna.Rectangle(Length=X+wStrip,Width=w50,Center=[ X/2 0]);
lineL = antenna.Rectangle(Length=X+wStrip,Width=w50,Center=[-X/2 0]);

% ---- final lambdaG/4 transformer (25 -> 50) at centre, then 50 ohm feed ----
xfmr = antenna.Rectangle(Length=wF,Width=LqF,Center=[0 -(LqF/2)]);
feedLineLen = (-(LqF)) - portY;      % from xfmr bottom (y=-LqF) down to port
feed = antenna.Rectangle(Length=w50,Width=feedLineLen,Center=[0 (-(LqF)+portY)/2]);

feedTrace = (pTR - slotTR) + (pTL - slotTL) + (pBR - slotBR) + (pBL - slotBL) ...
    + sTR + sTL + sBR + sBL + lineR + lineL + xfmr + feed;

fprintf('L=%.2fmm notch=%.2fmm | strip(100)=%.2f 50ohm=%.2f final(%.1f)=%.2fmm Lq=%.2fmm\n',...
    L*1e3, NOTCH_LEN*1e3, wStrip*1e3, w50*1e3, Z_FINAL, wF*1e3, LqF*1e3);

if SHOW, figure(Name="Inset feed trace"); show(feedTrace); end

% ---- assemble PCB ----
arrPCB = pcbStack;
arrPCB.BoardShape = antenna.Rectangle(Length=GroundPlaneLength,Width=GroundPlaneWidth);
arrPCB.BoardThickness = d.Thickness;
gnd = antenna.Rectangle(Length=GroundPlaneLength,Width=GroundPlaneWidth);
arrPCB.Layers = {feedTrace, d, gnd};
arrPCB.FeedLocations = [0 portY 1 3];
arrPCB.FeedDiameter  = w50/2;
arrPCB.ViaLocations  = [0 portY 1 3];
arrPCB.ViaDiameter   = w50/2;

if SHOW, figure; title("Inset PCB Antenna"); show(arrPCB); end

Zin = impedance(arrPCB,freq);
G = (Zin-50)/(Zin+50);
fprintf('\n==== INSET ARRAY @ %.3f GHz (Z_FINAL=%.2f) ====\n', freq/1e9, Z_FINAL);
fprintf('Zin = %.2f %+.2fj ohm | RL = %.2f dB | VSWR = %.2f\n', ...
    real(Zin), imag(Zin), 20*log10(abs(G)), (1+abs(G))/(1-abs(G)));
if ~QUICK
    figure; returnLoss(arrPCB,freqRange,50);
    figure; vswr(arrPCB,freqRange,50);
    figure; impedance(arrPCB,freqRange);
end
