freq = 3.25e9;
freqRange = linspace(3.1e9,3.4e9, 5);
c = physconst("lightspeed");
d = dielectric("FR4");
d.EpsilonR = 4.3;
d.Thickness = 1.6e-3;


W = c/(2*freq*sqrt((d.EpsilonR+1)/2));
epsilonEff = (d.EpsilonR+1)/2 + (d.EpsilonR-1)/2/sqrt(1+12*d.Thickness/W);
lambdaEff = c/(freq*sqrt(epsilonEff));
Leff = lambdaEff/2;
deltaL = 0.412*d.Thickness*(epsilonEff+0.3)*(W/d.Thickness+0.264)/(epsilonEff-0.258)/(W/d.Thickness+0.8);
L = Leff - 2*deltaL;



GroundPlaneLength = 0.12;
GroundPlaneWidth = 0.12;
patch = patchMicrostrip(Substrate=d, Height=d.Thickness, Length=L, Width=L,...
    GroundPlaneLength=GroundPlaneLength/2, GroundPlaneWidth=GroundPlaneWidth/2,...
    FeedOffset=[0,0])



spacing = lambdaEff*0.6;
arr = rectangularArray(Element=patch,RowSpacing=spacing,ColumnSpacing=spacing)



arr.Element



layout(arr)



figure
title("Rectangular Array of Microstrip Patch Antenna");
show(arr)



z0 = 50;
traceWidth = traceThickness(z0,d)



z1 = real(z0)*sqrt(2);
traceWidth2 = traceThickness(z1,d)


% ===================== FEED NETWORK =====================
% antenna.Rectangle(Length = X-extent, Width = Y-extent, Center = [x y])
%
% Impedance budget, source -> patches (symmetry keeps every node real):
%   50ohm port --t4(50ohm routing)--> [final lambdaG/4 xfmr t9] --> JUNCTION(Zj)
%   JUNCTION --t3(50ohm split bus)--> t1/t2 (70.7ohm, centre-fed: TWO lambdaG/4
%   arms) --t5..t8(50ohm patch routing)--> patch radiating edges.
%
% Textbook target: 50 -> (70.7 lines) -> 100 -> parallel-combine -> 25 ->
% final xfmr -> 50.  The patch here is SQUARE (Width=L), so its edge
% resistance is shifted from textbook and the upstream nodes will not land
% exactly on 100/25.  That is fine: we MEASURE the real junction load Zj with
% the probe below and size the final lambdaG/4 transformer as sqrt(Zj*50), so
% the match closes regardless of where the upstream nodes actually sit.
%
% KEY FIX vs the original: every transformer is a TRUE lambdaG/4 on its OWN
% line (quarterWave()).  Routing lengths (t3,t4,t5..t8) are SEPARATE, explicit
% variables; they only carry signal and are never used to set a transform
% ratio.  The original folded routing distance (2*y ~ lambdaG/2, feedLength/3)
% into the transformer lengths, which repeats the load instead of inverting it.

% ---- run-time controls (overridable from the calling workspace) ----
if ~exist('DEBUG_JUNCTION','var'); DEBUG_JUNCTION = false; end % probe raw Zj
if ~exist('SHOW','var');           SHOW = true;            end % draw figures
if ~exist('QUICK','var');          QUICK = false;          end % skip freq sweep
if ~exist('Z_FINAL','var');        Z_FINAL = 21.4;         end % final xfmr ohm
if ~exist('OFFSET','var');         OFFSET = 5.66e-3;       end % junction->xfmr line
%   The junction load is COMPLEX (15.2 - 39.5j ohm here, because the square
%   patch + 50ohm routing stubs leave reactance), so a bare lambdaG/4 cannot
%   match it. Two-element match: OFFSET of 50ohm line rotates the load onto the
%   real axis (a voltage minimum, ~9.1 ohm), then the final lambdaG/4 of
%   Z_FINAL = sqrt(50*9.1) ~ 21.4 ohm steps that up to 50 ohm.

% ---- line widths (each from its own characteristic impedance) ----
% traceWidth (50ohm) and traceWidth2 (70.7ohm) are computed above.
traceWidthFinal = traceThickness(Z_FINAL, d);

% ---- electrical (transformer) lengths: exactly lambdaG/4 on each line ----
[Lq70,  ~, lg70 ] = quarterWave(traceWidth2,     d, freq);  % 70.7ohm bus arms
[LqFin, ~, lgFin] = quarterWave(traceWidthFinal, d, freq);  % final transformer

% ---- node coordinates ----
x = abs(arr.FeedLocation(1,1));
y = abs(arr.FeedLocation(1,2));
offset = 3e-3;
feedLocation = -arr.GroundPlaneWidth/2 + offset;

% ---- vertical 70.7ohm buses (t1/t2): fed at their centre by t3, so each
%      half is one independent lambdaG/4 transformer ----
busHalf   = Lq70;          % transform section: exactly lambdaG/4
busLength = 2*busHalf;     % full bus = two back-to-back lambdaG/4 arms

% ---- patch routing (t5..t8): 50ohm line bridging bus end -> patch edge ----
yLower          = y - L/2;
patchFeedLength = L/2 + yLower/4;          % routing only (defined 50ohm line)
patchFeedCenter = y - patchFeedLength/2;
patchReach      = y - busHalf;             % geometric gap the routing covers

% ---- horizontal 50ohm split bus (t3): pure routing across to +/- x ----
secondTLength = 2*x + traceWidth2;

% ---- input 50ohm routing line (t4): port -> junction ----
feedLength      = traceWidth/2 - feedLocation;
feedTraceCenter = feedLocation + feedLength/2;

% ---- final lambdaG/4 transformer (t9): wide section overlaid on the top of
%      t4, ending exactly at the junction (y = 0) ----
finalLen    = LqFin;
finalCenter = -(OFFSET + finalLen/2);      % OFFSET of 50ohm line below junction,
                                           % then the lambdaG/4 transformer

% ---- report the length budget ----
fprintf('--- transformer lengths (true lambdaG/4) ---\n');
fprintf('70.7ohm bus arm : %6.2f mm  (lambdaG = %5.1f mm)\n', Lq70*1e3,  lg70*1e3);
fprintf('final %5.1fohm  : %6.2f mm  (lambdaG = %5.1f mm)\n', Z_FINAL, LqFin*1e3, lgFin*1e3);
fprintf('bus half-extent vs patch |y| : %.2f vs %.2f mm (routing reach %.2f mm)\n', ...
    busHalf*1e3, y*1e3, patchReach*1e3);

% ---- build sections ----
t1 = antenna.Rectangle(Length=traceWidth2,   Width=busLength,       Center=[ x  0]);
t2 = antenna.Rectangle(Length=traceWidth2,   Width=busLength,       Center=[-x  0]);
t3 = antenna.Rectangle(Length=secondTLength, Width=traceWidth,      Center=[ 0  0]);
t5 = antenna.Rectangle(Length=traceWidth,    Width=patchFeedLength, Center=[ x   patchFeedCenter]);
t6 = antenna.Rectangle(Length=traceWidth,    Width=patchFeedLength, Center=[-x   patchFeedCenter]);
t7 = antenna.Rectangle(Length=traceWidth,    Width=patchFeedLength, Center=[ x  -patchFeedCenter]);
t8 = antenna.Rectangle(Length=traceWidth,    Width=patchFeedLength, Center=[-x  -patchFeedCenter]);

arrayFeed = t1 + t2 + t3 + t5 + t6 + t7 + t8;   % everything array-side of t9

if DEBUG_JUNCTION
    % Probe: feed directly at the junction node [0 0]; no t4, no t9.
    % impedance() then reads the raw parallel-combined array load Zj.
    feedTrace   = arrayFeed;
    feedPortLoc = [0 0 1 3];
else
    t4 = antenna.Rectangle(Length=traceWidth,      Width=feedLength, Center=[0 feedTraceCenter]);
    t9 = antenna.Rectangle(Length=traceWidthFinal, Width=finalLen,   Center=[0 finalCenter]);
    feedTrace   = arrayFeed + t4 + t9;
    feedPortLoc = [0 feedLocation 1 3];
end

if SHOW
    figure(Name="Feed Trace")
    show(feedTrace)
end

arrPCB = pcbStack(arr);
arrPCB.Layers{1,1} = arrPCB.Layers{1,1} + feedTrace;
arrPCB.FeedLocations = feedPortLoc;
arrPCB.ViaLocations  = arrPCB.FeedLocations(1,:);
arrPCB.FeedDiameter  = traceWidth/2;
arrPCB.ViaDiameter   = arrPCB.FeedDiameter;
arrPCB.Layers{1,2}.Length = GroundPlaneLength;
arrPCB.Layers{1,2}.Width  = GroundPlaneWidth;
if SHOW
    figure
    title("PCB Antenna")
    show(arrPCB)
end

if DEBUG_JUNCTION
    Zj = impedance(arrPCB, freq);
    fprintf('\n==== JUNCTION PROBE @ %.3f GHz ====\n', freq/1e9);
    fprintf('Zj (raw array load behind final xfmr) = %.2f %+.2fj ohm\n', real(Zj), imag(Zj));
    fprintf('=> set Z_FINAL = sqrt(real(Zj)*50) = %.2f ohm\n', sqrt(real(Zj)*50));
    fprintf('   (~25ohm expected; ~62ohm => upstream half-wave t1/t2 repeating)\n');
else
    Zin = impedance(arrPCB, freq);
    G   = (Zin - z0)/(Zin + z0);
    fprintf('\n==== MATCH @ %.3f GHz (Z_FINAL = %.2f ohm) ====\n', freq/1e9, Z_FINAL);
    fprintf('Zin = %.2f %+.2fj ohm | RL = %.2f dB | VSWR = %.2f\n', ...
        real(Zin), imag(Zin), 20*log10(abs(G)), (1+abs(G))/(1-abs(G)));
    if ~QUICK
        figure; returnLoss(arrPCB,freqRange,z0);
        figure; vswr(arrPCB,freqRange,z0);
        figure; impedance(arrPCB,freqRange);
    end
end
