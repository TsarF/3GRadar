function matchOpt()
% matchOpt - EM-in-the-loop tuning of the final two-element matcher.
% The array-side network (t1..t8) is fixed; we tune only OFFSET (length of
% 50ohm line from the junction to the final transformer) and Z_FINAL (the
% transformer's characteristic impedance). Two real knobs are exactly enough
% to null a complex reflection coefficient, so this converges to a match the
% idealized de-embed math could not pin down (overlay xfmr != ideal line).

freq = 3.25e9;
c = physconst("lightspeed");
d = dielectric("FR4"); d.EpsilonR = 4.3; d.Thickness = 1.6e-3;

W = c/(2*freq*sqrt((d.EpsilonR+1)/2));
epsilonEff = (d.EpsilonR+1)/2 + (d.EpsilonR-1)/2/sqrt(1+12*d.Thickness/W);
lambdaEff = c/(freq*sqrt(epsilonEff));
deltaL = 0.412*d.Thickness*(epsilonEff+0.3)*(W/d.Thickness+0.264)/(epsilonEff-0.258)/(W/d.Thickness+0.8);
L = lambdaEff/2 - 2*deltaL;

GPL = 0.12; GPW = 0.12;
patch = patchMicrostrip(Substrate=d, Height=d.Thickness, Length=L, Width=L, ...
    GroundPlaneLength=GPL/2, GroundPlaneWidth=GPW/2, FeedOffset=[0,0]);
spacing = lambdaEff*0.6;
arr = rectangularArray(Element=patch,RowSpacing=spacing,ColumnSpacing=spacing);

z0 = 50;
traceWidth  = traceThickness(z0,d);
traceWidth2 = traceThickness(z0*sqrt(2),d);
[Lq70,~,lg50_70] = quarterWave(traceWidth2,d,freq); %#ok<ASGLU>
[~,~,lg50] = quarterWave(traceWidth,d,freq);

x = abs(arr.FeedLocation(1,1));
y = abs(arr.FeedLocation(1,2));
feedLocation = -arr.GroundPlaneWidth/2 + 3e-3;
busLength = 2*Lq70;
yLower = y - L/2;
patchFeedLength = L/2 + yLower/4;
patchFeedCenter = y - patchFeedLength/2;
secondTLength = 2*x + traceWidth2;
feedLength = traceWidth/2 - feedLocation;
feedTraceCenter = feedLocation + feedLength/2;

t1 = antenna.Rectangle(Length=traceWidth2,Width=busLength,Center=[ x 0]);
t2 = antenna.Rectangle(Length=traceWidth2,Width=busLength,Center=[-x 0]);
t3 = antenna.Rectangle(Length=secondTLength,Width=traceWidth,Center=[0 0]);
t5 = antenna.Rectangle(Length=traceWidth,Width=patchFeedLength,Center=[ x  patchFeedCenter]);
t6 = antenna.Rectangle(Length=traceWidth,Width=patchFeedLength,Center=[-x  patchFeedCenter]);
t7 = antenna.Rectangle(Length=traceWidth,Width=patchFeedLength,Center=[ x -patchFeedCenter]);
t8 = antenna.Rectangle(Length=traceWidth,Width=patchFeedLength,Center=[-x -patchFeedCenter]);
t4 = antenna.Rectangle(Length=traceWidth,Width=feedLength,Center=[0 feedTraceCenter]);
arrayFeed = t1+t2+t3+t5+t6+t7+t8+t4;

% 3 knobs: p = [OFFSET(m), Z_FINAL(ohm), lenFac (xfmr length / lambdaG4)]
clamp = @(p)[min(max(p(1),0),0.026), min(max(p(2),12),120), min(max(p(3),0.4),1.6)];
vof = @(p) localVSWR(portZ3(clamp(p)), z0);

% --- seeds from the prior 2-knob run + neighbours (lenFac=1 => true lambdaG/4) ---
seeds = [10.84e-3 24.08 1.00;
         12.76e-3 28.00 1.00;
         19.14e-3 18.00 1.00;
          6.38e-3 21.40 1.00];
fprintf('seed evaluation (fixed mesh):\n');
bestV = inf; bestP = seeds(1,:);
for k = 1:size(seeds,1)
    Z = portZ3(seeds(k,:)); v = localVSWR(Z,z0);
    fprintf('  off=%5.2fmm Zf=%5.1f len=%.2f  Zin=%6.2f%+7.2fj  VSWR=%5.2f\n', ...
        seeds(k,1)*1e3, seeds(k,2), seeds(k,3), real(Z), imag(Z), v);
    if v < bestV, bestV = v; bestP = seeds(k,:); end
end

% --- Nelder-Mead refine, then one restart from the result ---
opts = optimset('Display','iter','MaxFunEvals',90,'TolX',1e-4,'TolFun',5e-3);
[pBest,vBest] = fminsearch(vof, bestP, opts);
[pBest,vBest] = fminsearch(vof, pBest, opts);
pBest = clamp(pBest);
Zf = portZ3(pBest);
fprintf('\n==== BEST MATCH (fixed mesh) ====\n');
fprintf('OFFSET = %.3f mm | Z_FINAL = %.2f ohm | lenFac = %.3f\n', pBest(1)*1e3, pBest(2), pBest(3));
fprintf('Zin = %.2f %+.2fj ohm | VSWR = %.3f\n', real(Zf), imag(Zf), vBest);

    function Zin = portZ3(p)
        OFFSET = p(1); Z_FINAL = p(2); lenFac = p(3);
        wF = traceThickness(Z_FINAL,d);
        [LqF,~,~] = quarterWave(wF,d,freq);
        finalLen = lenFac*LqF;
        t9 = antenna.Rectangle(Length=wF,Width=finalLen,Center=[0 -(OFFSET+finalLen/2)]);
        ft = arrayFeed + t9;
        pcb = pcbStack(arr);
        pcb.Layers{1,1} = pcb.Layers{1,1} + ft;
        pcb.FeedLocations = [0 feedLocation 1 3];
        pcb.ViaLocations  = pcb.FeedLocations(1,:);
        pcb.FeedDiameter  = traceWidth/2;
        pcb.ViaDiameter   = pcb.FeedDiameter;
        pcb.Layers{1,2}.Length = GPL;
        pcb.Layers{1,2}.Width  = GPW;
        mesh(pcb,'MaxEdgeLength',2e-3);   % fixed mesh => low-noise objective
        Zin = impedance(pcb,freq);
    end
end

function v = localVSWR(Z,z0)
    G = abs((Z-z0)/(Z+z0));
    v = (1+G)/(1-G);
end
