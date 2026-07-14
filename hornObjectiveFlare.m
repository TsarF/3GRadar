function [cost, D, RL] = hornObjectiveFlare(x, freq, Z0, antRef)
%HORNOBJECTIVEFLARE  Maximize directivity while preserving the match.
%
%   [COST,D,RL] = hornObjectiveFlare(X, FREQ, Z0, ANTREF) copies the matched
%   reference horn ANTREF, changes ONLY the flare to X, and returns:
%       COST = -D + wMatch*max(0, RLtarget - RL)   (MINIMIZED by ga)
%       D    = peak directivity [dBi]
%       RL   = return loss [dB], positive = good  (RL=24 means S11=-24 dB)
%
%   Design vector X (metres): x(1)=FlareWidth x(2)=FlareHeight x(3)=FlareLength
%
%   Two things make this version behave:
%   1) copy(antRef): antenna objects are HANDLE objects and design() tunes
%      the feed/flare together. Copying the reference and editing only the
%      flare reproduces the reference EXACTLY when X is the reference flare,
%      so the seeded design really is the matched -24 dB horn. Each call gets
%      its own copy, which is also required for parallel safety.
%   2) The flare is the impedance transition, so changing it detunes the
%      match. A strong return-loss penalty keeps the optimizer among matched
%      designs and maximizes directivity within them (a soft constraint).

    ant = copy(antRef);                 % preserve everything design() set
    ant.FlareWidth  = x(1);
    ant.FlareHeight = x(2);
    ant.FlareLength = x(3);

    % Reject non-physical geometry (aperture must exceed the guide).
    if ant.FlareWidth  <= ant.Width  || ...
       ant.FlareHeight <= ant.Height || ...
       ant.FlareLength <= 0
        cost = 1e3; D = NaN; RL = NaN;
        return;
    end

    try
        % Directivity (MoM). impedance() below reuses the cached solve.
        D   = max(pattern(ant, freq, 0:10:350, -90:10:90), [], 'all');   % dBi
        Zin = impedance(ant, freq);
        RL  = -20*log10(abs((Zin - Z0)/(Zin + Z0)));    % return loss [dB]

        RLtarget = 15;     % dB  -> keep S11 <= -15 dB
        wMatch   = 5;      % strong: an unmatched design (RL~0) costs ~+75,
                           %         which dwarfs any plausible directivity.
        cost = -D + wMatch * max(0, RLtarget - RL);
    catch
        cost = 1e3; D = NaN; RL = NaN;
    end
end
