function cost = hornObjective(x, freq, Z0, wg)
%HORNOBJECTIVE  Cost function for pyramidal (flared) horn optimization.
%
%   COST = hornObjective(X, FREQ, Z0, WG) builds a pyramidal horn antenna
%   from the design vector X, evaluates it with the Antenna Toolbox method
%   of moments (MoM) solver at FREQ, and returns a scalar COST to be
%   MINIMIZED by the optimizer.
%
%   Design vector X (all lengths in metres):
%       x(1) = FlareWidth     E-plane aperture width
%       x(2) = FlareHeight    H-plane aperture height
%       x(3) = FlareLength    axial length of the flared section
%       x(4) = Length         length of the feeding rectangular waveguide
%       x(5) = FeedOffset(1)  axial position of the feed probe along the
%                             guide. This sets the probe-to-back-short
%                             distance and therefore the impedance match.
%                             It MUST be optimized: the horn is fed by a
%                             monopole/delta-gap probe whose default
%                             position is sized for the default WR-75 horn,
%                             so a hand-rescaled waveguide is grossly
%                             mismatched until this is re-tuned.
%
%   WG holds the FIXED feed-waveguide cross-section: wg.Width, wg.Height.
%
%   The objective is REALIZED GAIN = directivity + 10*log10(1-|Gamma|^2).
%   Folding the mismatch into the objective stops the optimizer from
%   inflating directivity with a reflective feed (that is what produced the
%   VSWR ~ 1e4). A poorly matched design now scores poorly by construction.

    % --- Build the antenna from the design vector -----------------------
    ant             = horn;
    ant.Width       = wg.Width;     % feed waveguide broad wall  (fixed)
    ant.Height      = wg.Height;    % feed waveguide narrow wall (fixed)
    ant.FlareWidth  = x(1);
    ant.FlareHeight = x(2);
    ant.FlareLength = x(3);
    ant.Length      = x(4);
    ant.FeedOffset  = [x(5) 0];     % axial probe offset; lateral kept centred

    % --- Reject non-physical geometry quickly (no EM solve needed) ------
    % Aperture must exceed the feed, flare positive, and the probe must sit
    % inside the waveguide section (|offset| < half the guide length).
    if ant.FlareWidth  <= ant.Width  || ...
       ant.FlareHeight <= ant.Height || ...
       ant.FlareLength <= 0          || ...
       abs(x(5))       >= 0.5*x(4)
        cost = 1e3;
        return;
    end

    try
        % --- Radiation performance (Antenna Toolbox / MoM) --------------
        % Peak directivity over a coarse global grid (no boresight assumed).
        az    = 0:10:350;
        el    = -90:10:90;
        Dpeak = max(pattern(ant, freq, az, el), [], 'all');   % directivity [dBi]

        % --- Impedance match --------------------------------------------
        Zin   = impedance(ant, freq);
        gamma = (Zin - Z0) ./ (Zin + Z0);
        magG  = min(abs(gamma), 0.999);     % clamp: MoM can return |G|>=1

        % --- Realized gain = directivity minus mismatch loss ------------
        % This single quantity rewards high gain AND a good match together.
        realizedGain = Dpeak + 10*log10(1 - magG.^2);   % dBi

        cost = -realizedGain;               % minimise -> maximise realized gain
    catch
        % A failed mesh/solve must not kill the optimization run.
        cost = 1e3;
    end
end
