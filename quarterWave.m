function [Lq, epsilonEff, lambdaG] = quarterWave(W, d, freq)
%QUARTERWAVE  Quarter-wave length of a microstrip line of a given width.
%   [Lq, epsilonEff, lambdaG] = quarterWave(W, d, freq) returns:
%       Lq          - physical quarter-wavelength (metres) for this line
%       epsilonEff  - effective permittivity for THIS trace width
%       lambdaG     - guided wavelength (metres) for THIS trace width
%
%   Inputs:
%       W     - microstrip trace width (metres). Pass the width that came
%               out of traceThickness(Z, d) for the impedance you want.
%       d     - dielectric object with EpsilonR and Thickness set.
%       freq  - design frequency (Hz).
%
%   Why this exists: lambdaEff in the main script is the PATCH's guided
%   wavelength (wide W, high epsilonEff). The feed traces are narrower and
%   have their own epsilonEff, so their lambda/4 is different. A quarter-
%   wave transformer only inverts impedance (Zin = Zt^2 / Zload) when it is
%   exactly lambdaG/4 long on its OWN line. Compute length and width from
%   the same line.
%
%   Uses the Hammerstad effective-permittivity model with the correct
%   branch for W/h < 1 and W/h >= 1.

    c  = physconst("lightspeed");
    er = d.EpsilonR;
    h  = d.Thickness;
    u  = W / h;                       % normalised width W/h

    if u >= 1
        epsilonEff = (er + 1)/2 + (er - 1)/2 * (1 + 12/u)^(-0.5);
    else
        epsilonEff = (er + 1)/2 + (er - 1)/2 * ( (1 + 12/u)^(-0.5) ...
                     + 0.04 * (1 - u)^2 );
    end

    lambda0 = c / freq;
    lambdaG = lambda0 / sqrt(epsilonEff);
    Lq      = lambdaG / 4;
end