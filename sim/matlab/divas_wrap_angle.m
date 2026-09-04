function out = divas_wrap_angle(a)
%DIVAS_WRAP_ANGLE Wrap an angle to (-pi, pi].
%
%   Mirrors divas.types.wrap_angle:
%       (a + pi) mod 2pi - pi
%
%   Written out rather than using wrapToPi, which lives in the Mapping
%   Toolbox. This validation must run on a base MATLAB + Simulink licence,
%   because a cross-validation that needs a toolbox the reviewer does not have
%   is a cross-validation nobody runs. MATLAB's mod() follows the sign of the
%   divisor, as Python's % does, so this is the same expression and not merely
%   a similar one -- rem() would differ for negative angles.
out = mod(a + pi, 2*pi) - pi;
end
