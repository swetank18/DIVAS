function validate_against_python(varargin)
%VALIDATE_AGAINST_PYTHON Cross-validate the DIVAS models against the Python stack.
%
%   Replays two reference trajectories exported by
%   scripts/export_for_matlab.py through independent MATLAB implementations of
%   the same two models, and reports the disagreement.
%
%       cd sim/matlab
%       validate_against_python
%
%   What this establishes, and what it does not.
%
%   It establishes that the kinematic bicycle and the longitudinal controller
%   are not artefacts of one codebase: written twice, in two languages, from
%   the same equations, they agree. That matters because every result in the
%   ablation is produced by that integrator, and because the pedal model was
%   identified against a live CARLA server tonight and nothing else had checked
%   it.
%
%   It does NOT validate the planner, the predictor, the risk field or the
%   perception stubs -- none of those are reimplemented here. It also cannot
%   catch an error shared by both implementations, because they were written
%   from the same equations by the same author. It is a check on transcription
%   and on arithmetic, which is the class of bug that actually happens.
%
%   Passes if both models agree to within TOL (default 1e-9, i.e. floating
%   point noise). Raises an error otherwise, so `matlab -batch` exits nonzero
%   and CI can use it.

opts = struct('tol', 1e-9, 'plot', true, 'outdir', '.');
for k = 1:2:numel(varargin)
    opts.(varargin{k}) = varargin{k+1};
end

here = fileparts(mfilename('fullpath'));
refdir = fullfile(here, 'reference');
if ~isfolder(refdir)
    error('divas:noReference', ...
          ['no reference data in %s\n' ...
           'run first:  python3 scripts/export_for_matlab.py'], refdir);
end

p = read_params(fullfile(refdir, 'params.csv'));
fprintf('DIVAS MATLAB cross-validation\n');
fprintf('  MATLAB %s\n', version('-release'));
fprintf('  wheelbase %.3f m   cruise %.2f m/s   dt %.3f s\n\n', ...
        p.wheelbase, p.cruise_speed, p.dt);

results = {};

% ---------------------------------------------------------------- bicycle
ref = readtable(fullfile(refdir, 'bicycle_reference.csv'));
n = height(ref);
s = struct('x', ref.x(1), 'y', ref.y(1), 'theta', ref.theta(1), ...
           'v', ref.v(1), 'delta', ref.delta(1), 'a', ref.a(1));
mine = zeros(n, 4);
mine(1, :) = [s.x, s.y, s.theta, s.v];
for k = 1:n-1
    s = divas_bicycle_step(s, ref.accel_cmd(k), ref.steer_cmd(k), p.dt, p);
    mine(k+1, :) = [s.x, s.y, s.theta, s.v];
end
got = [ref.x, ref.y, ref.theta, ref.v];
err_xy = max(abs(mine(:, 1:2) - got(:, 1:2)), [], 'all');
err_th = max(abs(divas_wrap_angle(mine(:, 3) - got(:, 3))));
err_v  = max(abs(mine(:, 4) - got(:, 4)));

results(end+1, :) = {'bicycle: position, m',  err_xy, opts.tol};
results(end+1, :) = {'bicycle: heading, rad', err_th, opts.tol};
results(end+1, :) = {'bicycle: speed, m/s',   err_v,  opts.tol};

% ----------------------------------------------------------- longitudinal
lref = readtable(fullfile(refdir, 'longitudinal_reference.csv'));
m = height(lref);
v = 0.0; a_meas = 0.0; trim = 0.0;
v_mine = zeros(m, 1);
for k = 1:m
    a_cmd = min(max(1.1 * (p.speed_reference - v), p.min_accel), p.max_accel);
    [throttle, brake, trim] = divas_longitudinal(a_cmd, v, a_meas, p.dt, p, trim);
    a_applied = throttle * p.throttle_gain - brake * p.brake_gain;
    resistance = p.resistance_c0 + p.resistance_c1 * v + p.resistance_c2 * v^2;
    v_next = max(0.0, v + (a_applied - resistance) * p.dt);
    v_mine(k) = v;
    a_meas = (v_next - v) / p.dt;
    v = v_next;
end
err_long = max(abs(v_mine - lref.v_closed));
results(end+1, :) = {'longitudinal: speed, m/s', err_long, opts.tol};

% ------------------------------------------------------------------ report
fprintf('%-28s %14s %12s   %s\n', 'model', 'max |error|', 'tolerance', '');
fprintf('%s\n', repmat('-', 1, 72));
ok = true;
for k = 1:size(results, 1)
    passed = results{k, 2} <= results{k, 3};
    ok = ok && passed;
    fprintf('%-28s %14.3e %12.0e   %s\n', results{k, 1}, results{k, 2}, ...
            results{k, 3}, ternary(passed, 'PASS', 'FAIL'));
end
fprintf('%s\n', repmat('-', 1, 72));

% The headline the deck can use: MATLAB independently reproduces the speeds
% the live CARLA calibration measured.
settled_closed = mean(v_mine(end-99:end));
settled_open   = mean(lref.v_open(end-99:end));
fprintf(['\nlongitudinal controller, settled speed against a %.1f m/s reference:\n' ...
         '  closed loop (MATLAB)      %.2f m/s\n' ...
         '  open loop   (old mapping) %.2f m/s\n' ...
         '  measured live in CARLA    9.01 / 7.88 m/s\n'], ...
        p.speed_reference, settled_closed, settled_open);

if opts.plot
    make_figure(ref, mine, got, lref, v_mine, p, opts.outdir);
end

if ~ok
    error('divas:crossValidationFailed', ...
          'the MATLAB and Python models disagree by more than %g', opts.tol);
end
fprintf('\nOK -- both models agree to floating-point noise.\n');
end

% =========================================================================

function p = read_params(path)
%READ_PARAMS Load the name,value CSV into a struct.
%   Read rather than hard-coded: a hand-copied wheelbase is the classic way a
%   cross-validation "fails" by comparing two correct implementations of
%   different vehicles.
t = readtable(path, 'TextType', 'string');
p = struct();
for k = 1:height(t)
    p.(char(t.name(k))) = double(t.value(k));
end
end

function out = ternary(cond, a, b)
if cond, out = a; else, out = b; end
end

function make_figure(ref, mine, got, lref, v_mine, p, outdir)
f = figure('Position', [100 100 1180 420], 'Color', 'w');

subplot(1, 3, 1);
plot(got(:, 1), got(:, 2), 'LineWidth', 3, 'Color', [0.75 0.75 0.75]); hold on;
plot(mine(:, 1), mine(:, 2), '--', 'LineWidth', 1.4, 'Color', [0.16 0.62 0.56]);
axis equal; grid on;
xlabel('x [m]'); ylabel('y [m]');
title('Kinematic bicycle: path');
legend({'Python (divas)', 'MATLAB'}, 'Location', 'best', 'FontSize', 8);

subplot(1, 3, 2);
plot(ref.t, got(:, 4), 'LineWidth', 3, 'Color', [0.75 0.75 0.75]); hold on;
plot(ref.t, mine(:, 4), '--', 'LineWidth', 1.4, 'Color', [0.16 0.62 0.56]);
grid on; xlabel('t [s]'); ylabel('speed [m/s]');
title('Kinematic bicycle: speed');

subplot(1, 3, 3);
yline(p.speed_reference, 'k--', 'LineWidth', 1); hold on;
plot(lref.t, lref.v_open, 'LineWidth', 1.6, 'Color', [0.76 0.07 0.12]);
plot(lref.t, lref.v_closed, 'LineWidth', 3, 'Color', [0.75 0.75 0.75]);
plot(lref.t, v_mine, '--', 'LineWidth', 1.4, 'Color', [0.16 0.62 0.56]);
grid on; xlabel('t [s]'); ylabel('speed [m/s]');
title('Longitudinal controller vs CARLA plant');
legend({'reference', 'open loop (old)', 'closed loop (Python)', ...
        'closed loop (MATLAB)'}, 'Location', 'southeast', 'FontSize', 7);

out = fullfile(outdir, 'matlab_cross_validation.png');
exportgraphics_compat(f, out);
fprintf('\nwrote %s\n', out);
end

function exportgraphics_compat(f, path)
%EXPORTGRAPHICS_COMPAT exportgraphics on R2020a+, print() before that.
%   which() rather than exist(..., 'file'): exportgraphics is shipped as a
%   built-in in some releases, and exist's 'file' filter answers 0 for those,
%   which would silently take the fallback path on a release that has it.
if ~isempty(which('exportgraphics'))
    exportgraphics(f, path, 'Resolution', 150);
else
    print(f, path, '-dpng', '-r150');
end
end
