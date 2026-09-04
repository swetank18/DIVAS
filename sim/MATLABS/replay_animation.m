function replay_animation(jsonfile, save_video)
%REPLAY_ANIMATION Side-by-side MATLAB playback of a real closed-loop run.
%
%   replay_animation()                     plays the default replay file
%   replay_animation('docs/replay-mixed_traffic.json')
%   replay_animation(jsonfile, true)       also writes an .avi next to it
%
%   Reads the exact same JSON the closed-loop runner exports for the web
%   replay (scripts/export_replay.py -- see docs/replay-*.json). This is
%   not a re-enactment: road, actors, ego pose, planned path and the
%   safety margin are the real trace from a real divas.eval.runner.run
%   call, frame by frame. If it looks different from the Python
%   comparison PNG for the same scenario/seed, one of the two has a bug.
%
%   Two panels, side by side: the left arm and the right arm from the
%   JSON (usually baseline_conventional vs. the proposed stack), same
%   seed, therefore identical traffic -- the only fair way to show two
%   stacks side by side. See scripts/make_comparison.py for the Python
%   equivalent this mirrors.
%
%   No toolboxes. jsondecode and VideoWriter are both base MATLAB.
%
%   See also SIM/MATLAB/VALIDATE_AGAINST_PYTHON, which cross-validates
%   the *model* this replay is a recording of; this script draws a
%   recording, it does not re-simulate anything.

if nargin < 1 || isempty(jsonfile)
    here = fileparts(mfilename('fullpath'));
    jsonfile = fullfile(here, '..', '..', 'docs', 'replay-pedestrian_crossing.json');
end
if nargin < 2
    save_video = false;
end

if ~isfile(jsonfile)
    error('replay_animation:notfound', ...
        ['%s does not exist.\nGenerate one with:\n' ...
         '  python3 scripts/export_replay.py --scenario <name>'], jsonfile);
end

data = jsondecode(fileread(jsonfile));
fprintf('%s -- %s\n', data.scenario, data.description);
fprintf('tests: %s\n', data.tests);

fig = figure('Name', sprintf('DIVAS replay -- %s', data.scenario), ...
             'Color', 'w', 'Position', [100 100 1400 650]);

n_arms = numel(data.arms);
ax = gobjects(1, n_arms);
for a = 1:n_arms
    ax(a) = subplot(1, n_arms, a);
    hold(ax(a), 'on');
    draw_static_scene(ax(a), data);
    arm = idx_any(data.arms, a);
    title(ax(a), strrep(arm.stack, '_', ' '), 'Interpreter', 'none');
end

if save_video
    % 'MPEG-4' is a Windows/Mac-only VideoWriter profile -- it errors on
    % Linux, which is where this team's MATLAB install actually lives
    % (see STATUS.md). 'Motion JPEG AVI' works everywhere; re-encode with
    % ffmpeg afterwards if an .mp4 is needed for the deck.
    [p, f] = fileparts(jsonfile);
    vidfile = fullfile(p, [f '.avi']);
    v = VideoWriter(vidfile, 'Motion JPEG AVI');
    v.FrameRate = 20;
    open(v);
    fprintf('recording to %s\n', vidfile);
end

frame_counts = zeros(1, n_arms);
for a = 1:n_arms
    arm = idx_any(data.arms, a);
    frame_counts(a) = numel(arm.frames);
end
n_frames = max(frame_counts);

handles = cell(1, n_arms);
for a = 1:n_arms
    handles{a} = struct('ego', gobjects(0), 'actors', gobjects(0), ...
                         'path', gobjects(0), 'hud', gobjects(0), ...
                         'margin', gobjects(0), 'done', false);
end

for k = 1:n_frames
    for a = 1:n_arms
        arm = idx_any(data.arms, a);
        if k > numel(arm.frames)
            if ~handles{a}.done
                mark_outcome(ax(a), arm);
                handles{a}.done = true;
            end
            continue
        end
        frame = idx_any(arm.frames, k);
        handles{a} = draw_frame(ax(a), handles{a}, arm, frame, data.vehicle);
        if k == numel(arm.frames)
            mark_outcome(ax(a), arm);
            handles{a}.done = true;
        end
    end
    drawnow limitrate
    if save_video
        writeVideo(v, getframe(fig));
    else
        pause(0.03);
    end
end

if save_video
    close(v);
    fprintf('wrote %s\n', vidfile);
end
end

% -------------------------------------------------------------------------
function draw_static_scene(ax, data)
%DRAW_STATIC_SCENE Road corridor, centreline, and static obstacles -- drawn
%   once per panel, never redrawn per frame.
road = data.road;
patch(ax, road(:, 1), road(:, 2), [0.88 0.88 0.88], ...
      'EdgeColor', 'none', 'FaceAlpha', 1.0);
cl = data.centerline;
plot(ax, cl(:, 1), cl(:, 2), '--', 'Color', [0.55 0.55 0.55], 'LineWidth', 0.75);

for i = 1:numel(data.statics)
    s = idx_any(data.statics, i);
    if strcmp(s.kind, 'pothole')
        draw_circle(ax, s.x, s.y, s.r, [0.55 0.35 0.10], [0.75 0.55 0.25]);
    else
        corners = oriented_rect(s.x, s.y, s.theta, s.l, s.w);
        patch(ax, corners(:, 1), corners(:, 2), [0.4 0.4 0.4], 'EdgeColor', 'k');
    end
end

axis(ax, 'equal');
xlim(ax, [min(road(:, 1)) - 2, max(road(:, 1)) + 2]);
ylim(ax, [min(road(:, 2)) - 2, max(road(:, 2)) + 2]);
xlabel(ax, 'x [m]'); ylabel(ax, 'y [m]');
box(ax, 'on');
end

% -------------------------------------------------------------------------
function h = draw_frame(ax, h, arm, frame, vehicle)
%DRAW_FRAME Update one panel to one recorded control step.
%   Actor graphics are fully recreated every frame rather than moved,
%   because the actor count and identities are not guaranteed constant --
%   an actor can be added or dropped between frames (e.g. leaving the
%   sensed window), and this is an offline render, not a real-time loop,
%   so the cost of recreation does not matter.
delete(h.ego); delete(h.actors); delete(h.path); delete(h.hud); delete(h.margin);

% -- planned path ribbon
if ~isempty(frame.path)
    p = frame.path;
    h.path = plot(ax, p(:, 1), p(:, 2), '-', 'Color', [0.85 0.33 0.10], ...
                  'LineWidth', 1.5);
else
    h.path = gobjects(0);
end

% -- ego, oriented rectangle
ec = oriented_rect(frame.ego(1), frame.ego(2), frame.ego(3), ...
                    vehicle.length, vehicle.width);
h.ego = patch(ax, ec(:, 1), ec(:, 2), [0.0 0.60 0.55], 'EdgeColor', 'k', ...
              'LineWidth', 1.0);

% -- the safety margin in force, as a ring around the ego -- the number
% that decides whether this frame's traffic gets a wide berth or a tight
% one. Drawn as an outline, not filled, so it never hides an actor inside it.
if isfield(frame, 'd_safe') && frame.d_safe > 0
    ring = circle_points(frame.ego(1), frame.ego(2), ...
                          frame.d_safe + 0.5 * hypot(vehicle.length, vehicle.width));
    h.margin = plot(ax, ring(:, 1), ring(:, 2), ':', ...
                     'Color', [0.9 0.6 0.1], 'LineWidth', 1.0);
else
    h.margin = gobjects(0);
end

% -- actors, oriented boxes, coloured and sized by class
%
% Each actor entry is [x, y, theta, class] -- three numbers and a string,
% so jsondecode cannot collapse it to a numeric matrix and gives a cell
% array instead (one 1x4 cell per actor). idx_any + a further {} handles
% both that and the plain-matrix case, in case a future export ever drops
% the class label and the row becomes homogeneous.
actor_handles = gobjects(0);
if ~isempty(frame.actors)
    for i = 1:numel_rows(frame.actors)
        a = idx_any(frame.actors, i);
        if iscell(a)
            ax_ = a{1}; ay_ = a{2}; ath = a{3}; cls = a{4};
        else
            ax_ = a(1); ay_ = a(2); ath = a(3); cls = 'unknown';
        end
        [hl, hw, col] = class_extent(cls);
        c = oriented_rect(ax_, ay_, ath, 2 * hl, 2 * hw);
        actor_handles(end + 1) = patch(ax, c(:, 1), c(:, 2), col, 'EdgeColor', 'k'); %#ok<AGROW>
    end
end
h.actors = actor_handles;

% -- HUD text, top-left of the panel in axis units
xl = xlim(ax);
outcome = 'running';
h.hud = text(ax, xl(1) + 1, max(ylim(ax)) - 1, ...
    sprintf('t=%.1fs  v=%.1fm/s  prog=%.0fm  d\\_safe=%.1fm  (%s)', ...
            frame.t, frame.v, frame.progress, frame.d_safe, outcome), ...
    'FontSize', 8, 'VerticalAlignment', 'top', 'FontName', 'FixedWidth');
end

% -------------------------------------------------------------------------
function mark_outcome(ax, arm)
%MARK_OUTCOME Stamp the final result once a run's last frame has played.
if arm.success
    txt = sprintf('GOAL REACHED -- %.0f m, %.1f m/s mean', ...
                   arm.progress_m, arm.mean_speed);
    col = [0.0 0.55 0.0];
elseif ~isempty(arm.collision)
    txt = sprintf('COLLISION: %s -- at %.0f m', arm.collision, arm.progress_m);
    col = [0.75 0.0 0.0];
else
    txt = sprintf('TIMED OUT -- %.0f m', arm.progress_m);
    col = [0.6 0.4 0.0];
end
xl = xlim(ax); yl = ylim(ax);
text(ax, mean(xl), yl(1) + 1.5, txt, 'Color', col, 'FontWeight', 'bold', ...
     'FontSize', 10, 'HorizontalAlignment', 'center', 'BackgroundColor', 'w');
end

% -------------------------------------------------------------------------
function corners = oriented_rect(x, y, theta, length_, width_)
%ORIENTED_RECT Four corners of a length x width box centred at (x,y),
%   rotated by theta -- the same box every ego/actor footprint in this
%   repo is drawn as. Order is closed for direct use with patch().
hl = length_ / 2; hw = width_ / 2;
local = [ hl  hw; hl -hw; -hl -hw; -hl  hw];
c = cos(theta); s = sin(theta);
R = [c -s; s c];
world = (R * local')' + [x, y];
corners = world;
end

function pts = circle_points(x, y, r, n)
if nargin < 4, n = 40; end
th = linspace(0, 2 * pi, n)';
pts = [x + r * cos(th), y + r * sin(th)];
end

function draw_circle(ax, x, y, r, edge, face)
pts = circle_points(x, y, r);
patch(ax, pts(:, 1), pts(:, 2), face, 'EdgeColor', edge, 'LineWidth', 1.0);
end

function e = idx_any(x, i)
%IDX_ANY Index element i whether jsondecode gave a struct/matrix array or
%   a cell array -- which one you get depends on whether every element
%   had the same field names / types, and JSON alone doesn't guarantee
%   that (see draw_frame's actors handling and draw_static_scene's
%   statics handling for the two concrete cases this repo hits).
if iscell(x)
    e = x{i};
else
    e = x(i);
end
end

function n = numel_rows(x)
%NUMEL_ROWS Row count whether x is a cell array (one row per cell) or a
%   plain N x M numeric matrix.
if iscell(x)
    n = numel(x);
else
    n = size(x, 1);
end
end

function [hl, hw, col] = class_extent(cls)
%CLASS_EXTENT Half-length, half-width and display colour per actor class.
%   Mirrors divas.types.CLASS_EXTENT (Python) -- the two must stay in
%   step by hand, there is no shared source of truth across languages.
%   Sizes in metres; colours only need to be visually distinct.
switch cls
    case 'car'
        hl = 2.0; hw = 0.85; col = [0.95 0.75 0.10];
    case 'truck'
        hl = 3.5; hw = 1.2;  col = [0.90 0.45 0.05];
    case 'bus'
        hl = 5.5; hw = 1.3;  col = [0.70 0.20 0.20];
    case 'autorickshaw'
        hl = 1.3; hw = 0.7;  col = [0.95 0.55 0.10];
    case 'motorcycle'
        hl = 1.0; hw = 0.35; col = [0.10 0.55 0.85];
    case 'bicycle'
        hl = 0.9; hw = 0.3;  col = [0.10 0.65 0.85];
    case 'pedestrian'
        hl = 0.3; hw = 0.3;  col = [0.85 0.10 0.60];
    case 'animal'
        hl = 0.9; hw = 0.4;  col = [0.55 0.25 0.75];
    otherwise
        hl = 1.0; hw = 0.6;  col = [0.5 0.5 0.5];
end
end
