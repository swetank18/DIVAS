function s = divas_bicycle_step(s, accel_cmd, steer_cmd, dt, p)
%DIVAS_BICYCLE_STEP One step of the kinematic bicycle, rate- and jerk-limited.
%
%   An independent MATLAB implementation of divas.sim.world.World.step, which
%   is the integrator underneath every number this project publishes. If the
%   two disagree, one of them is wrong.
%
%   S is a struct with fields x, y, theta, v, delta, a.
%
%   THE ORDER OF THESE LINES IS THE MODEL, not a detail. Position is advanced
%   with the speed and heading held at the *start* of the interval; heading
%   uses the start-of-interval speed but the *new* steer angle; and speed is
%   updated last. That is explicit Euler with a particular staggering, and any
%   other ordering is a different integrator that happens to converge to the
%   same continuous system as dt goes to zero. At dt = 0.05 s the difference
%   is visible, so it is reproduced exactly rather than tidied up.
%
%   See also DIVAS_LONGITUDINAL, VALIDATE_AGAINST_PYTHON.

% -- steering: clamp to lock, then rate-limit towards the command
steer = min(max(steer_cmd, -p.max_steer), p.max_steer);
max_d = p.max_steer_rate * dt;
s.delta = s.delta + min(max(steer - s.delta, -max_d), max_d);

% -- longitudinal: clamp to the envelope, then jerk-limit
accel = min(max(accel_cmd, p.min_accel), p.max_accel);
max_da = p.max_jerk * dt;
accel = min(max(accel, s.a - max_da), s.a + max_da);
s.a = accel;

% -- integrate
s.x = s.x + s.v * cos(s.theta) * dt;
s.y = s.y + s.v * sin(s.theta) * dt;
s.theta = divas_wrap_angle(s.theta + s.v / p.wheelbase * tan(s.delta) * dt);
s.v = min(max(s.v + accel * dt, 0.0), p.max_speed);
end
