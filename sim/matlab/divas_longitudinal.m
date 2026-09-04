function [throttle, brake, trim] = divas_longitudinal(a_cmd, v, a_meas, dt, p, trim)
%DIVAS_LONGITUDINAL Commanded acceleration -> pedal positions, closed-loop.
%
%   An independent MATLAB implementation of
%   divas.sim.carla_bridge.LongitudinalTracker, the controller identified
%   against the live CARLA plant. Stage 6 emits an acceleration; CARLA takes
%   pedal positions; this is the conversion, and getting it wrong open-loop
%   cost the first live run 1.1 m/s of cruise speed.
%
%   Three pieces:
%     * feedforward of the measured coast-down resistance, so a zero
%       acceleration command holds speed instead of decaying. In CARLA that
%       resistance is engine braking and is about 4 m/s^2 at 9 m/s -- an order
%       of magnitude more than a real car's drag.
%     * measured pedal gains, because params.max_accel and params.min_accel
%       are the *planner's* comfort limits and not what the actuators do.
%     * integral trim on the acceleration tracking error, with anti-windup,
%       to absorb the fit's residual and any gradient.
%
%   TRIM is carried by the caller: this is a stateful controller and the trim
%   is its state.

v = max(v, 0.0);
a_cmd = min(max(a_cmd, p.min_accel), p.max_accel);

% Standstill hold. The feedforward asks for rolling-resistance throttle at
% every speed including zero, so a held stop needs an explicit branch or the
% vehicle creeps off the line.
if v < p.stop_speed && a_cmd <= 0.0
    throttle = 0.0;
    brake = 1.0;
    trim = 0.0;
    return
end

resistance = p.resistance_c0 + p.resistance_c1 * v + p.resistance_c2 * v^2;
a_req = a_cmd + resistance + trim;

if a_req >= 0.0
    raw = a_req / p.throttle_gain;
    throttle = min(max(raw, 0.0), 1.0);
    brake = 0.0;
    saturated = raw > 1.0;
elseif a_req > -p.coast_band
    % A deceleration this small is what coasting already delivers; pedalling
    % it would chatter between throttle and brake every step at cruise.
    throttle = 0.0;
    brake = 0.0;
    saturated = false;
else
    raw = -a_req / p.brake_gain;
    throttle = 0.0;
    brake = min(max(raw, 0.0), 1.0);
    saturated = raw > 1.0;
end

% Integrate last, and only when the pedal has room to act on it.
err = a_cmd - a_meas;
if ~(saturated && err * a_req > 0.0)
    trim = min(max(trim + p.throttle_ki * err * max(dt, 0.0), -p.i_limit), p.i_limit);
end
end
