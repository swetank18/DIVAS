# One hypothesis for how an actor will move.
float32              probability      # sums to 1 across an actor's modes
geometry_msgs/Point[] points          # positions at t = dt, 2*dt, ... horizon
