# One tracked dynamic actor, in the odom frame.
# Mirrors divas.types.Track -- keep the two in step.

uint32   id
string   cls                # car | truck | bus | autorickshaw | motorcycle |
                            # bicycle | pedestrian | animal | unknown
float64  x                  # m
float64  y                  # m
float64  vx                 # m/s
float64  vy                 # m/s
float64  half_length        # m, oriented extent along heading
float64  half_width         # m, oriented extent across heading
float64[16] cov             # row-major 4x4 over (x, y, vx, vy)
