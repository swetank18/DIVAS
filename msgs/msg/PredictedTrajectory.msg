# Multi-modal prediction for one track.
uint32                  track_id
string                  cls
float64                 half_length
float64                 half_width
divas_msgs/TrajectoryMode[] modes
# Per-step confidence in [0,1], derived from how far apart the modes are in
# space.  This is the term that scales the controller's dynamic safety margin;
# it is published rather than recomputed so that planner and controller cannot
# disagree about it.
float32[]               confidence
