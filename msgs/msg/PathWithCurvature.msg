# Planner output.  nav_msgs/Path carries poses but not curvature, and the
# controller needs curvature for its speed reference.
std_msgs/Header header
nav_msgs/Path path
float64[]     kappa                     # 1/m, per pose
# False when the path ends merely because the route goal was that far ahead.
# The controller must not brake to a stop at the end of such a path.
bool          terminal_stop
