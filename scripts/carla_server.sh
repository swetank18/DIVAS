#!/usr/bin/env bash
# Start the CARLA server on the *discrete* GPU.
#
# This machine is a hybrid-graphics laptop: an AMD Radeon 780M integrated in the
# CPU and an RTX 3050 on PCI 01:00.0. Vulkan enumerates the AMD part as GPU0, so
# CarlaUE4 launched plainly picks the integrated chip -- which has no dedicated
# VRAM and renders a couple of frames a second, making a 20 Hz control loop
# meaningless. It does not error; it is just slow, which is the worst way for a
# problem like this to present itself.
#
# The two __NV_ variables put the NVIDIA card first in Vulkan's device list.
# Verify with:  __NV_PRIME_RENDER_OFFLOAD=1 __VK_LAYER_NV_optimus=NVIDIA_only \
#                 vulkaninfo --summary | grep deviceName
#
#   ./scripts/carla_server.sh                 # offscreen, low quality (batch)
#   ./scripts/carla_server.sh -quality-level=Epic --window   # for video capture
set -euo pipefail

CARLA_ROOT="${CARLA_ROOT:-$HOME/carla/0.9.16}"
[ -x "$CARLA_ROOT/CarlaUE4.sh" ] || {
    echo "no CarlaUE4.sh under $CARLA_ROOT -- set CARLA_ROOT" >&2
    exit 1
}

args=(-quality-level=Low -RenderOffScreen)
for a in "$@"; do
    if [ "$a" = "--window" ]; then
        # Rendering a window uses the desktop's GL/Vulkan stack. After a driver
        # upgrade the running session still has the old libraries mapped, so log
        # out and back in first or the window may fail to create.
        args=("${args[@]/-RenderOffScreen/}")
    else
        args+=("$a")
    fi
done

exec env __NV_PRIME_RENDER_OFFLOAD=1 __VK_LAYER_NV_optimus=NVIDIA_only \
     "$CARLA_ROOT/CarlaUE4.sh" "${args[@]}"
