MJPATH=$HOME/.mujoco/mjpro150/bin
if [[ "$LD_LIBRARY_PATH" != *"$MJPATH"* ]]; then
    export LD_LIBRARY_PATH=/usr/lib/nvidia-384/:$LD_LIBRARY_PATH
    export LD_LIBRARY_PATH=$MJPATH:$LD_LIBRARY_PATH
fi
if [[ ! -f "/usr/bin/patchelf" ]]; then
    sudo apt install patchelf
fi
if [[ ! -f "/usr/include/GL/glew.h" ]]; then
    sudo apt install libglew-dev
fi
