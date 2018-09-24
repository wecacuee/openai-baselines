MJPATH=$HOME/.mujoco/mjpro150/bin
prependonce LD_LIBRARY_PATH $MJPATH
prependonce LD_LIBRARY_PATH /usr/lib/nvidia-384/
prependonce LD_LIBRARY_PATH /usr/lib/nvidia-390/
prependonce LD_LIBRARY_PATH /usr/lib/nvidia-396/
if [[ ! -f "/usr/bin/patchelf" ]]; then
    sudo apt install patchelf
fi
if [[ ! -f "/usr/include/GL/glew.h" ]]; then
    sudo apt install libglew-dev
fi
# Install mujoco in a machine specific directory
MUJOCOPIPDIR=$PIPDIR/$(hostname)
mkdir -p $MUJOCOPIPDIR
MPYPATH=$MUJOCOPIPDIR/lib/python3.6/site-packages/
prependonce PYTHONPATH "$MPYPATH"
prependonce PATH "$MUJOCOPIPDIR/bin"
PYTHONUSERBASE=$MUJOCOPIPDIR pip install --user --upgrade mujoco_py
