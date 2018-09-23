MJPATH=$HOME/.mujoco/mjpro150/bin
prependonce LD_LIBRARY_PATH $MJPATH
prependonce LD_LIBRARY_PATH /usr/lib/nvidia-384/
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
PYTHONUSERBASE=$MUJOCOPIPDIR pip install --user --upgrade mujoco_py
MPYPATH=$MUJOCOPIPDIR/lib/python3.6/site-packages/
if [[ "$PYTHONPATH" != *"$MPYPATH"* ]]; then
    export PYTHONPATH=$MPYPATH:$PYTHONPATH
fi
if [[ "$PATH" != *"$MUJOCOPIPDIR/bin"* ]]; then
    export PATH=$MUJOCOPIPDIR/bin:$PATH
fi
