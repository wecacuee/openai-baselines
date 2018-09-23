export PROJECT_NAME=floyd-warshall-rl/openai-baselines

THISFILE="${BASH_SOURCE[0]}"
if [[ -z "$THISFILE" ]]; then
    THISDIR=$(pwd)
else
    THISDIR=$(dirname $(readlink -m $THISFILE))
fi

if [ -f /etc/profile.d/modules.sh ]; then
    . /etc/profile.d/modules.sh
    module use /z/home/dhiman/wrk/common/modulefiles/
    module load miniconda3 numpy/py3.6 cuda cudnn gflags/2.2.1 tensorflow/py3.6 ipython/py3.6
fi

export MID_DIR=/z/home/dhiman/mid/
PIPDIR=$MID_DIR/$PROJECT_NAME/build
mkdir -p $PIPDIR
PYPATH=$PIPDIR/lib/python3.6/site-packages/
if [[ "$PYTHONPATH" != *"$PYPATH"* ]]; then
    export PYTHONPATH=$PYPATH:$PYTHONPATH
fi
if [[ "$PATH" != *"$PIPDIR/bin"* ]]; then
    export PATH=$PIPDIR/bin:$PATH
fi
. ${THISDIR}/setup-mujoco.sh
#if [[ "$PYTHONPATH" != *"$THISDIR"* ]]; then
#    export PYTHONPATH=$THISDIR:$PYTHONPATH
#fi
PYTHONUSERBASE=$PIPDIR pip install --user --upgrade -e $THISDIR
