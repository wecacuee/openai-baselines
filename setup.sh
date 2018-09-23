export PROJECT_NAME=floyd-warshall-rl/openai-baselines

THISFILE="${BASH_SOURCE[0]}"
if [[ -z "$THISFILE" ]]; then
    THISDIR=$(pwd)
else
    THISDIR=$(dirname $(readlink -m $THISFILE))
fi

# Prepend a path to environment variable only once
prependonce() {
    envname="$1";
    ppath="${2}";
    if [ ! -d "$ppath" ]; then
        echo "No such directory $ppath"
        return
    fi
    if [[ ":${!envname}:" != "*:$ppath:*" ]]; then
        eval ${envname}="${ppath}:${!envname}"
    fi
}

if [ -f /etc/profile.d/modules.sh ]; then
    . /etc/profile.d/modules.sh
    module use /z/home/dhiman/wrk/common/modulefiles/
    module load miniconda3 numpy/py3.6 cuda cudnn gflags/2.2.1 tensorflow/py3.6 ipython/py3.6
fi

export MID_DIR=/z/home/dhiman/mid/
PIPDIR=$MID_DIR/$PROJECT_NAME/build
PIPDIRVERSIONED=$PIPDIR-0.3.0
mkdir -p $PIPDIR
PYPATH=$PIPDIR/lib/python3.6/site-packages/
prependonce PYTHONPATH "$PYPATH"
prependonce PATH "$PIPDIR/bin"
. ${THISDIR}/setup-mujoco.sh

PYTHONUSERBASE=$PIPDIR pip install --user --upgrade -e $THISDIR
if [[ "$1" == "ENV" ]]; then
    prependonce PYTHONPATH "$THISDIR"
else
    mkdir -p $PIPDIRVERSIONED
    prependonce PYTHONPATH "$PIPDIRVERSIONED/lib/python3.6/site-packages/"
    prependonce PATH "$PIPDIRVERSIONED/bin"
    PYTHONUSERBASE=$PIPDIRVERSIONED pip install --user --upgrade -e $THISDIR
fi
