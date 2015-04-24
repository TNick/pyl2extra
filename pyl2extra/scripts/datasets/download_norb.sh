#!/bin/bash
# This script downloads norb dataset to $PYLEARN2_DATA_PATH/norb
# and $PYLEARN2_DATA_PATH/norb_small/original
# It is based on download_cifar10.sh from pylearn2


[ -z "$PYLEARN2_DATA_PATH" ] && echo "PYLEARN2_DATA_PATH is not set" && exit 1

NORB_BIG_DIR=$PYLEARN2_DATA_PATH/norb
NORB_BIG_URL="http://www.cs.nyu.edu/~ylclab/data"
NORB_SML_DIR=$PYLEARN2_DATA_PATH/norb_small/original
NORB_SML_URL="http://www.cs.nyu.edu/~ylclab/data/norb-v1.0-small/smallnorb"


which wget > /dev/null
WGET=$?
which curl > /dev/null
CURL=$?

if [ "$WGET" -eq 0 ]; then
    DL_CMD="wget --no-verbose -O -"
elif [ "$CURL" -eq 0 ]; then
    DL_CMD="curl --silent -o -"
else
    echo "You need wget or curl installed to download"
    exit 1
fi

function get_file {
    $DL_CMD $BASE_URL/$1.gz
    gzip -c $1.gz > $1
    rm $1.gz
}


echo "Downloading and extracting small NORB dataset into $NORB_SML_DIR..."
BASE_URL="$NORB_SML_URL"
if [ ! -d $NORB_SML_DIR ]; then
    mkdir -p $NORB_SML_DIR
fi
pushd $NORB_SML_DIR > /dev/null
get_file norb-v1.0-5x46789x9x18x6x2x96x96-training-dat.mat
get_file norb-v1.0-5x46789x9x18x6x2x96x96-training-cat.mat
get_file norb-v1.0-5x46789x9x18x6x2x96x96-training-info.mat
get_file norb-v1.0-5x01235x9x18x6x2x96x96-testing-dat.mat
get_file norb-v1.0-5x01235x9x18x6x2x96x96-testing-cat.mat
get_file norb-v1.0-5x01235x9x18x6x2x96x96-testing-info.mat
popd > /dev/null


echo "Downloading and extracting big NORB dataset into $NORB_BIG_DIR..."
BASE_URL="$NORB_BIG_URL"
if [ ! -d $NORB_BIG_DIR ]; then
    mkdir -p $NORB_BIG_DIR
fi
pushd $NORB_BIG_DIR > /dev/null
get_file norb-5x01235x9x18x6x2x108x108-testing-01-cat.mat
get_file norb-5x01235x9x18x6x2x108x108-testing-01-dat.mat
get_file norb-5x01235x9x18x6x2x108x108-testing-01-info.mat
get_file norb-5x01235x9x18x6x2x108x108-testing-02-cat.mat
get_file norb-5x01235x9x18x6x2x108x108-testing-02-dat.mat
get_file norb-5x01235x9x18x6x2x108x108-testing-02-info.mat
get_file norb-5x46789x9x18x6x2x108x108-training-01-cat.mat
get_file norb-5x46789x9x18x6x2x108x108-training-01-dat.mat
get_file norb-5x46789x9x18x6x2x108x108-training-01-info.mat
get_file norb-5x46789x9x18x6x2x108x108-training-02-cat.mat
get_file norb-5x46789x9x18x6x2x108x108-training-02-dat.mat
get_file norb-5x46789x9x18x6x2x108x108-training-02-info.mat
get_file norb-5x46789x9x18x6x2x108x108-training-03-cat.mat
get_file norb-5x46789x9x18x6x2x108x108-training-03-dat.mat
get_file norb-5x46789x9x18x6x2x108x108-training-03-info.mat
get_file norb-5x46789x9x18x6x2x108x108-training-04-cat.mat
get_file norb-5x46789x9x18x6x2x108x108-training-04-dat.mat
get_file norb-5x46789x9x18x6x2x108x108-training-04-info.mat
get_file norb-5x46789x9x18x6x2x108x108-training-05-cat.mat
get_file norb-5x46789x9x18x6x2x108x108-training-05-dat.mat
get_file norb-5x46789x9x18x6x2x108x108-training-05-info.mat
get_file norb-5x46789x9x18x6x2x108x108-training-06-cat.mat
get_file norb-5x46789x9x18x6x2x108x108-training-06-dat.mat
get_file norb-5x46789x9x18x6x2x108x108-training-06-info.mat
get_file norb-5x46789x9x18x6x2x108x108-training-07-cat.mat
get_file norb-5x46789x9x18x6x2x108x108-training-07-dat.mat
get_file norb-5x46789x9x18x6x2x108x108-training-07-info.mat
get_file norb-5x46789x9x18x6x2x108x108-training-08-cat.mat
get_file norb-5x46789x9x18x6x2x108x108-training-08-dat.mat
get_file norb-5x46789x9x18x6x2x108x108-training-08-info.mat
get_file norb-5x46789x9x18x6x2x108x108-training-09-cat.mat
get_file norb-5x46789x9x18x6x2x108x108-training-09-dat.mat
get_file norb-5x46789x9x18x6x2x108x108-training-09-info.mat
get_file norb-5x46789x9x18x6x2x108x108-training-10-cat.mat
get_file norb-5x46789x9x18x6x2x108x108-training-10-dat.mat
get_file norb-5x46789x9x18x6x2x108x108-training-10-info.mat
popd > /dev/null
