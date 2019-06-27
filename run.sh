#!/bin/sh

if [ "$#" -ne 1 ]; then
    echo "Usage: ./run.sh IMAGES_DIR"
    exit 1
fi

./build.sh

DATAS_DIR="$(python -c "import os; print(os.path.realpath('$1'))")"

if [ ! -d output ]; then
    echo "Creating output directory"
    mkdir output
fi

docker run -v "$(pwd)/output:/output" -v "${DATAS_DIR}:/datas" abeille-cool

echo "Check output directory for results"
