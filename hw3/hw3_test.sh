#!/bin/bash
if [ -f "./first.h5" ]; then
    echo "File first.h5 exists, skip download."
else
    wget https://www.dropbox.com/s/6ua8py9tct6dhya/first.h5
fi
if [ -f "./ensemble.h5" ]; then
    # 檔案 /path/to/dir/filename 存在
    echo "File ensemble.h5 exists, skip download."
else
    wget https://www.dropbox.com/s/qsi069s9z9t2fgn/ensemble.h5
fi
python predict.py $1 $2 $3
