#!/bin/bash
if [ -f "./ensemble.h5" ]; then
    # 檔案 /path/to/dir/filename 存在
    echo "File ensemble.h5 exists, skip download."
else
    wget https://www.dropbox.com/s/47336h56evcxzrg/ensemble.h5
fi
python predict.py $1 $2 $3