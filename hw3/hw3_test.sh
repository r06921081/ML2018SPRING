#!/bin/bash
wget https://www.dropbox.com/s/6ua8py9tct6dhya/first.h5
wget https://www.dropbox.com/s/qsi069s9z9t2fgn/ensemble.h5
python predict.py $1 $2 $3
