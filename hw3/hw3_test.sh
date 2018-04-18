#!/bin/bash
wget https://www.dropbox.com/s/6ua8py9tct6dhya/model.0279-0.7157.h5?dl=1
python predict.py $1 $2 $3
