#!/bin/bash
wget https://github.com/r06921081/ML2018SPRING/releases/download/0.0.0/model.0279-0.7157.h5
python predict.py $1 $2 $3
