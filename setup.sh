#!/bin/bash

PLATE_CONF="yolov4.cfg"
PLATE_WEIGHT="yolov4.weights"
LINE_CONF="yolo_line.cfg"
LINE_WEIGHT="yolo_line_2.weights"

if test -f $PLATE_CONF; then
    echo "'$PLATE_CONF' already exists"
else
    wget https://github.com/saahiluppal/alpr/releases/download/1.0/$PLATE_CONF
fi

if test -f $PLATE_WEIGHT; then
    echo "'$PLATE_WEIGHT' already exists"
else
    wget https://github.com/saahiluppal/alpr/releases/download/1.0/$PLATE_WEIGHT
fi

if test -f $LINE_CONF; then
    echo "'$LINE_CONF' already exists"
else
    wget https://github.com/saahiluppal/alpr/releases/download/1.0/$LINE_CONF
fi

if test -f $LINE_WEIGHT; then
    echo "'$LINE_WEIGHT' already exists"
else
    wget https://github.com/saahiluppal/alpr/releases/download/1.0/$LINE_WEIGHT
fi
