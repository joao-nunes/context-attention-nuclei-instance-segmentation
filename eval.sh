#!/bin/bash

sources=("glas" "dpath" "consep" "pannuke" "crag")
outdirs=("mask-rcnn-se-resnet50-fpn-glas" "mask-rcnn-se-resnet50-fpn-dpath" "mask-rcnn-se-resnet50-fpn-consep" "mask-rcnn-se-resnet50-fpn-pannuke" "mask-rcnn-se-resnet50-fpn-crag")

for i in 0 1 2 3 4; do
    python3 eval-se-resnet.py --out ${outdirs[$i]} --source ${sources[$i]}
done
