#!/bin/bash
run="python3"
main="src/main.py"

${run} ${main} --mode train --model "VGGNET" --function "classify" --difficulty "hard" --batch 20 --learning-rate 0.01 #--continue-train True
