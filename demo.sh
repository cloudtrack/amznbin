#!/bin/bash
run="python3"
main="src/main.py"

${run} ${main} --mode train --model "ALEXNET" --function "count" --difficulty "moderate" --batch 10 --learning-rate 0.00025 #--continue-train True
