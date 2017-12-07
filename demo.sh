#!/bin/bash
run="python3"
main="src/main.py"

${run} ${main} --mode train --model "LENET" --function "count" --difficulty "moderate" --batch 5 #--continue-train True
