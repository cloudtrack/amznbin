#!/bin/bash
run="python3"
main="src/main.py"

${run} ${main} --mode train --model "VGGNET" --function "classify" --difficulty "hard" --batch 50 #--continue-train True
