#!/bin/bash
run="python3"
main="src/main.py"

${run} ${main} --mode train --model "VGGNET" --function "count" --difficulty "moderate" --batch 98 --continue-train True
