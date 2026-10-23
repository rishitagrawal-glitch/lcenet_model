@echo off
rem Run Strategy A
python "..\\strategy_a.py" --data "..\\dataset_extracted"

rem Run Strategy B
python "..\\strategy_b.py" --data "..\\dataset_extracted"

rem Run Baseline LCENET model
python "..\\baseline_lcenet.py" --data "..\\dataset_extracted"

echo All runs completed.
pause
