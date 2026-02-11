# When Test-Time Training Hurts: Diagnosing and Fixing Generalizable Log Parsing


## Parsing the benchmark log datasets

For RQ1, RQ4, and RQ6, we first need to parse the loghub-2k datasets.  We can do
this with the following commands.

```
python Online_parsing_withTTT_demo.py                   # Parse with Log3T
python Online_parsing_withTTT_demo.py --use_optimized   # Parse with GeLT
```

The outputs will be in the `output/` folder, for parsing results both with and
without Test-Time Training (TTT).

## Assessing mutated log generation

To evaluate the accuracy of mutated log generation strategy of Log3T, we can
run the following commands.

```
python mutated_log_generation_accuracy.py
```

## Assessing token-level classification

For both RQ3 and RQ5, run the following command.

```
python token_classification_accuracy.py
```

## Generating the graphs for RQ1/RQ6 and RQ3/RQ5

Refer to the scripts `rq1_rq6.py` and `rq3_rq5.py`
