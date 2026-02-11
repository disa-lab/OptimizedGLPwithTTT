# When Test-Time Training Hurts: Diagnosing and Fixing Generalizable Log Parsing

This is a replication package.  The package includes the following contents.


```
.
├── analyze.py                              # Script for RQ3,RQ5
├── evaluator                               # Evaluator
├── figures                                 # Directory containing figures
├── GeLT                                    # GeLT optimizations in Log3T implementation
│   ├── GeLT.py
│   ├── preprocess.py
├── mutated_log_generation_accuracy.py      # Script for RQ2
├── Online_parsing_withTTT_demo.py          # Script for RQ1, RQ6
├── rq1_rq6.py                              # Script for generating graphs for RQ1,RQ6
├── rq3_rq5.py                              # Script for generating graphs for RQ3,RQ5
├── settings.py                             # Benchmark settings
├── token_classification_accuracy.py        # Script for RQ3,RQ5
├── torch_model
├── Transfomer_encoder                      # Transformer encoder model
├── variablelist_generator.py               # Script for historical variable and constants generator
├── Variableset                             # Directory containing historical variables and constants
```


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
