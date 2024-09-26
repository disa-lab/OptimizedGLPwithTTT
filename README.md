# Log Parsing with Generalization Ability under New Log Types
——Proceedings of the 31st ACM Joint European Software Engineering Conference and Symposium on the Foundations of Software Engineering (ESEC/FSE 203)

## Note on Code Organization
This repository code that may not yet adhere to standardized conventions or best practices. While efforts are ongoing to improve clarity, readability, and organization. However, all provided code is capable of reproducing the intended results. Your patience and understanding are appreciated as we work towards enhancing the quality of this codebase.

## Reproduction

1.pip install -r requirements.txt

2.Run "Log3T/classify.py" to get the parsing accuracy on 16 benchmark datasets of Log3T directly.

3.Run "Log3T/parsing_time_training.py" to get the parsing accuracy with test-time-training at parsing time.

4.Run "Log3T/Unseentemplate_count.py" to get the number of unseen logs.

5.Run "Log3T/efficiency_evaluation.py" to reproduce results of efficiency evaluation (BGL dataset required)

6.The existing parser code reproduced in this paper relies on LogPai.

7.For the parsers compared in our experiment, we reproduce the code in https://github.com/logpai/logparser.

paper link: 


