import os
import pandas as pd
import re

class format_log:
    '''
    # this part of code is from LogPai https://github.com/LogPai
    '''

    def __init__(self, log_format, indir='./'):
        self.path = indir
        self.logName = None
        self.df_log = None
        self.log_format = log_format

    def format(self, logName):
        self.logName=logName
        self.load_data()
        return self.df_log



    def generate_logformat_regex(self, logformat):
        """ Function to generate regular expression to split log messages
        """
        headers = []
        splitters = re.split(r'(<[^<>]+>)', logformat)
        regex = ''
        for k in range(len(splitters)):
            if k % 2 == 0:
                splitter = re.sub(' +', '\\\s+', splitters[k])
                regex += splitter
            else:
                header = splitters[k].strip('<').strip('>')
                regex += '(?P<%s>.*?)' % header
                headers.append(header)
        regex = re.compile('^' + regex + '$')
        return headers, regex
    def log_to_dataframe(self, log_file, regex, headers, logformat):
        """ Function to transform log file to dataframe
        """
        log_messages = []
        linecount = 0
        with open(log_file, 'r', encoding='UTF-8') as fin:
            for line in fin.readlines():
                try:
                    match = regex.search(line.strip())
                    message = [match.group(header) for header in headers]
                    log_messages.append(message)
                    linecount += 1
                except Exception as e:
                    pass
        logdf = pd.DataFrame(log_messages, columns=headers)
        logdf.insert(0, 'LineId', None)
        logdf['LineId'] = [i + 1 for i in range(linecount)]
        return logdf


    def load_data(self):
        headers, regex = self.generate_logformat_regex(self.log_format)
        self.df_log = self.log_to_dataframe(os.path.join(self.path, self.logName), regex, headers, self.log_format)


def wordsplit(log,dataset,regx=None,regx_use=False):
    '''
    Function to split the log to words using delimiters. If 'regx_use' is True, some words
    matched with 'regx' will be converted to '<*>'

    Input
    logs: list
    dataset: string
    regx: regular expression
    regx_use: boolean
    '''

    if dataset == 'HealthApp':
        log = re.sub(':', ': ', log)
        log = re.sub('=', '= ', log)
        log = re.sub(',', ', ', log)
    if dataset == 'Android':
        log = re.sub('\(', '( ', log)
        log = re.sub('\)', ') ', log)
        log = re.sub(':', ': ', log)
        log = re.sub('=', '= ', log)
        log = re.sub(',', ', ', log)
    if dataset == 'HPC':
        log = re.sub('=', '= ', log)
        log = re.sub(',', ', ', log)
    if dataset == 'Hadoop':
        log = re.sub(':', ': ', log)
        log = re.sub('=', '= ', log)

    if dataset == 'OpenSSH':
        log = re.sub('=', '= ', log)
        log = re.sub(':', ': ', log)
        log = re.sub(',', ', ', log)
    if dataset == 'Thunderbird':
        log = re.sub(':', ': ', log)
        log = re.sub('=', '= ', log)
        log = re.sub(',', ', ', log)
    if dataset == 'Windows':
        log = re.sub(':', ': ', log)
        log = re.sub('=', '= ', log)
        log = re.sub('\[', '[ ', log)
        log = re.sub(']', '] ', log)
        log = re.sub(',', ', ', log)
    if dataset == 'Zookeeper':
        log = re.sub(':', ': ', log)
        log = re.sub('=', '= ', log)
        log = re.sub(',', ', ', log)
    if dataset == 'Mac':
        log = re.sub('\[', '[ ', log)
        log = re.sub(']', '] ', log)
        log = re.sub(',', ', ', log)
    if dataset == 'BGL':
        log = re.sub('=', '= ', log)
        log = re.sub(',', ', ', log)
    if dataset == 'Proxifier':
        log = re.sub('\(.*?\)', '', log)
        log = re.sub(':', ' ', log)
        log = re.sub(',', ', ', log)
    if dataset == 'Linux':
        log = re.sub('=', '= ', log)
        log = re.sub(',', ', ', log)
    if regx_use == True:
        for ree in regx:
            log = re.sub(ree, '<*>', log)
    log = re.split(' +', log)
    return log

#  -- This is from LILAC ---------------------------------------------------

def post_process_tokens(tokens, punc):
    excluded_str = ['=', '|', '(', ')']
    for i in range(len(tokens)):
        if tokens[i].find("<*>") != -1:
            tokens[i] = "<*>"
        else:
            new_str = ""
            for s in tokens[i]:
                if (s not in punc and s != ' ') or s in excluded_str:
                    new_str += s
            tokens[i] = new_str
    return tokens

def _wordsplit(message,dataset,regx=None,regx_use=False):
    punc = "!\"#$%&'()+,-/:;=?@.[\]^_`{|}~"
    splitters = "\s\\" + "\\".join(punc)
    # splitter_regex = re.compile("([{}]+)".format(splitters))
    splitter_regex = re.compile("([{}])".format(splitters))
    tokens = re.split(splitter_regex, message)

    tokens = list(filter(lambda x: x != "", tokens))
    
    #print("tokens: ", tokens)
    tokens = post_process_tokens(tokens, punc)

    tokens = [
        token.strip()
        for token in tokens
        if token != "" and token != ' ' 
    ]
    tokens = [
        token
        for idx, token in enumerate(tokens)
        if not (token == "<*>" and idx > 0 and tokens[idx - 1] == "<*>")
    ]
    #print("tokens: ", tokens)
    #tokens = [token.strip() for token in message.split()]
    #print(tokens)
    return tokens

#  -- This is from LILAC ---------------------------------------------------
