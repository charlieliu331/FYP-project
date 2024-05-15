import torch
import os
import yaml
import argparse
import json

from tqdm import tqdm
from dotmap import DotMap
from autopunct.wrappers.BertPunctuatorWrapper import BertPunctuatorWrapper
import time

# model_path = '/home/users/ntu/liuc0062/scratch/new_Multilingual-Sentence-Boundary-detection/punctuator-model/libriheavy_10epoch-xlm-roberta-base-epoch-5_traced.pth'  #'/home/users/ntu/liuc0062/scratch/new_Multilingual-Sentence-Boundary-detection/punctuator-model/libriheavy_10epoch-xlm-roberta-base-epoch-6.pth'
model_path = '/home/users/ntu/liuc0062/scratch/new_Multilingual-Sentence-Boundary-detection/punctuator-model/libriheavy_10epoch-xlm-roberta-base-epoch-7_traced.pth'
ptype = 'all'

def get_config_from_yaml(yaml_file):
    with open(yaml_file, 'r') as config_file:
        config_yaml = yaml.load(config_file, Loader=yaml.Loader)
    # Using DotMap we will be able to reference nested parameters via attribute such as x.y instead of x['y']
    config = DotMap(config_yaml)
    return config



def sentenceCase(inSentence, lowercaseBefore):
    inSentence = '' if (inSentence is None) else inSentence
    if lowercaseBefore:
        inSentence = inSentence.lower()
    
    ## capitalize first letter
    words = inSentence.split()
    words[0] = words[0].capitalize() 

    ## finish the rest
    for i in range(0,len(words)-1):
        if words[i][-1] in ['.','?']:
            words[i+1] = words[i+1].capitalize()
    
    inSentence = " ".join(words)
    return inSentence    


def predict(message,model):
    if len(message) > 0:
        text = model.predict(message,ptype=ptype)
    else:
        text = "ERROR LENGTH = 0"
    return text.strip()


if __name__ == '__main__':
    
    #model = BertPunctuatorWrapper(get_config_from_yaml('./config-XLM-roberta-base-uncased.yaml'),torch.load(os.path.join(root_model_path,triple_model_path),map_location=torch.device('cpu')))  
    # model = BertPunctuatorWrapper(get_config_from_yaml('./config-XLM-roberta-base-uncased.yaml'),os.path.join(root_model_path,triple_model_path))
    model = BertPunctuatorWrapper(get_config_from_yaml('./config-XLM-roberta-base-uncased.yaml'),model_path)
    '''
    txt = 'hi this is a sample text to be tested because i need to test it'
        # Start timing
    start_time = time.time()

    # Execute the function you want to test
    # prediction = predict(txt, model)
    print(predict(txt,model))
    # End timing
    end_time = time.time()

    # Calculate the execution time
    execution_time = end_time - start_time

    print(execution_time)
    '''
    input_path = 'path to unpunctuated input text file'
    output_path = 'output text file path'
    with open(input_path,'r') as f1, open(output_path,'w') as f2:
        input_text = f1.readlines()
        output_text = []
        for line in tqdm(input_text):
            output_text.append(predict(line,model))
        f2.write('\n'.join(output_text))

    


