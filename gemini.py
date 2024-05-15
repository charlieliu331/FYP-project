import pathlib
import textwrap
from tqdm import tqdm

import google.generativeai as genai

from vertexai.preview.generative_models import (
    HarmCategory, 
    HarmBlockThreshold )
from google.cloud.aiplatform_v1beta1.types.content import SafetySetting
import google.api_core.exceptions



genai.configure(api_key='your_API_KEY')
for m in genai.list_models():
  if 'generateContent' in m.supported_generation_methods:
    print(m.name)

model = genai.GenerativeModel('gemini-pro')


def gemini_generate(input_file_name,output_file_name,output_error_file_name):
    with open(input_file_name,'r') as f, open(output_file_name,'w') as f1, open(output_error_file_name,'w') as f2:
        block_count=0
        for line_number,line in enumerate(tqdm(f,desc="processing lines"),start=1):
            # if line_number==100:
            #     break
            try:
                response = model.generate_content(line,safety_settings = [
            {
                "category":"HARM_CATEGORY_SEXUALLY_EXPLICIT",
                "threshold":"BLOCK_NONE",
            },
            {
                "category":"HARM_CATEGORY_DANGEROUS_CONTENT",
                "threshold":"BLOCK_NONE",
            },
            {
                "category":"HARM_CATEGORY_HATE_SPEECH",
                "threshold":"BLOCK_NONE",
            },
            {
                "category":"HARM_CATEGORY_HARASSMENT",
                "threshold":"BLOCK_NONE",
            },
        ])
                try:
                    if response.text is not None:
                        f1.write(response.text.strip()+'\n')
                except:
                    try:
                        if response.parts is not None:
                            f1.write(response.parts)
                            f1.write('\n')
                    except Exception as e:
                        block_count+=1
                        f2.write(f"Line {line_number}: {line.strip()}\n- Error: {str(response.prompt_feedback)}\n{str(e)}\n")
               
            except google.api_core.exceptions.InternalServerError as e:
                print(f"Internal Server Error encountered for line {line_number}. Skipping to next line.")
                # Optionally log the error or write to the error file
                f2.write(f"Line {line_number}: {line.strip()}\n- Error: Internal Server Error encountered. Skipping line.\n")
                block_count+=1
                continue
            except Exception as e:
                print(f"Unexpected error for line {line_number}: {e}")
                block_count+=1
                continue
        print(block_count)
        f2.write(f"number of blocks is {block_count}")
         
input_file= 'unpunctuated text file with prefix'      
output_file = 'file to be output'
output_error_file_name= 'log file'
gemini_generate(input_file,output_file,output_error_file_name)
