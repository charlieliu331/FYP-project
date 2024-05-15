import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    pipeline
)
from peft import LoraConfig,PeftModel, get_peft_model
from trl import SFTTrainer
from datasets import Dataset
from tqdm import tqdm


def fine_tuning(training_set_path,test_set_path,base_model_name = 'meta-llama/Llama-2-13b-chat-hf', refined_model_name = 'llama-2-chat-13b-pr-0213-1', output_dir="./scratch/llama_finetune_results/"):
    
    # Dataset
    training_data={}
    testing_data={}
    with open(training_set_path,'r') as f: #TODO modified the instruction
        training_data['text'] = f.readlines()
    with open(test_set_path,'r') as f:
        testing_data['text'] = f.readlines()

    # Convert dictionaries to datasets
    training_dataset = Dataset.from_dict(training_data)
    testing_dataset = Dataset.from_dict(testing_data)

    
    # Model and tokenizer names
    #base_model_name ='scratch/Llama-2-13b-chat-hf'
    #base_model_name = "meta-llama/Llama-2-13b-hf"
    #refined_model = "llama-2-chat-13b-pr-0213-1" #You can give it your own name
    
    # Tokenizer
    llama_tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    llama_tokenizer.pad_token = llama_tokenizer.eos_token
    llama_tokenizer.padding_side = "right"  # Fix for fp16

    # Quantization Config
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=False
    )

    # Model
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        quantization_config=quant_config,
        device_map="auto",
    )
    base_model.config.use_cache = False
    base_model.config.pretraining_tp = 1

    # LoRA Config
    peft_parameters = LoraConfig(
        lora_alpha=16,
        lora_dropout=0.1,
        r=8,
        bias="none",
        task_type="CAUSAL_LM"
    )

    # Training Params
    train_params = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=1,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=1,
        optim="paged_adamw_32bit",
        save_steps=1000,
        logging_steps=100,
        learning_rate=2e-4,
        weight_decay=0.001,
        fp16=False,
        bf16=False,
        max_grad_norm=0.3,
        max_steps=-1,
        warmup_ratio=0.03,
        group_by_length=True,
        lr_scheduler_type="constant",
        #report_to="tensorboard"
    )

    # Trainer
    fine_tuning = SFTTrainer(
        model=base_model,
        train_dataset=training_dataset,
        eval_dataset=testing_dataset,
        peft_config=peft_parameters,
        dataset_text_field="text",
        tokenizer=llama_tokenizer,
        args=train_params,
        max_seq_length=1024 #TODO 2048->512
    )

    # # Training
    fine_tuning.train()

    # # Save Model
    fine_tuning.model.save_pretrained(refined_model)
    



def text_gen_eval_wrapper(model, tokenizer, file_name, output_file_name, model_id=1, show_metrics=True, temp=0.7): #TODO 2048->512
    """
    A wrapper function for inferencing, evaluating, and logging text generation pipeline.

    Parameters:
        model (str or object): The model name or the initialized text generation model.
        tokenizer (str or object): The tokenizer name or the initialized tokenizer for the model.
        prompt (str): The input prompt text for text generation.
        model_id (int, optional): An identifier for the model. Defaults to 1.
        show_metrics (bool, optional): Whether to calculate and show evaluation metrics.
                                       Defaults to True.
        max_length (int, optional): The maximum length of the generated text sequence.
                                    Defaults to 200.

    Returns:
        generated_text (str): The generated text by the model.
        metrics (dict): Evaluation metrics for the generated text (if show_metrics is True).
    """
    # Suppress Hugging Face pipeline logging
    #logging.set_verbosity(logging.CRITICAL)

    # Initialize the pipeline
    '''
    pipe = pipeline(task="text-generation",
                    model=model,
                    tokenizer=tokenizer,
                    max_length=max_length,
                    do_sample=True, #TODO True->False
                    top_k=3,
                    top_p = 0.9,
                    temperature=temp)
    '''
    # Generate text using the pipeline
    pipe = pipeline(task="text-generation",
                    model=model,
                    tokenizer=tokenizer,
                    max_length=2000,
                    batch_size=2, #8
                    top_k=1,
                    #return_tensors='pt'
                    ) #TODO change to appropriate length 512
    
    texts = [] #with index
    with open(file_name) as f:
        for idx, line in enumerate(f):
        #for line in f:
            prompt = line.strip().split("[/INST]")[0]+"[/INST]" 
            texts.append((idx,prompt))
            #texts.append(prompt)

    dataset = Dataset.from_dict({'index': [idx for idx, _ in texts], 'text': [text for _, text in texts]})
    #dataset = Dataset.from_dict({'text': texts})  
    results = []
    #TODO
    for batch in tqdm(DataLoader(dataset, batch_size=2,shuffle=False)):
    # for batch in tqdm(DataLoader(dataset, batch_size=16,shuffle=False)):
        indexes = batch['index']
        texts = batch['text']
        generated_texts = pipe(texts)

        indexed_results = list(zip(indexes, generated_texts))
        results.extend(indexed_results)
    sorted_results = sorted(results, key=lambda x: x[0])
        
    with open(output_file_name,'a') as f:
        for _, result in sorted_results:
            generated_text = result[0]['generated_text']#.split('\n')[0:2]
            f.write(generated_text+"\n")



if __name__=="__main__":
    #fine_tuning(training_set_path,test_set_path) #TODO uncomment this if you need to train; otherwise comment this for testing
    base_model_name ='scratch/Llama-2-13b-chat-hf'
    refined_model = "llama-2-chat-13b-pr-0213-0" # the name you defined in fine_tuning()

    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        low_cpu_mem_usage=True, #TODO True->False  to see if speed up inference must True for device_map
        return_dict=True,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    base_model.config.use_cache = False #TODO add this line to see if speed up inference
    model = PeftModel.from_pretrained(base_model, refined_model)
    model = model.merge_and_unload()

    # Reload tokenizer to save it
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # Restoration Punctuation
    file_path = "text file of unpunctuated sentences with prefix"
    output_file_path = "output file name"
    text_gen_eval_wrapper(base_model, tokenizer, file_path,"output file name",show_metrics=False)
    '''
    inst = "As a master of English grammar and punctuation, your task is to meticulously restore capitalization and insert missing punctuation marks into the following text without changing any words or adding new content. Ensure the text's original meaning and integrity are preserved, focusing solely on restoring punctuation and capitalization where it is needed. Pay close attention to proper nouns, ensuring their original spelling remains untouched. Below are examples to illustrate the desired transformations:\\nExample 1:\\nOriginal: curiosity prompts a desire for knowledge whereas it is only thou who knowest all things supremely\\nWith Punctuation: Curiosity prompts a desire for knowledge, whereas it is only thou who knowest all things supremely.\\nExample 2:\\nOriginal: good morning ms nga i am changsong and i am doing my final year project\\nWith Punctuation: Good morning Ms. Nga. I am Changsong, and I am doing my final year project.\\nUsing the guidance provided by the examples above, proceed to restore capitalization and insert the necessary punctuation marks into the text below:"
    query = input("Please input an unpunctuated sentence for punctuation restoration:")
    text_gen = pipeline(task="text-generation", model=model, tokenizer=tokenizer, max_length=320)
    output = text_gen(f"<s>[INST] {inst} {query} [/INST]")
    print(output[0]['generated_text'])
    '''