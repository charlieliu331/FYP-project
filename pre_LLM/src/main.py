from neural_punctuator.utils.data import get_config_from_yaml
from neural_punctuator.wrappers.BertPunctuatorWrapper import BertPunctuatorWrapper
import logging
import argparse
import wandb

parser = argparse.ArgumentParser(description='arguments for the model')
parser.add_argument('--save-model', action='store_true',
                    help='save model')
parser.add_argument('--break-train-loop', action='store_true',
                    help='prevent training for debugging purposes')
parser.add_argument('--stage',type=str, default='',help='load model from checkpoint stage')
#parser.add_argument('--model-path',type=str,default='/home/liuc0062/Multilingual-Sentence-Boundary-detection/reproduce_result/neural_punctuator/models-xlm-roberta/',help='path to model directory')
parser.add_argument('--model-path',type=str,default='/home/liuc0062/punctuator-model/merged_mml_0epoch-',help='path to model directory')
#parser.add_argument('--data-path',type=str,default='/home/liuc0062/neural-punctuator-dataset/dual/xlm-roberta-base/',help='path to dataset directory')
parser.add_argument('--train-data-path',type=str,default='/home/liuc0062/preprocess/dataset/train_val_test/merged_mml/xlm-roberta-base/',help='path to train dataset directory')
parser.add_argument('--val-data-path',type=str,default='/home/liuc0062/preprocess/dataset/train_val_test/merged_mml/xlm-roberta-base/',help='path to eval dataset directory')
parser.add_argument('--eval-type',type=str,default='valid',help='valid_dataset or test_dataset for eval dataset')
parser.add_argument('--num-epochs',type=int,default=-1,help='no. of epochs to run the model')
parser.add_argument('--log-level',type=str,choices=['INFO','DEBUG','WARNING','ERROR'],default='INFO',help='logging info to be displayed')
parser.add_argument('--save-n-steps',type=int,help='Save after n steps, default=1 epoch',default=-1)
parser.add_argument('--force-save',action='store_true',help='Force save, overriding all settings')
parser.add_argument('--config',type=str, default='/home/liuc0062/Eperiod_Multilingual-Sentence-Boundary-detection_llama/src/neural_punctuator/configs/config-XLM-roberta-base-uncased.yaml', help='Path to config directory')
parser.add_argument('--action',type=str, default='train', help="train or val")


wandb.init(project="punctuator")

log_enum = {
    'INFO': logging.INFO,
    'DEBUG': logging.DEBUG,
    'WARNING': logging.WARNING,
    'ERROR': logging.ERROR
}

def override_arguments(args,config):
    logging.basicConfig(level=log_enum[args.log_level])
    
    if args.save_model:
        config.model.save_model=True
    
    if args.break_train_loop:
        if config.model.save_model:
            if args.force_save:
                logging.warning("Force saving despite break train loop!! Make sure you don't save too many epochs!!")
            else:
                logging.warning("Breaking train loop while save_model set to true. Overriding config to NOT save model")
                config.model.save_model=False
        config.debug.break_train_loop=True
    
    if args.num_epochs > 0:
        config.trainer.num_epochs = args.num_epochs
        logging.warning(f"config num_epochs overriden to {args.num_epochs}!!")
    
    if args.save_n_steps > 0:
        config.trainer.save_n_steps = args.save_n_steps
    else:
        config.trainer.save_n_steps = -1
        
    #config.data.data_path = args.data_path
    config.data.train_data_path = args.train_data_path
    config.data.val_data_path = args.val_data_path
    config.trainer.load_model = args.stage
    config.model.save_model_path = args.model_path
    config.data.eval_type = args.eval_type
    
    logging.info("*******************************")
    logging.info(str(config))
    logging.info("*******************************")
    
    if args.model_path:
        logging.warning(f"Config model path has been overridden!!!! New path is: {config.model.save_model_path}")
    
    return config
    

if __name__ == '__main__':
    args = parser.parse_args()
    config = get_config_from_yaml(args.config)
    #config = get_config_from_yaml(
    #    'neural_punctuator/configs/config-bert-base-multilingual-uncased.yaml')
    config = override_arguments(args,config)

    if config.model.save_model == False:
        print("WARNING, MODEL NOT BEING SAVED")
    if config.debug.break_train_loop == True:
        print("Warning, no training!")

    pipe = BertPunctuatorWrapper(config)
    if args.action == "train":
        pipe.train()
    elif args.action == "val" and args.stage is not None:
        pipe.validate()
    else:
        logging.error("check if you want to train or validate, if validate please check if your .pth file is stated")

