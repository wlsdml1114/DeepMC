import warnings
warnings.filterwarnings(action='ignore') 
import argparse
import torch

from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.plugins import DDPPlugin

from net.deepmc import DeepMC
from utils.dataloader import DeepMCDataLoader


def get_args():
	parser = argparse.ArgumentParser(description='Arguments for the testing purpose.')	
	parser.add_argument('--batch_size', type=int, required=False, default=32)
	parser.add_argument('--num_gpus', type=int, required=False, default=3)
	parser.add_argument('--num_epochs', type=int, required=False, default=2)
	parser.add_argument('--learning_rate', type=int, required=False, default=1e-3)
	parser.add_argument('--num_workers', type=int, required=False, default=24)
	parser.add_argument('--model_save', type=bool, required=False, default=True)
	parser.add_argument('--seq_len', type=int, required=False, default=24)
	parser.add_argument('--st_num', type=int, required=False, default=90)
	parser.add_argument('--predictor', type=str, required=False, default='평균 기온')
	parser.add_argument('--data_path', type=str, required=False, default='/home/ubuntu/jini1114/aws.csv')
	parser.add_argument('--model_path', type=str, required=False, default='/home/ubuntu/jini1114/ddp_test/model')
	args = parser.parse_args()

	return args

if __name__ == "__main__":
	#get arg
	args = get_args()
	print(args)

	#set logger
	wandb_logger = WandbLogger(project="deepmc")
	wandb_logger.config = args

	#dataloader loading
	dl = DeepMCDataLoader(
		file_path = '/home/ubuntu/jini1114/aws.csv',
		predictor = ['평균 기온', '최고 기온', '최저 기온'],
		target = '평균 기온', 
		seq_len = 24, 
		pred_len = 12
	)
	dl.setup()

	# setup model
	deepmc = DeepMC(
		num_encoder_hidden=7, 
		num_encoder_times=18,
		num_decoder_hidden = 20,
		num_decoder_times=12, 
		batch_size= 16, 
		num_of_CNN_stacks = 7,
		cnn_output_size = 105,
		num_feature = 3
	)

	# setup trainer
	# single GPU trainer
	'''
	trainer = Trainer(
		max_epochs=args.num_epochs, 
		gpus = 1,
		logger = wandb_logger
	)
	'''
	# ddp trainer
	trainer = Trainer(
		max_epochs=args.num_epochs, 
		gpus = args.num_gpus,
		accelerator = 'ddp',
		plugins = DDPPlugin(find_unused_parameters = False),
		logger = wandb_logger
	)

	#training
	trainer.fit(deepmc, datamodule=dl)
	trainer.save_checkpoint()