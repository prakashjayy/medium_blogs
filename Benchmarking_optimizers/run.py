import os
import time
import datetime
import multiprocessing
import logging
from logging.handlers import RotatingFileHandler
from multiprocessing import Pool
from tensorflow.examples.tutorials.mnist import input_data
from mlp import BO


# Setting up logging
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

fh = RotatingFileHandler(os.getcwd()+'/logs'+'optimization_logs.csv')
fh.setLevel(logging.DEBUG)
fh.setFormatter(formatter)

logger.addHandler(fh)

# Adds the new logs to the file rather than overwriting the existing
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
ch.setFormatter(formatter)
logger.addHandler(ch)

def data_prep():
	mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

	return mnist

def run(optimizer, data):
	logger.info("===================Optimization using '{}' started==================="\
		.format(optimizer))

	try:
		X_train = data.train.images
		y_train = data.train.labels
		X_valid = data.validation.images
		y_valid = data.validation.labels
		X_test = data.test.images
		y_test = data.test.labels

		logger.info("====================Data loaded successfully for optimizer: '{}'\
			====================".format(optimizer))
	except Exception as e:
		logger.info("====================Data Loading Error for '{}'===================="\
			.format(optimizer))
		logger.error("The error encountered is {}".format(str(e)))

	try:
		model = BO(X_train, y_train, X_valid, y_valid, X_test, y_test)

		model.build_graph()

		start = time.time()
		
		model.compile_graph(optimize = optimizer, learning_rate = 0.01)

		logger.info("=======================The graph successfully compiled for '{}'"\
			.format(optimizer))

		logger.info("Saving the previous logs to archives with timestamp {}".\
			format(datetime.datetime.now().strftime('%d.%m.%Y %H:%M:%S,%f')))
		os.rename(os.getcwd()+"/optimizers/"+optimizer, \
			os.getcwd()+"/optimizers/"+optimizer+datetime.datetime.now().strftime('%d.%m.%Y %H:%M:%S,%f'))


		logger.info(":::::::::::::::::::::::::::Clearing the current Tensorboard \
			repository:::::::::::::::::::::::::::")
		
		# Saving the model
		model.train(summary_dir = os.getcwd()+"/optimizers/"+optimizer)

		logger.info("=======================The model saved successfully for '{}'".format(optimizer))

		# Opening up Tensorboard


		# Closing Tensorboard


		end = time.time()

		print ("The time required to finish {} optimization: {:0.2f}"\
			.format(optimizer, end-start))
		logger.info("The time required to finish {} optimization: {:0.2f}"\
			.format(optimizer, end-start))

	except Exception as e:
		logger.error("The error encountered is {}".format(str(e)))

if __name__ == '__main__':

	logger.info("===================The process has started===================")
	
	try:
		start = time.time()

		#optimizer = ["adam", "rmsprop"]

		logger.info("The CPU count is {}".format(multiprocessing.cpu_count()))
		print('\n')
		print ("The CPU count is {}".format(multiprocessing.cpu_count()))
		print ("Please select the optimizers from the list below")
		print ("(NOTE: Select an optimal number that doesn't crash your system)")

		print (":::::The optimizers available are (copy paste the same names):::::")
		print ("1. adam")
		print ("2. sgd")
		print ("3. rmsprop")
		print ("4. adadelta")
		print ("5. adagrad")
		print ("6. ftrl")
		print ("7. momentum")

		optimizerlist = input('Enter your optimizers as a list. Eg. ["adam", "sgd"] :::: ')
		optimizer = eval(optimizerlist)


		data = data_prep()
		
		for opt in optimizer:

			try:
				p = multiprocessing.Process(target=run, args=(opt, data, ))
				p.start()
				logger.error("Processing started successfully for:::::::::::::{}".format(opt))
			except:
				logger.error("The issue is with:::::::::::::{}".format(opt))
			#p.close()

		end = time.time()

		logger.info("The time required to finish entire optimization process: {:0.2f}".\
			format(optimizer, end-start))
		logger.info("=======================================The optimization process \
			completed successfully=======================================")

	except Exception as e:
		logger.error("One or more optimization processes encountered some problem")
		logger.error("The error encountered is {}".format(str(e)))


