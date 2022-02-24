import tensorflow as tf
import cv2
import sys
import os
import random
import matplotlib.pyplot as plt
from Network.Network import Unsupervised_HomographyModel
from Misc.MiscUtils import *
from Misc.DataUtils import *
import numpy as np
import argparse
from tqdm import tqdm
from matplotlib import pyplot as plt
from Misc.TFSpatialTransformer import *
import random

sys.dont_write_bytecode = True


def GenerateBatch(BasePath, DirNamesTrain, TrainLabels, ImageSize, MiniBatchSize):
	I1FullBatch = []
	PatchBatch = []
	CornerBatch = []
	I2Batch = []

	ImageNum = 0
	while ImageNum < MiniBatchSize:
		RandIdx = random.randint(0, len(DirNamesTrain)-1)

		RandImageName = BasePath + os.sep + DirNamesTrain[RandIdx] + '.jpg'   
		ImageNum += 1
		patchSize = 128
		r = 32

		img_orig = plt.imread(RandImageName)
		
		img_orig = np.float32(img_orig)

		if(len(img_orig.shape)==3):
			img = cv2.cvtColor(img_orig,cv2.COLOR_RGB2GRAY)
		else:
			img = img_orig

		img=(img-np.mean(img))/255
		img = cv2.resize(img,(320,240))
		x = np.random.randint(r, img.shape[1]-r-patchSize)  
		y = np.random.randint(r, img.shape[0]-r-patchSize)

		p1 = (x,y)
		p2 = (patchSize+x, y)
		p3 = (patchSize+x, patchSize+y)
		p4 = (x, patchSize+y)
		src = [p1, p2, p3, p4]
		src = np.array(src)
		dst = []
		for pt in src:
			dst.append((pt[0]+np.random.randint(-r, r), pt[1]+np.random.randint(-r, r)))

		H = cv2.getPerspectiveTransform(np.float32(src), np.float32(dst))
		H_inv = np.linalg.inv(H)
		warpImg = cv2.warpPerspective(img, H_inv, (img.shape[1],img.shape[0]))

		patch1 = img[y:y + patchSize, x:x + patchSize]
		patch2 = warpImg[y:y + patchSize, x:x + patchSize]

		imgData = np.dstack((patch1, patch2))

		I1FullBatch.append(np.float32(img))
		PatchBatch.append(imgData)
		CornerBatch.append(np.float32(src))
		I2Batch.append(np.float32(patch2.reshape(128,128,1)))

	return I1FullBatch, PatchBatch, CornerBatch, I2Batch            

	
def TrainOperation(ImgPH, LabelPH, CornerPH, I2PH, I1FullPH, DirNamesTrain, TrainLabels, NumTrainSamples, ImageSize,
				   NumEpochs, MiniBatchSize, SaveCheckPoint, CheckPointPath,
				   DivTrain, LatestFile, BasePath, LogsPath, ModelType):
	pred_I2,I2 = Unsupervised_HomographyModel(ImgPH, CornerPH, I2PH, ImageSize, MiniBatchSize)

	with tf.compat.v1.name_scope('Loss'):
		loss = tf.reduce_mean(input_tensor=tf.abs(pred_I2 - I2))


	with tf.compat.v1.name_scope('Adam'):
		Optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=1e-5).minimize(loss)

	EpochLossPH = tf.compat.v1.placeholder(tf.float32, shape=None)
	loss_summary = tf.compat.v1.summary.scalar('LossEveryIter', loss)
	epoch_loss_summary = tf.compat.v1.summary.scalar('LossPerEpoch', EpochLossPH)

	MergedSummaryOP1 = tf.compat.v1.summary.merge([loss_summary])
	MergedSummaryOP2 = tf.compat.v1.summary.merge([epoch_loss_summary])

	Saver = tf.compat.v1.train.Saver()
	AccOverEpochs = np.array([0,0])
	with tf.compat.v1.Session() as sess:       
		if LatestFile is not None:
			Saver.restore(sess, CheckPointPath + LatestFile + '.ckpt')
			StartEpoch = int(''.join(c for c in LatestFile.split('a')[0] if c.isdigit()))
			print('Loaded latest checkpoint with the name ' + LatestFile + '....')
		else:
			sess.run(tf.compat.v1.global_variables_initializer())
			StartEpoch = 0
			print('New model initialized....')

		Writer = tf.compat.v1.summary.FileWriter(LogsPath, graph=tf.compat.v1.get_default_graph())
			
		for Epochs in tqdm(range(StartEpoch, NumEpochs)):
			NumIterationsPerEpoch = int(NumTrainSamples/MiniBatchSize/DivTrain)
			Loss=[]
			epoch_loss=0
			for PerEpochCounter in tqdm(range(NumIterationsPerEpoch)):
				if ModelType.lower() == "sup":
					ImgBatch,LabelBatch = GenerateBatch(BasePath, DirNamesTrain, TrainLabels, ImageSize, MiniBatchSize,ModelType)
					FeedDict = {ImgPH: ImgBatch, LabelPH: LabelBatch}
					
				else:
					I1FullBatch, PatchBatch, CornerBatch, I2Batch = GenerateBatch(BasePath, DirNamesTrain, TrainLabels, ImageSize, MiniBatchSize,ModelType)
					FeedDict = {ImgPH: PatchBatch, CornerPH: CornerBatch, I2PH: I2Batch}
					
				_, LossThisBatch, Summary = sess.run([Optimizer, loss, MergedSummaryOP1], feed_dict=FeedDict)
				Loss.append(LossThisBatch)
				epoch_loss = epoch_loss + LossThisBatch
				# Save checkpoint every some SaveCheckPoint's iterations
				if PerEpochCounter % SaveCheckPoint == 0:
					# Save the Model learnt in this epoch
					SaveName =  CheckPointPath + str(Epochs) + 'a' + str(PerEpochCounter) + 'model.ckpt'
					Saver.save(sess,  save_path=SaveName, save_format='h5')
					print('\n' + SaveName + ' Model Saved...')

				Writer.add_summary(Summary, Epochs*NumIterationsPerEpoch + PerEpochCounter)
			  

			epoch_loss = epoch_loss/NumIterationsPerEpoch
			
			print(np.mean(Loss))
			# Save model every epoch
			SaveName = CheckPointPath + str(Epochs) + 'model.ckpt'
			Saver.save(sess, save_path=SaveName, save_format='h5')
			print('\n' + SaveName + ' Model Saved...')
			Summary_epoch = sess.run(MergedSummaryOP2,feed_dict={EpochLossPH: epoch_loss})
			Writer.add_summary(Summary_epoch,Epochs)
			Writer.flush()
			

def main():
	Parser = argparse.ArgumentParser()
	Parser.add_argument('--BasePath', default='../Data', help='Base path of images, Default:/media/nitin/Research/Homing/SpectralCompression/COCO')
	Parser.add_argument('--CheckPointPath', default='../Checkpoints/', help='Path to save Checkpoints, Default: ../Checkpoints/')
	Parser.add_argument('--ModelType', default='unsup', help='Model type, Supervised or Unsupervised? Choose from Sup and Unsup, Default:Unsup')
	Parser.add_argument('--NumEpochs', type=int, default=50, help='Number of Epochs to Train for, Default:50')
	Parser.add_argument('--DivTrain', type=int, default=1, help='Factor to reduce Train data by per epoch, Default:1')
	Parser.add_argument('--MiniBatchSize', type=int, default=64, help='Size of the MiniBatch to use, Default:1')
	Parser.add_argument('--LoadCheckPoint', type=int, default=0, help='Load Model from latest Checkpoint from CheckPointsPath?, Default:0')
	Parser.add_argument('--LogsPath', default='Logs/', help='Path to save Logs for Tensorboard, Default=Logs/')

	Args = Parser.parse_args()
	NumEpochs = Args.NumEpochs
	BasePath = Args.BasePath
	DivTrain = float(Args.DivTrain)
	MiniBatchSize = Args.MiniBatchSize
	LoadCheckPoint = Args.LoadCheckPoint
	CheckPointPath = Args.CheckPointPath
	LogsPath = Args.LogsPath
	ModelType = Args.ModelType

	# Setup all needed parameters including file reading
	DirNamesTrain, SaveCheckPoint, ImageSize, NumTrainSamples, TrainLabels, NumClasses = SetupAll(BasePath, CheckPointPath)

	patchSize = 128
	# Find Latest Checkpoint File
	if LoadCheckPoint==1:
		LatestFile = FindLatestModel(CheckPointPath)
	else:
		LatestFile = None

	ImgPH = tf.compat.v1.placeholder(tf.float32, shape=(MiniBatchSize, 128, 128, 2))
	
	NumTrainSamples = 5000

	LabelPH = tf.compat.v1.placeholder(tf.float32, shape=(MiniBatchSize, 8))
	CornerPH = tf.compat.v1.placeholder(tf.float32, shape=(MiniBatchSize, 4,2))
	I2PH = tf.compat.v1.placeholder(tf.float32, shape=(MiniBatchSize, 128, 128,1))
	I1FullPH = tf.compat.v1.placeholder(tf.float32, shape=(MiniBatchSize, ImageSize[0], ImageSize[1],ImageSize[2]))
   
	TrainOperation(ImgPH, LabelPH,CornerPH, I2PH, I1FullPH, DirNamesTrain, TrainLabels, NumTrainSamples, ImageSize,
				   NumEpochs, MiniBatchSize, SaveCheckPoint, CheckPointPath,
				   DivTrain, LatestFile, BasePath, LogsPath, ModelType)
		
	
if __name__ == '__main__':
	main()

