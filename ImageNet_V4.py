# ImageNet- V4 Architecture using Tensorflow 


"""

Architecture: (Overall Schema)

Input (299*299*3)
Stem - output (35*35*384)
4*Inception-A output 35*35*384
Reduction-A  output - 17*17*1024
7*Inception-B output: 17*17*1024
Reduction-B output - 8*8*1536
3*Inception-C output 8*8*1536
Average-Pooling output 1536
Dropout(Keep-0.8) Output: 1536
Softmax - Output 1000

"""

"""
Input Images will be of shape [None,299,299,3] - 3 layer Images with dimension 299*299*3 
"""
import tensorflow as tf 
import numpy as np 


def init_weights(shape):
	return tf.Variable(tf.random_normal(shape,stddev=0.01),dtype=tf.float32)

"""
stem weights
"""


"""
stem Architecture 
"""
def stem_arc(inputdata,w1,w2,w3,w4,w5,w6,w7,w8,w9,w10,w11):
	stem = tf.nn.conv2d(inputdata,w1,strides=[1,2,2,1],padding="VALID",name="stem1") # Output shape (149*149*32)
	stem = tf.nn.conv2d(stem,w2,strides=[1,1,1,1],padding="VALID",name = "stem2")
	stem = tf.nn.conv2d(stem,w3,strides=[1,1,1,1],padding="SAME",name="stem3")
	stem1a= tf.nn.max_pool(stem,ksize=[1,3,3,1],strides=[1,2,2,1],padding="VALID",name="stem4.1")
	stem1b= tf.nn.conv2d(stem,w4,strides=[1,2,2,1],padding="VALID",name="stem4.2")
	stem = tf.concat(3,[stem1a,stem1b],name="concat1")
	stem2a1 = tf.nn.conv2d(stem,w5,strides=[1,1,1,1],padding="SAME",name="stem5a.1")
	stem2a2 = tf.nn.conv2d(stem2a1,w6,strides=[1,1,1,1],padding="VALID",name="stem5a.2")
	stem2b1 = tf.nn.conv2d(stem,w7,strides=[1,1,1,1],padding="SAME",name="stem5b.1")
	stem2b2 = tf.nn.conv2d(stem2b1,w8,strides=[1,1,1,1],padding="SAME",name="stem5b.2")
	stem2b3 = tf.nn.conv2d(stem2b2,w9,strides=[1,1,1,1],padding="SAME",name="stem5b.3")
	stem2b4 = tf.nn.conv2d(stem2b3,w10,strides=[1,1,1,1],padding="VALID",name="stem5b.4")
	stem = tf.concat(3,[stem2a2,stem2b4],name="concat2")
	stem3a = tf.nn.conv2d(stem,w11,strides=[1,2,2,1],padding="VALID",name="stem6a")
	stem3b = tf.nn.max_pool(stem,ksize=[1,2,2,1],strides=[1,2,2,1],padding="VALID",name="stem6b")
	stem = tf.concat(3,[stem3a,stem3b]) #output (35*35*384)
	return stem 

"""
4*Inception-A 
"""

def inceptionA1(stemw12,w13,w14,w15,w16,w17,w18):
	incep1a1 = tf.nn.avg_pool(stem,ksize=[1,2,2,1],strides=[1,1,1,1],padding="SAME",name="incep1a1")
	incep1a2 = tf.nn.conv2d(incep1a1,w12,strides=[1,1,1,1],padding="SAME",name="incep1a2")
	incep2a1 = tf.nn.conv2d(stem,w13,strides=[1,1,1,1],padding="SAME",name="incep2a1")
	incep3a1 = tf.nn.conv2d(stem,w14,strides=[1,1,1,1],padding="SAME",name="incep3a1")
	incep3a2 = tf.nn.conv2d(incep3a1,w15,strides=[1,1,1,1],padding="SAME",name="incep3a2")
	incep4a1 = tf.nn.conv2d(stem,w16,strides=[1,1,1,1],padding="SAME",name="incep4a1")
	incep4a2 = tf.nn.conv2d(incep4a1,w17,strides=[1,1,1,1],padding="SAME",name="incep4a2")
	incep4a3 = tf.nn.conv2d(incep4a2,w18,strides=[1,1,1,1],padding="SAME",name="incep4a3")
	inceptionA1 = tf.concat(3,[incep1a2,incep2a1,incep3a2,incep4a3])
	return inceptionA1


"""
ReductionA 

"""

def ReductionA(inceptionA4,w19,w20,w21,w22):
	reductiona1 = tf.nn.max_pool(inceptionA4,ksize=[1,3,3,1],strides=[1,2,2,1],padding="VALID",name="reductiona1")
	reductionb1 = tf.nn.conv2d(inceptionA4,w19,strides=[1,2,2,1],padding="VALID",name="reductionb1")
	reductionc11 = tf.nn.conv2d(inceptionA4,w20,strides=[1,1,1,1],padding="SAME",name="reductionc11")
	reductionc12 = tf.nn.conv2d(reductionc11,w21,strides=[1,1,1,1],padding="SAME",name="reductionc12")
	reductionc13 = tf.nn.conv2d(reductionc12,w22,strides=[1,2,2,1],padding="VALID",name="reductionc13")
	reductionA = tf.concat(3,[reductiona1,reductionb1,reductionc13])
	return reductionA





"""
inception-B block 
"""

#B1
def inceptionB(reductionA,w23,w24,w25,w26,w27,w28,w29,w30,w31,w32):
	incepb1a = tf.nn.avg_pool(reductionA,ksize=[1,2,2,1],strides=[1,1,1,1],padding="SAME",name="inceptionb1a")
	incepb1b = tf.nn.conv2d(incepb1a,w23,strides=[1,1,1,1],padding="SAME",name="incepb1b")
	incepb2a = tf.nn.conv2d(reductionA,w24,strides=[1,1,1,1],padding="SAME",name="incepb2a")
	incepb3a = tf.nn.conv2d(reductionA,w25,strides=[1,1,1,1],padding="SAME",name="incepb3a")
	incepb3b = tf.nn.conv2d(incepb3a,w26,strides=[1,1,1,1],padding="SAME",name="incepb3b")
	incepb3c = tf.nn.conv2d(incepb3b,w27,strides =[1,1,1,1],padding="SAME",name="incepb3c")
	incepb4a = tf.nn.conv2d(reductionA,w28,strides=[1,1,1,1],padding="SAME",name="incepb4a")
	incepb4b = tf.nn.conv2d(incepb4a,w29,strides=[1,1,1,1],padding="SAME",name="incepb4b")
	incepb4c = tf.nn.conv2d(incepb4b,w30,strides=[1,1,1,1],padding="SAME",name="incepb4c")
	incepb4d = tf.nn.conv2d(incepb4c,w31,strides=[1,1,1,1],padding="SAME",name="incepb4d")
	incepb4e = tf.nn.conv2d(incepb4d,w32,strides=[1,1,1,1],padding="SAME",name="incepb4e")
	inceptionB1 = tf.concat(3,[incepb1b,incepb2a,incepb3c,incepb4e])
	return inceptionB1
		
"""
Reduction-B

"""

def ReductionB(inceptionB7,w33,w34,w35,w36,w37,w38):
	reduction1a = tf.nn.max_pool(inceptionB7,ksize=[1,3,3,1],strides=[1,2,2,1],padding="VALID",name="reduction1a")
	reduction1b = tf.nn.conv2d(inceptionB7,w33,strides=[1,1,1,1],padding="SAME",name="reduction1b")
	reduction1b = tf.nn.conv2d(reduction1b,w34,strides=[1,2,2,1],padding="VALID",name="reduction1b2")
	reduction1c = tf.nn.conv2d(inceptionB7,w35,strides=[1,1,1,1],padding="SAME",name="reduction1c")
	reduction1c = tf.nn.conv2d(reduction1c,w36,strides=[1,1,1,1],padding="SAME",name="reduction1c2")
	reduction1c = tf.nn.conv2d(reduction1c,w37,strides=[1,1,1,1],padding="SAME",name="reductionc13")
	reduction1c = tf.nn.conv2d(reduction1c,w38,strides=[1,2,2,1],padding="VALID",name="reduction1c4")
	ReductionB = tf.concat(3,[reduction1a,reduction1b,reduction1c])
	return ReductionB 





"""
Inception-C

"""

def InceptionC(ReductionB,w39,w40,w41,w42,w43,w44,w45,w46,w47,w48):
	incepa = tf.nn.avg_pool(ReductionB,ksize=[1,2,2,1],strides=[1,1,1,1],padding="SAME",name="incepa")
	incepa = tf.nn.conv2d(incepa,w39,strides=[1,1,1,1],padding="SAME",name="incepa1")
	incepb = tf.nn.conv2d(ReductionB,w40,strides=[1,1,1,1],padding="SAME",name="incepb")
	incepc = tf.nn.conv2d(ReductionB,w41,strides=[1,1,1,1],padding="SAME",name="incepc")
	incepc1 = tf.nn.conv2d(incepc,w42,strides=[1,1,1,1],padding="SAME",name="incepc1")
	incepc2 = tf.nn.conv2d(incepc,w43,strides=[1,1,1,1],padding="SAME",name="incepc2")
	incepd = tf.nn.conv2d(ReductionB,w44,strides=[1,1,1,1],padding="SAME",name="incepd1")
	incepd = tf.nn.conv2d(incepd,w45,strides=[1,1,1,1],padding="SAME",name="incepd2")
	incepd = tf.nn.conv2d(incepd,w46,strides=[1,1,1,1],padding="SAME",name="incepd3")
	incepd1 = tf.nn.conv2d(incepd,w47,strides=[1,1,1,1],padding="SAME",name="incepd41")
	incepd2 = tf.nn.conv2d(incepd,w48,strides=[1,1,1,1],padding="SAME",name="incepd42")
	InceptionC = tf.concat(3,[incepa,incepb,incepc1,incepc2,incepd1,incepd2])
	return InceptionC 




with mnist_graph.as_default():
	# Building the model 

	# Stem- Weights 
	w1 = init_weights([3,3,3,32]) 
	w2 = init_weights([3,3,32,32])
	w3 = init_weights([3,3,32,64])
	w4 = init_weights([3,3,64,96])
	w5 = init_weights([1,1,160,64])
	w6 = init_weights([3,3,64,96])
	w7 = init_weights([1,1,160,64])
	w8 = init_weights([7,1,64,64])
	w9 = init_weights([1,7,64,64])
	w10 = init_weights([3,3,64,96])
	w11 = init_weights([3,3,192,192])

	# Stem 
	stem = stem_arc(inputdata,w1,w2,w3,w4,w5,w6,w7,w8,w9,w10,w11)

	# Inception-A weights 
	w12 = w122 = w123 = w124  = init_weights([1,1,384,96])
	w13 = w132 = w133 = w134 =  init_weights([1,1,384,96])
	w14 = w142 = w143 = w144 = init_weights([1,1,384,64])
	w15 = w152 = w153 = w154 = init_weights([3,3,64,96])
	w16 = w162 = w163 = w164 = init_weights([1,1,384,64])
	w17 = w172 = w173 = w174 = init_weights([3,3,64,96])
	w18 = w182 = w183 = w184 = init_weights([3,3,96,96])


	#4*Inception-A 
	inceptionA1 = inceptionA(stem,w12,w13,w14,w15,w16,w17,w18)
	inceptionA2 = inceptionA(inceptionA1,w122,w132,w142,w152,w162,w172,w182)
	inceptionA3 = inceptionA(inceptionA2,w123,w133,w143,w153,w163,w173,w183)
	inceptionA4 = inceptionA(inceptionA3,w124,w134,w144,w154,w164,w174,w184)


	# Reduction-A weights 
	w19 = init_weights([3,3,384,384])
	w20 = init_weights([1,1,384,192])
	w21 = init_weights([3,3,192,224])
	w22 = init_weights([3,3,224,256])

	# Reduction-A model 
	ReductionA = ReductionA(inceptionA4,w19,w20,w21,w22)


	#Inception-B weights 
	w23 = w232 = w233 = w234 = w235 = w236 = w237 =  init_weights([1,1,1024,128])
	w24 = w242 = w243 = w244 = w245 = w246 = w247 = init_weights([1,1,1024,384])
	w25 = w252 = w253 = w254 = w255 = w256 = w257 = init_weights([1,1,1024,192])
	w26 = w262 = w363 = w264 = w265 = w266 = w267 = init_weights([1,7,192,224])
	w27 = w272 = w273 = w274 = w275 = w276 = w277 = init_weights([7,1,224,256])
	w28 = w282 = w283 = w284 = w285 = w286 = w287 = init_weights([1,1,1024,192])
	w29 = w292 = w293 = w294 = w295 = w296 = w297 = init_weights([1,7,192,192])
	w30 = w302 = w303 = w304 = w305 = w306 = w307 = init_weights([7,1,192,224])
	w31 = w312 = w313 = w314 = w315 = w316 = w317 = init_weights([1,7,224,224])
	w32 = w322 = w323 = w324 = w325 = w326 = w327 = init_weights([7,1,224,256])


	#Inception-B model
	inceptionB1 = inceptionB(ReductionA,w23,w24,w25,w26,w27,w28,w29,w30,w31,w32)
	inceptionB2 = inceptionB(inceptionB1,w232,w242,w252,w262,w272,w282,w292,w302,w312,w322)
	inceptionB3 = inceptionB(inceptionB2,w233,w243,w253,w263,w273,w283,w293,w393,w313,w323)
	inceptionB4 = inceptionB(inceptionB3,w234,w244,w254,w264,w274,w284,w294,w394,w314,w324)
	inceptionB5 = inceptionB(inceptionB4,w235,w245,w255,w265,w275,w285,w295,w395,w315,w325)
	inceptionB6 = inceptionB(inceptionB5,w236,w246,w256,w266,w276,w286,w296,w396,w316,w326)
	inceptionB7 = inceptionB(inceptionB6,w237,w247,w257,w267,w277,w287,w297,w397,w317,w327)


	# Reduction-B weights
	w33 = init_weights([1,1,1024,192])
	w34 = init_weights([3,3,192,192])
	w35 = init_weights([1,1,1024,256])
	w36 = init_weights([1,7,256,256])
	w37 = init_weights([7,1,256,320])
	w38 = init_weights([3,3,320,320])

	# Reduction-B model 
	ReductionB = ReductionB(inceptionB7,w33,w34,w35,w36,w37,w38)


	# Inception-C weights 
	w39 = w392 = w393 = init_weights([1,1,1536,256])
	w40 = w402 = w403 = init_weights([1,1,1536,256])
	w41 = w412 = w413 = init_weights([1,1,1536,384])
	w42 = w422 = w423 = init_weights([1,3,384,256])
	w43 = w432 = w433 = init_weights([3,1,384,256])
	w44 = w442 = w443 = init_weights([1,1,1536,384])
	w45 = w452 = w453 = init_weights([1,3,384,448])
	w46 = w462 = w463 = init_weights([3,1,448,512])
	w47 = w472 = w473 = init_weights([3,1,512,256])
	w48 = w482 = w483 = init_weights([1,3,512,256])

	# 3* Inception model 
	InceptionC1 = InceptionC(ReductionB,w39,w40,w41,w42,w43,w44,w45,w46,w47,w48)
	InceptionC2 = InceptionC(InceptionC2,w392,w402,w412,w422,w432,w442,w452,w462,w472,w482)
	InceptionC3 = InceptionC(InceptionC3,w393,w403,w413,w423,w433,w443,w453,w463,w473,w483)

	# Average Pooling
	Average_Pooling = tf.nn.avg_pool(InceptionC3,ksize=[1,8,8,1],strides=[1,1,1,1],padding="VALID",name="avg_pool")

	# Drop-out layer 
	Dropout_layer = tf.nn.dropout(Average_Pooling,keep_prob=0.8,seed=45)

	# Flatten layer 
	Flatten_layer = tf.reshape(Dropout_layer,[-1,1536])

	# Output Softmax layer
	w_o = init_weights([1536,1000])
	Output_softmax = tf.nn.softmax(tf.matmul(Flatten_layer,w_o),name="Softmax_Layer")














