import numpy as np
from svmlight_loader import dump_svmlight_file
import time 
import os
import sys
import lmdb
from multiprocessing import Pool
import threading
import caffe
from caffe.proto import caffe_pb2

class DenseNet(caffe.Net):
	def __init__(self, source_model_file, target_model_file, pretrained_file, mean_file, src_layers=['fc6', 'fc7', 'fc8'], dst_layers=['fc6-conv', 'fc7-conv', 'fc8-conv']):
		# load the source model
		net = caffe.Net(source_model_file, pretrained_file, caffe.TEST)
		fc_params = {pr: (net.params[pr][0].data, net.params[pr][1].data) for pr in src_layers}
		
		# load the target model
		caffe.Net.__init__(self, target_model_file, pretrained_file, caffe.TEST)
		conv_params = {pr: (self.params[pr][0].data, self.params[pr][1].data) for pr in dst_layers}
		
		# transplanting parameters from the source to the target, the number of parameters is unchanged except 
		# the its shape
		for pr, pr_conv in zip(src_layers, dst_layers):
			conv_params[pr_conv][0].flat = fc_params[pr][0].flat
			conv_params[pr_conv][1][...] = fc_params[pr][1]
			
		self.transformer = caffe.io.Transformer({'data': self.blobs['data'].data.shape})
		self.transformer.set_mean('data', self._load_mean(mean_file).mean(1).mean(1))
		self.transformer.set_transpose('data', (2,0,1))
		self.transformer.set_channel_swap('data', (2,1,0))
		self.transformer.set_raw_scale('data', 255.0)
		
	def predict_densemap(self, images):
		out = self.forward_all(data=np.asarray([self.transformer.preprocess('data', im) for im in images]))
		# produce feature vector for each of image
		last_layer = self.params.items()[-1][0]
		compact_features = np.zeros((len(images), self.params[last_layer][0].data.shape[0]), dtype=np.float32)
		for i in range(len(images)):
			index_map = np.array(out['prob'][i].argmax(axis=0).ravel(), dtype=int)
			score_map = out['prob'][i].max(axis=0).ravel()
			
			feature_vector = np.zeros((1, len(index_map)), dtype=np.float32)
			for ix, score in zip(index_map, score_map):
				compact_features[i, ix] += score
		return compact_features
		
	def _load_mean(self, mean_file):
		if mean_file.split('.')[-1] == 'binaryproto':
			blob = caffe.proto.caffe_pb2.BlobProto()
			data = open(mean_file, 'rb').read()
			blob.ParseFromString(data)
			return np.array(caffe.io.blobproto_to_array(blob))[0]
		elif mean_file.split('.')[-1] == 'npy':
			return np.load(mean_file)


# create a super-class of Classifier, this  is the Extractor 
class Extractor(caffe.Classifier):
	def __init__(self, model_file, pretrained_file, image_dims=None, 
			mean=None, input_scale=None, raw_scale=None,
			channel_swap=None):
		caffe.Classifier.__init__(self, model_file, pretrained_file, image_dims=image_dims,
						mean=mean, input_scale=input_scale, raw_scale=raw_scale,
						channel_swap=channel_swap)

	def _preprocess_images(self, inputs):
		input_ = np.zeros((len(inputs), self.image_dims[0], self.image_dims[1], inputs[0].shape[2]), dtype=np.float32)
		# resize images
		for ix, in_ in enumerate(inputs):
			input_[ix] = caffe.io.resize_image(in_, self.image_dims)
			
		# crop them to 227 x 227
		center = np.array(self.image_dims) / 2.0
		crop = np.tile(center, (1,2))[0] + np.concatenate([-self.crop_dims / 2.0, self.crop_dims / 2.0])
		input_ = input_[:, crop[0]:crop[2], crop[1]:crop[3], :]
		
		# feature extraction
		caffe_in = np.zeros(np.array(input_.shape)[[0,3,1,2]], dtype=np.float32)
		for ix, in_ in enumerate(input_):
			caffe_in[ix] = self.transformer.preprocess(self.inputs[0], in_)
		return caffe_in
		
	
	def compute_featvecs(self, inputs, layer):
		caffe_in = self._preprocess_images(inputs)
		out = self.forward_all(data=caffe_in, blobs=[layer])[layer]
		# there are just two types of blobs, four-dimensional and two-dimensional (fc)
		if len(out.shape) == 2:
			return out.reshape((out.shape[0], out.shape[1]))
		else:
			return out.reshape((out.shape[0], out.shape[1]*out.shape[2]*out.shape[3]))
		
	def compute_compound_featvecs(self, inputs, layers):
		caffe_in = self._preprocess_images(inputs)
		out = self.forward_all(data=caffe_in, blobs=layers)
		f = []
		for layer in layers:
			resp = out[layer]
			f.append(resp.reshape((resp.shape[0], resp.shape[1])))
		return np.concatenate(f, axis=1)
		

def caffe_batch_predict(network_proto, network_weights, mean_protofile, imagelist_file, outfile, batch_size=100, top=5):
	# load learned weights
	print 'Loading network weights...'
	if not os.path.isfile(mean_protofile):
		raise ValueError('mean file not found!')
	print 'Converting mean protofile into numpy format...'
	blob = caffe.proto.caffe_pb2.BlobProto()
	data = open(mean_protofile, 'rb').read()
	blob.ParseFromString(data)
	arr = np.array(caffe.io.blobproto_to_array(blob))[0]
#	caffe.set_mode_gpu()
	net = Extractor(network_proto, network_weights, mean=arr.mean(1).mean(1), channel_swap=(2,1,0), raw_scale=255, image_dims=(256,256))

	# verify again image list in order to make sure they just contain valid image format
	print 'Loading images and their labels...'
	start_ix = 0
	stop_ix = start_ix + batch_size

	# load the imagelist and labelist
	imagelist = []
	with open(imagelist_file, 'rt') as fin:
		for line in fin:
			fpath = line.strip()
			imagelist.append(fpath)
	print 'Total ', len(imagelist), ' images are loaded'

	if batch_size == -1:
		batch_size = len(imagelist)
	
	# open file for writing prediction results
	fout = open(outfile, 'wt')
	while True:
		images_data = []
		for img in imagelist[start_ix:stop_ix]:
			if os.path.isfile(img):
				images_data.append(caffe.io.load_image(img))
			else:
				continue
		print '... a batch of ', len(images_data), 'images were loaded'
		tic = time.time()
		# start extraction
		print 'extracting features...'
		Y = net.predict(images_data, oversample=False)
		for y in Y:
			y = np.argsort(y)[-top:]
			y = y[::-1]
			fout.write(','.join([str(lbl_ix) for lbl_ix in y.tolist()]) + '\n')
		toc = time.time()
		print '...elapsed time ', (toc-tic)/batch_size, 'secs per image'
	
		# batch incremental
		start_ix = stop_ix
		stop_ix += batch_size
		if start_ix >= len(imagelist):
			break
	fout.close()

def ndarray2binaryproto(array, mean_protofile):
	if array.ndim != 3:
		raise InputError('The input array must be three-dimensional.')

	arr = np.ndarray((1,array.shape[0], array.shape[1], array.shape[2]))
	arr[0] = array
	blob = caffe.io.array_to_blobproto(arr)
	with open(mean_protofile, 'wb') as fout:
		fout.write(blob.SerializeToString(blob))
		
def binaryproto2array(mean_protofile):
	blob = caffe.proto.caffe_pb2.BlobProto()
	data = open(mean_protofile, 'rb').read()
	blob.ParseFromString(data)
	arr = np.array(caffe.io.blobproto_to_array(blob))[0]
	return arr
	
def caffe_set_device(gpu=True, devid='0'):
	if gpu:
		caffe.set_mode_gpu()	
		os.environ["CUDA_VISIBLE_DEVICES"] = devid
		caffe.set_device(int(devid))
	else:
		caffe.set_mode_cpu()
		
def caffe_load_images(imagelist):
	return [caffe.io.load_image(img) for img in imagelist]

# Work for moderate-size dataset because the function stores features vectors in memory
# and write them down disk at once.
def caffe_batch_extract_features(network_proto, network_weights, mean_protofile, imagelist_file, outfile, blob_names=['fc7'], batch_size=100, use_gpu=True, cuda_dev=0):
	# load learned weights
	if not os.path.isfile(mean_protofile):
		raise ValueError('mean file not found!')
		
	if os.path.isfile(outfile):
		print 'file exist. exit.'
		return
	
	if not mean_protofile.split('.')[-1] == 'npy':
		print 'Converting mean protofile into numpy format...'
		blob = caffe.proto.caffe_pb2.BlobProto()
		data = open(mean_protofile, 'rb').read()
		blob.ParseFromString(data)
		arr = np.array(caffe.io.blobproto_to_array(blob))[0]
		np.save(os.path.join(os.path.dirname(mean_protofile), os.path.basename(mean_protofile).split('.')[0] + '.npy'), arr)
	else:
		print 'Loading mean file...'
		arr = np.load(mean_protofile)
	
	net = Extractor(network_proto, network_weights, mean=arr.mean(1).mean(1), raw_scale=255, channel_swap=(2,1,0), image_dims=(256,256))
	# verify again image list in order to make sure they just contain valid image format
	print 'Extracting features from listing file ', imagelist_file, '...'
	start_ix = 0
	stop_ix = start_ix + batch_size

	# load the imagelist and labelist
	imagelist = []
	with open(imagelist_file, 'rt') as fin:
		for line in fin:
			fpath = line.strip().split(' ')
			fpath = fpath[0]
			imagelist.append(fpath)
	print 'Total ', len(imagelist), ' images are enlisted'

	if batch_size == -1:
		batch_size = len(imagelist)

	while True:
		images_data = []
		for img in imagelist[start_ix:stop_ix]:
			if os.path.isfile(img):
				try:
					images_data.append(caffe.io.load_image(img))
				except:
					print 'Warning: unknown/bad format file'
			else:
				raise ValueError('Image file(s) not found: ' + img)
		print '... a batch of ', len(images_data), 'images were loaded'
#		stop_ix = len(images_data)
		tic = time.time()
		# start extraction
#		print 'extracting features...'
		if len(blob_names) == 1:
			x = net.compute_featvecs(images_data, blob_names[0])
		else:
			x = net.compute_compound_featvecs(images_data, blob_names)
#		x = x.reshape((x.shape[0], x.shape[1]))
		toc = time.time()
		print '...elapsed time ', (toc-tic)/batch_size, 'secs per image'

#		print 'Writing feature to file...'
		dump_svmlight_file(x, np.zeros((x.shape[0], 1), dtype=np.int32), outfile, do_append=True)

		# batch incremental
		start_ix = stop_ix
		stop_ix += batch_size
		if start_ix >= len(imagelist):
			break

	print 'DONE.'

def caffe_batch_extract_predictionmap(network_proto, dense_network_proto, network_weights, mean_protofile, imagelist, outfile, src_layers, batch_size=100, dst_layers=['fc6-conv', 'fc7-conv', 'fc8-conv']):
	caffe.set_mode_cpu()	
	
	# load learned weights
	print 'Loading network weights...'
	net = DenseNet(network_proto, dense_network_proto, network_weights, mean_protofile, src_layers=src_layers, dst_layers=dst_layers)

	# verify again image list in order to make sure they just contain valid image format
	print 'Loading images and their labels...'
	start_ix = 0
	stop_ix = start_ix + batch_size

	if batch_size == -1:
		batch_size = len(imagelist)

	X = None
	first_time = True
	while True:
		images_data = []
		for img in imagelist[start_ix:stop_ix]:
			if os.path.isfile(img):
				images_data.append(caffe.io.load_image(img))
			else:
				continue
		print '... a batch of ', len(images_data), 'images were loaded'
#		stop_ix = len(images_data)
		tic = time.time()
		# start extraction
		print 'extracting features...'
		x = net.predict_densemap(images_data)
		toc = time.time()
		print '...elapsed time ', (toc-tic)/batch_size, 'secs per image'
	
		if first_time:
			X = x
			first_time = False
		else:
			X = np.r_[X, x]	
		# batch incremental
		start_ix = stop_ix
		stop_ix += batch_size
		if start_ix >= len(imagelist):
			break

	print 'Writing feature to file...'
	dump_svmlight_file(X, np.zeros((len(imagelist),1)), outfile)
	print 'DONE.'

def caffe_get_data_chunk(lmdb_file, chunk_size):
	lmdb_env = lmdb.open(lmdb_file)
	lmdb_txn = lmdb_env.begin()
	lmdb_cursor = lmdb_txn.cursor()
	datum = caffe_pb2.Datum()
	data = []
	i = 0
	for key, value in lmdb_cursor:
		if i == chunk_size:
			break
		datum.ParseFromString(value)
		label = datum.label
		data.append(caffe.io.datum_to_array(datum).ravel())
		i += 1
	return np.array(data, dtype=np.float32)

def caffe_lmdb2csr(lmdb_file, gt_file, out_file):
	imgs = []
	lbls = []
	with open(gt_file, 'rt') as fin:
		for line in fin:
			try:
				img, lbl = line.strip().split(' ')
			except ValueError, e:
				print(e)
				print(line)
				raise
			imgs.append(img)
			lbls.append(lbl)
	X = caffe_get_data_chunk(lmdb_file, len(imgs))
	if X.shape[0] != len(imgs):
#		print 'Length mismatch between ', gt_file, ' and ', lmdb_file
#		print ' ', X.shape[0], ' vs ', len(imgs)
		raise ValueError('Length mismatch between ' + gt_file, ' and ' + lmdb_file)
                                                                 
	dump_svmlight_file(X, np.array(lbls), out_file)
                                                                        
