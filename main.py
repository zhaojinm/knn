import matplotlib.image as mpimg
import numpy as np
import matplotlib.pyplot as plt
import scipy.misc


def find_closest_pix(cur_pix,color_set):
	min_dis = 255*255*3
	best = color_set[0]
	for color in color_set:
		distance = (cur_pix[0]-color[0])**2 + (cur_pix[1]-color[1])**2 + (cur_pix[2]-color[2])**2
		if min_dis>distance:
			min_dis = distance
			best = color
	return best

def modify_init_pic(name,save_name):
	img = mpimg.imread(name)

	base_color = [[0,0,0],[41,36,33],[192,192,192],[128,138,135],[112,128,105],
			  	  [128,128,105],[225,225,225],[250,235,215],[240,255,255],[245,245,245],
			  	  [255,235,205],[255,248,220],[252,230,201],[255,250,240],[220,220,220],
			  	  [248,248,255],[240,255,240],[250,240,230],[255,222,173],[253,245,230]]
	new_img = []
	for i in range(img.shape[0]):
		row = []
		if i%100 == 0:
			print(i)
		for j in range(img.shape[1]):
			cur_pix = img[i][j]
			new_pix = find_closest_pix(cur_pix,base_color)
			row.append(new_pix)
		new_img.append(row)
	new_img = np.array(new_img,dtype = np.uint8)
	scipy.misc.imsave(save_name,new_img)
	print(new_img.shape)
	print(new_img[0][0])
	plt.imshow(new_img)
	plt.show()


class KNN:
	def __init__(self,k,data):
		self.k = k
		self.data = data

	def get_new_img(self):
		new_img = []
		for i in range(self.data.shape[0]):
			row = []
			if i%100 == 0:
				print(i)
			for j in range(self.data.shape[1]):
				neighbors = self.get_k_nearest(i,j)
				u, indices = np.unique(neighbors, return_inverse=True)
				most_common = u[np.argmax(np.apply_along_axis(np.bincount,0, indices.reshape(neighbors.shape),None, np.max(indices) + 1), axis=0)].tolist()
				row.append(most_common)
			new_img.append(row)
		new_img = np.array(new_img,dtype = np.uint8)
		print(new_img.shape)
		print(new_img[0][0])
		return new_img

	def get_k_nearest(self,row,col):
		result = self.data[row][col]
		for i in range(max(0,row-self.k),min(row+self.k,self.data.shape[0])):
			for j in range(max(0,col-self.k),min(col+self.k,self.data.shape[1])):
				if i!=row and j!=col:
					result = np.vstack((result,self.data[i][j]))
		return result

if __name__ == '__main__':
	#modify_init_pic("init.jpg","3.jpg")
	img = mpimg.imread('3.jpg')
	k = 4
	knn = KNN(k,img)
	new_img = knn.get_new_img()
	plt.imshow(new_img)
	plt.show()


