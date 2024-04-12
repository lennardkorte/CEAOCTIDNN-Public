
import cv2 as cv
import scipy
import scipy.interpolate
from scipy.ndimage import gaussian_filter
from scipy import ndimage
import numpy as np
import polarTransform as pt
import torch
from torchvision import transforms as T
from torchvision.transforms.functional import InterpolationMode
from scipy import signal
from random import randrange
from guide_wire import guide_wire_literal
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter

class IdentityTransform(object):
    def __init__(self):
        pass
    
    def __call__(self, image):
        return image

'''
Pre-processing:
'''
class GuideWireRemoval(object):
    def __init__(self):
        pass
    
    def __call__(self, image_orig):
        image = np.copy(image_orig)
        # Cutt off the first rows to blend out the catheter reflections
        cuttoff = 80
        image[0:cuttoff, :] = 0
        column_sums = image.sum(axis=0)
        width = 30
        column_group_min = float('inf')
        for i in range(0, image.shape[1]):
            column_group = 0
            for j in range(0, width+1):
                col_num = (i + j) % image.shape[1]
                column_group = column_group + column_sums[col_num]
            if column_group < column_group_min:
                column_group_min = column_group
                column_group_min_ind = i
        
        for column in range(0, image.shape[1]):
            if column_group_min_ind+width <= image.shape[1]:
                if column_group_min_ind <= column < (column_group_min_ind+width):
                    image_orig[:,column] = 0
            else:
                if column >= column_group_min_ind or column < (column_group_min_ind+width) % image.shape[1]:
                    image_orig[:,column] = 0
        
        return image_orig
    
class CartToPolar(object):
    '''
    Input: 
    Output:
    '''
    def __init__(self, radius=112):
        self.radius = radius
        pass
    
    @staticmethod
    def to_cart(image:np.ndarray, shape_desired:tuple):
        shape_original = image.shape
        image = image.astype(float)
        
        # Polarcoordinates assuming uniform rotation
        lin_space1 = np.linspace(0, 2*np.pi, shape_original[1])
        lin_space2 = np.arange(1, shape_original[0]+1)
        theta,rho = np.meshgrid(lin_space1, lin_space2)
        
        # Cartesian coordinates of desired image
        lin_space1 = np.linspace(-shape_original[0],shape_original[0],shape_desired[1])
        lin_space2 = np.linspace(-shape_original[0],shape_original[0],shape_desired[0])
        x1, x2 = np.meshgrid(lin_space1, lin_space2)
        
        # Transform to polar and interpolate
        theta2 = np.arctan2(x2, x1)
        rho2 = np.sqrt(x1**2 + x2**2)

        image_polar = scipy.interpolate.griddata((theta.ravel(),rho.ravel()),image.ravel(),((theta2+np.pi).ravel(),rho2.ravel()), fill_value=0.0, method='nearest', rescale=True)
        image_polar = np.resize(image_polar, shape_desired)
        
        image_polar = image_polar.astype(np.uint16)
        
        return image_polar
    
    def __call__(self, image_orig):
        # personal version:
        #image = self.to_cart(image=image, shape_desired=(300, 300))
        
        # more multithread speedup version
        image = image_orig.transpose()
        image, _ = pt.convertToCartesianImage(image, finalRadius=300, border='constant', order=2)
        
        return image

class MeanNormalization(object):
    def __init__(self):
        pass
    
    def __call__(self, image:torch.Tensor):
        numerator_right = torch.mean(image)
        denominator = torch.max(image) - torch.min(image)
        # Normalize: (data - numerator_right) / denominator
        transformation = T.Normalize(numerator_right, denominator)
        image = transformation(image)
        return image
    
class Standardization_zero(object):
    def __init__(self):
        pass
    
    def __call__(self, image:torch.Tensor):
        numerator_right = torch.mean(image)
        denominator = torch.std(image)
        # Normalize: (data - numerator_right) / denominator
        transformation = T.Normalize(numerator_right, denominator)
        image = transformation(image)
        return image
    
class Standardization_zero_five(object):
    def __init__(self):
        pass
    
    def __call__(self, image:torch.Tensor):
        numerator_right = torch.mean(image) - 0.5
        denominator = torch.std(image) * 4
        # Normalize: (data - numerator_right) / denominator
        transformation = T.Normalize(numerator_right, denominator)
        image = transformation(image)
        return image
    
class Standardization_IN(object):
    def __init__(self):
        pass
    
    def __call__(self, image:torch.Tensor):
        # [0.485, 0.456, 0.406]
        # [0.229, 0.224, 0.225]
        numerator_right = torch.mean(image[0,:,:]) - 0.485
        denominator = torch.std(image[0,:,:]) * (1.0/0.229)
        # Normalize: (data - numerator_right) / denominator
        transformation = T.Normalize(numerator_right, denominator)
        image1 = transformation(torch.unsqueeze(image[0,:,:],0))
        
        numerator_right = torch.mean(image[1,:,:]) - 0.456
        denominator = torch.std(image[1,:,:]) * (1.0/0.224)
        # Normalize: (data - numerator_right) / denominator
        transformation = T.Normalize(numerator_right, denominator)
        image2 = transformation(torch.unsqueeze(image[1,:,:],0))
        
        numerator_right = torch.mean(image[2,:,:]) - 0.406
        denominator = torch.std(image[2,:,:]) * (1.0/0.225)
        # Normalize: (data - numerator_right) / denominator
        transformation = T.Normalize(numerator_right, denominator)
        image3 = transformation(torch.unsqueeze(image[2,:,:],0))
        
        image_return = torch.cat((image1, image2, image3), 0)
        return image_return

class Rescaling(object):
    def __init__(self):
        pass
    
    def __call__(self, image:torch.Tensor):
        numerator_right = torch.min(image)
        denominator = torch.max(image) - torch.min(image)
        # Normalize: (data - numerator_right) / denominator
        transformation = T.Normalize(numerator_right, denominator)
        image = transformation(image)
        return image

class ToFloat(object):
    def __init__(self, p=0.5):
        pass
    
    def __call__(self, image:np.array):
        return image.astype(np.float32)

class AddDoubleZeroPadding(object):
    def __init__(self):
        pass
    
    def __call__(self, img:torch.Tensor):
        dimension_of_img = img.shape
        double_padding = torch.zeros((2,) + tuple(dimension_of_img)[1:])
        img_return = torch.cat((img, double_padding), 0)
        #print(img_return.shape)
        return img_return
    
class ThreeChannelCopy(object):
    def __init__(self):
        pass
    
    def __call__(self, img:torch.Tensor):
        img_return = torch.cat((img, img, img), 0)
        #print(img_return.shape)
        return img_return

'''
DA
--------------------------------------------
'''

class RemoveGuidewire(object):
    def __init__(self):
        pass
    
    def __call__(self, image_orig):
        # Make a deep copy of the original image
        image_cutoff = np.copy(image_orig)
        
        # Cutt off the first rows to blend out the catheter reflections
        cuttoff = 85
        image_cutoff[0:cuttoff, :] = 0
        
        # Sum all vallues of each column
        column_sums = image_cutoff.sum(axis=0)
        width = 30
        
        # Find the group of columns, that has the lowest overall sum with width of 25 pixels
        column_group_min = float('inf')
        for i in range(0, image_cutoff.shape[1]):
            column_group = 0
            for j in range(0, width+1):
                col_num = (i + j) % image_cutoff.shape[1]
                column_group = column_group + column_sums[col_num]
            if column_group < column_group_min:
                column_group_min = column_group
                column_group_min_ind = i
                
        image_return = np.copy(image_orig)
        for column in range(0, image_cutoff.shape[1]):
            # Check if the guide wire lies in this column.
            if column_group_min_ind+width <= image_cutoff.shape[1]:
                if column_group_min_ind <= column < (column_group_min_ind+width):
                    image_return[:,column] = 0
            else:
                if column >= column_group_min_ind or column < (column_group_min_ind+width) % image_cutoff.shape[1]:
                    image_return[:,column] = 0
                
        return image_return

class RandomPosterize(object):
    def __init__(self, bits, p=0.5):
        self.bits = bits
        self.p = p
        
    def __call__(self, image:torch.Tensor):
        datatype_original = image.dtype
        datatype = torch.int16
        scalar = torch.iinfo(datatype).max
        image = torch.mul(image, scalar)
        image = image.to(datatype)
        #image = T.RandomPosterize(self.bits, self.p)(image)
        image = image.to(datatype_original)
        image = torch.div(image, scalar)
        return image
    
'''
DA New
--------------------------------------------
'''
class MoveCurve(object):
    
    @staticmethod
    def move_curve(image_orig):
        # Make a deep copy of the original image
        image_wave = np.copy(image_orig)
        
        # Cutt off the first rows to blend out the catheter reflections
        cuttoff = 75
        image_wave[0:cuttoff, :] = 0
        
        # Sum all vallues of each column
        column_sums = image_wave.sum(axis=0)
        width = 25
        
        # Find the group of columns, that has the lowest overall sum with width of 25 pixels
        column_group_min = float('inf')
        for i in range(0, image_wave.shape[1]):
            column_group = 0
            for j in range(0, width+1):
                col_num = (i + j) % image_wave.shape[1]
                column_group = column_group + column_sums[col_num]
            if column_group < column_group_min:
                column_group_min = column_group
                column_group_min_ind = i
        
        # extract the curve to investigate its result
        image_wave = cv.GaussianBlur(image_wave, (21,21), 15)
        
        # Apply threshold to create binary black and white image
        tre, image_wave = cv.threshold(image_wave, 450, 65535, cv.THRESH_BINARY)
        image_wave = image_wave.astype(np.uint8)
        
        # Finds all your connected components
        (numLabels, labels, stats, centroids) = cv.connectedComponentsWithStats(image_wave, connectivity=4)
        # The following part is just taking out the background which is also considered a component
        sizes = stats[1:, -1]; numLabels = numLabels - 1
        # Only Keep objects with certain size
        min_size = 1700
        image_clean = np.zeros((image_wave.shape))
        for i in range(0, numLabels):
            if sizes[i] >= min_size:
                image_clean[labels == i + 1] = 255
        image_wave = image_clean
        
        '''
        # https://opencv24-python-tutorials.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_morphological_ops/py_morphological_ops.html
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11,11))
        morph = cv2.morphologyEx(image_wave, cv2.MORPH_CLOSE, kernel, iterations=1)
        image_wave = cv2.morphologyEx(morph, cv2.MORPH_OPEN, kernel, iterations=1)
        '''
        
        # Scan the wave from the top to get its highest value of the curve
        columns_heights = [0] * image_wave.shape[1]
        for column in range(0, image_wave.shape[1]):
            out_of_shadow = True
            # Check if the guide wire lies in this column.
            if column_group_min_ind+width <= image_wave.shape[1]:
                if column_group_min_ind <= column < (column_group_min_ind+width):
                    columns_heights[column] = 0
                    image_wave[100,column] = 255
                    out_of_shadow = False
            else:
                if column >= column_group_min_ind or column < (column_group_min_ind+width) % image_wave.shape[1]:
                    columns_heights[column] = 0
                    image_wave[100,column] = 255
                    out_of_shadow = False
            # If guide wire not in this colum, then check height of the wave.
            if out_of_shadow:
                for row in range(0, image_wave.shape[0]):
                    if image_wave[row, column] > 0:
                        columns_heights[column] = row
                        break
                    if row == image_wave.shape[0]-1:
                        columns_heights[column] = row
        
        # Find the longest range, that does not include a guide wire and where the curve has enough distance.
        thresh1 = 110 # Maximum height of wave allowed to be modified
        min_max_best = [None, None]
        min_max_current = [None, None]
        for i in range(0, 2*image_wave.shape[1]):
            if columns_heights[i%image_wave.shape[1]] > thresh1:
                if min_max_current[0] is None:
                    min_max_current[0] = i
            else:
                if min_max_current[0] is not None:
                    min_max_current[1] = i-1
                    if min_max_best[0] is not None:
                        if min_max_current[1] - min_max_current[0] > min_max_best[1] - min_max_best[0]:
                            min_max_best = min_max_current
                    else:
                        min_max_best = min_max_current
                    min_max_current = [None, None]
        
        # Draw a line to indicate the allowed range with the hann-window overlay
        '''
        for i in range(min_max_best[0], min_max_best[1]+1):
            image_wave[10, i%image_wave.shape[1]] = 65535
        '''
        # Create deep copy of original image to return
        image_out = np.copy(image_orig)
        '''
        Find possible scalars for Hann-Function
        The new wave musst follow the rules:
            - The scalar for the hann function is selected at random
            - Scalars musst be between 1 and 100
            - The added hann Window is only allowed to change the wave at a maximum of half of the distance between thresh1 and current wave height (columns_heights)
            - The scalar is allowed to have positive and negative values
        '''
        if min_max_best[1] is None or min_max_best[0] is None:
            return image_orig
        
        # Create Hann Window with size calculated above
        hann_window = signal.hann(min_max_best[1] - min_max_best[0] + 1)
        # Get wave form / height in size calculated above
        if min_max_best[1] < image_wave.shape[1]:
            columns_heights_window = columns_heights[min_max_best[0]:min_max_best[1]+1]
        else:
            columns_heights_window = columns_heights[min_max_best[0]:] + columns_heights[:(min_max_best[1]%image_wave.shape[1])+1] 
        # Calculate maximum scalar for hann window
        scalar_max_possible = 1
        for scalar in range(1, 1001): 
            hann_window_scaled = hann_window * scalar
            hann_window_scaled = np.round(hann_window_scaled).astype(int)
            difference = (np.array(columns_heights_window) - thresh1) - hann_window_scaled
            if difference.min() > 0:
                scalar_max_possible = scalar
            else:
                break
        # Create random scalar based on maximum calculated above
        scalar_random = randrange(0, int((scalar_max_possible+1) * (2/3)))
        hann_window_scaled = hann_window * scalar_random
        hann_window_scaled = np.round(hann_window_scaled).astype(int)
        
        '''
        We add the hann window multiplied by the randomized scalar to the curve and move the columns accordingly
        '''
        index = 0
        thresh2 = 100 # Defines which parts are allowed to be moved
        for column in range(min_max_best[0], min_max_best[1]+1):
            image_column = column % image_wave.shape[1]
            hann_mult_local = hann_window_scaled[index]
            index = index + 1
            # If hann scalar is positive, padd the column with mirrored column at the bottom
            if hann_mult_local > 0:
                first_part = image_orig[thresh2+hann_mult_local:image_wave.shape[0]+1, image_column]
                second_part = image_orig[::-1, image_column][:hann_mult_local]
                image_out[thresh2:, image_column] = np.concatenate((first_part, second_part))
            # If hann scalar is positive, padd the column with mirrored column at the top (below threshold)
            elif hann_mult_local < 0:
                first_part = image_orig[thresh2+1:thresh2+1-hann_mult_local, image_column][::-1]
                second_part = image_orig[thresh2+1:image_wave.shape[0]+hann_mult_local, image_column]
                image_out[thresh2+1:, image_column] = np.concatenate((first_part, second_part))

        return image_out

    def __init__(self):
        pass
    
    def __call__(self, image):
        image = self.move_curve(image)
        return image
    
class RandomGuideWire(object):
    def __init__(self):
        pass
    
    def __call__(self, image_orig):
        # Make a deep copy of the original image
        image_wave = np.copy(image_orig)
        # Make a deep copy of the original image
        image_return = np.copy(image_orig)
        
        # Cutt off the first rows to blend out the catheter reflections
        cuttoff = 75
        image_wave[0:cuttoff, :] = 0
        
        # extract the curve to investigate its result
        image_wave = cv.GaussianBlur(image_wave, (21,21), 15)
        
        # Apply threshold to create binary black and white image
        tre, image_wave = cv.threshold(image_wave, 450, 65535, cv.THRESH_BINARY)
        image_wave = image_wave.astype(np.uint8)
        
        # Finds all your connected components
        (numLabels, labels, stats, centroids) = cv.connectedComponentsWithStats(image_wave, connectivity=4)
        # The following part is just taking out the background which is also considered a component
        sizes = stats[1:, -1]; numLabels = numLabels - 1
        # Only Keep objects with certain size
        min_size = 1700
        image_clean = np.zeros((image_wave.shape))
        for i in range(0, numLabels):
            if sizes[i] >= min_size:
                image_clean[labels == i + 1] = 255
        image_wave = image_clean
        
        # Scan the wave from the top to get its highest value of the wave
        columns_heights = [0] * image_wave.shape[1]
        for column in range(0, image_wave.shape[1]):
            for row in range(0, image_wave.shape[0]):
                if image_wave[row, column] > 0:
                    columns_heights[column] = row
                    break
                if row == image_wave.shape[0]-1:
                    columns_heights[column] = row
        
        
        if list(columns_heights) == []:
            print('no possible value')
        else:
            col_w_indices = list(enumerate(list(columns_heights)))
            col_without_curve = [t[0] for t in col_w_indices if t[1] >= 70]
            rand_ind = np.random.choice(col_without_curve)
            rand_col = (col_w_indices[rand_ind])[1]
            
            if rand_col <= 319:
                guide_wire_array = np.frombuffer(guide_wire_literal, dtype=np.uint16).reshape((623, 28))
                #x = randrange(0, 319)
                x = rand_col
                y = randrange(0, 20)
                image_return[60+y:, x:28+x] = (guide_wire_array[:623-y]  * 0.7).astype(np.uint16)
        
        if torch.rand(1) <= 0.5:
            return image_return
        else:
            return image_orig

class WhiteColumnArtefacts(object):
    def __init__(self, num_arrays = 10):
        self.num_arrays = num_arrays
        pass
    
    def __call__(self, image_orig):
        da_arrays = []
        
        for i in range(0, self.num_arrays):
            width1 = randrange(180, 221)
            hann_window1 = signal.hann(width1)
            rand_position1 = randrange(0, image_orig.shape[0]-width1)
            mult_vector1 = [0] * image_orig.shape[0]
            mult_vector1[rand_position1:rand_position1+width1] = hann_window1
            mult_array = np.tile(np.array([mult_vector1]).transpose(), (1, image_orig.shape[1]))
            
            width2 = randrange(40,61)
            hann_window2 = signal.hann(width2)
            rand_position2 = randrange(0, image_orig.shape[1]-width2)
            mult_vector2 = [0] * image_orig.shape[1]
            mult_vector2[rand_position2:rand_position2+width2] = hann_window2
            
            scalar = 5
            random_noise = np.random.rand(1, image_orig.shape[1])
            mult_array_noise = mult_array * random_noise
            mult_array_scaled = mult_array_noise * mult_vector2 * scalar
            da_arrays.append(image_orig * mult_array_scaled.astype(np.uint16))
        
        for i in range(0, self.num_arrays):
            image_return = image_orig + da_arrays[i]
        
        if torch.rand(1) <= 0.5:
            return image_orig
        else:
            return image_return
        

class BloodArtefacts(object):
    def __init__(self):
        pass
    
    def __call__(self, image_orig):
        n = 188
        l = 1024
        im = np.zeros((l, l))
        points = l*np.random.random((2, n**2))
        im[(points[0]).astype(int), (points[1]).astype(int)] = 1
        im = ndimage.gaussian_filter(im, sigma=l/(4.*n))
        mask = im > im.mean()
        
        label_im, nb_labels = ndimage.label(mask)
        nb_labels # how many regions?
        
        sizes = ndimage.sum(mask, label_im, range(nb_labels + 1))
        mean_vals = ndimage.sum(im, label_im, range(1, nb_labels + 1))
        
        mask_size = sizes < 1000
        remove_pixel = mask_size[label_im]
        remove_pixel.shape

        label_im[remove_pixel] = 0
        label_im = label_im.astype(np.float32)
        
        test, label_im = cv.threshold(label_im, 400, 65535, cv.THRESH_BINARY, 1)
        label_im = cv.resize(label_im, (319,347), interpolation = cv.INTER_AREA)
        label_im = gaussian_filter(label_im, sigma=1, output=np.uint16)
        result = (label_im * 0.02).astype(np.uint16)
        result = np.transpose(result)
        
        image_orig[:319] = image_orig[:319] + result
        result = (image_orig).astype(np.uint16)
        if torch.rand(1) <= 0.5:
            return image_orig
        else:
            return result

class AddGaussianNoise2(object):
    def __init__(self):
        pass
        
    def __call__(self, image_orig):
        # Add noise to image
        image = image_orig
        mean = 0
        stddev = 10000
        noise = np.zeros_like(image)
        cv.randn(noise, mean, stddev)
        noisy_img = cv.add(image, noise)

        return noisy_img


class CLAHE(object):
    def __init__(self):
        pass
        
    def __call__(self, image_orig):
        image = image_orig
        #image = cv.equalizeHist(image)
        
        #clahe = cv.createCLAHE(clipLimit=40.0, tileGridSize=(60,60))
        #image = clahe.apply(image)
        tile_size = 5
        bordered_image = cv.copyMakeBorder(image, tile_size, tile_size, tile_size, tile_size, cv.BORDER_CONSTANT, value=0)
        
        clahe = cv.createCLAHE(clipLimit=65535, tileGridSize=(tile_size,tile_size))
        clahe_image = clahe.apply(bordered_image)
        
        height, width = image.shape
        cropped_image = clahe_image[tile_size:height+tile_size, tile_size:width+tile_size]
        '''
        mask = np.zeros_like(cropped_image)
        cv.circle(mask, (width//2, height//2), min(width, height)//2, (65535), thickness=-1)
        final_image = cv.bitwise_and(cropped_image, mask)
        '''

        return cropped_image
    
    
class RandomShiftHor(object):
    def __init__(self, max_amount = 173):
        self.max_amount = max_amount
        pass
        
    def __call__(self, image_orig):
        image = image_orig
        
        rand_num = randrange(-self.max_amount, self.max_amount)
        
        if rand_num > 0:
            image_return = np.concatenate((image[:,-rand_num:], image[:,:-rand_num]), axis=1)
        elif rand_num < 0:
            image_return = np.concatenate((image[:,-rand_num:], image[:,:-rand_num]), axis=1)
        else:
            image_return = image_orig
            
        #return image_orig
        return image_return
    
class RandomShiftVert(object):
    def __init__(self, max_amount = 341):
        self.max_amount = max_amount
        pass
        
    def __call__(self, image_orig):
        image = image_orig
        
        rand_num = randrange(-self.max_amount, self.max_amount)
        
        if rand_num > 0:
            image_return = np.concatenate((image[rand_num:,:], image[:rand_num,:]), axis=0)
        elif rand_num < 0:
            image_return = np.concatenate((image[rand_num:,:], image[:rand_num,:]), axis=0)
        else:
            image_return = image_orig
            
        return image_return
    
class HorizontalFlip(object):
    def __init__(self):
        pass
        
    def __call__(self, image_orig):
        image = image_orig
        
        if torch.rand(1) <= 0.5:
            image_return = np.fliplr(image)
        else:
            image_return = image
            
        return image_return
    
class VerticalFlip(object):
    def __init__(self):
        pass
        
    def __call__(self, image_orig):
        image = image_orig
        
        if torch.rand(1) <= 0.5:
            image_return = image[::-1]
        else:
            image_return = image
            
        return image_return
    
class Shearing(object):
    def __init__(self, max_amount = 100):
        self.max_amount = max_amount
        pass
        
    def __call__(self, image_orig):
        image = image_orig
        
        rand_num = int((self.max_amount / 2) * torch.rand(1))
        if rand_num != 0:
            image_return = image[:,rand_num:-rand_num]
        else:
            image_return = image
            
        return image_return
    
class Stretching(object):
    def __init__(self, max_amount = 200):
        self.max_amount = max_amount
        pass
        
    def __call__(self, image_orig):
        image = image_orig
        
        rand_num = int(self.max_amount * torch.rand(1))
        if rand_num != 0:
            image_return = image[:-rand_num,:]
        else:
            image_return = image
            
        return image_return
    
class Scaling(object):
    def __init__(self, max_amount = 50):
        self.max_amount = max_amount
        pass
        
    def __call__(self, image_orig):
        image = image_orig
        
        rand_num = int(self.max_amount - (0.6 * self.max_amount * torch.rand(1)))
        if rand_num != 0:
            rand_y = int(torch.rand(1) * (image_orig.shape[1] - rand_num))
            rand_x = int(torch.rand(1) * (image_orig.shape[0] - rand_num))
            image_return = np.copy(image_orig)
            image_return[rand_x:rand_x+rand_num, rand_y:rand_y+rand_num] = 0.0
        else:
            image_return = image
            
        return image_return
    
class RandomDist(object):
    def __init__(self):
        pass
        
    def __call__(self, image_orig):
        """Elastic deformation of images as described in [Simard2003]_.
        [Simard2003] Simard, Steinkraus and Platt, "Best Practices for
        Convolutional Neural Networks applied to Visual Document Analysis", in
        Proc. of the International Conference on Document Analysis and
        Recognition, 2003.
        """
        image = image_orig
        
        alpha = 500
        sigma = 25
        random_state = None
        
        assert len(image.shape)==2

        if random_state is None:
            random_state = np.random.RandomState(None)

        shape = image.shape

        dx = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha
        dy = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha

        x, y = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), indexing='ij')
        indices = np.reshape(x+dx, (-1, 1)), np.reshape(y+dy, (-1, 1))
        
        return map_coordinates(image, indices, order=1).reshape(shape)
    
class Davella(object):
    def __init__(self):
        pass
    
    def __call__(self, image):
        max = 1000.0
        a = torch.rand(image.shape).numpy() * 0.1 * max
        b = torch.rand(image.shape).numpy() * 0.1 * max
        c = torch.rand(image.shape).numpy() * 0.8 * max + 0.6 * max
        image = -a + (a + b + 1) * image ** c
            
        return image

class Hussain(object):
    def __init__(self):
        pass
    
    def __call__(self, image):
        n = np.random.normal(1, 0, image.shape)
        r = torch.rand(image.shape)
        image = image ** (n * r + 1)
                
        return image
    
class GaussianBlur(object):
    def __init__(self):
        pass
    
    def __call__(self, image):
        if torch.rand(1) <= 0.5:
            image = cv.GaussianBlur(image, (13,13), 2)
        return image
    
class AddGaussianNoise(object):
    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean
        
    def __call__(self, tensor):
        max_float16 = np.finfo(np.float16).max
        noisy_tensor = tensor + torch.randn(tensor.size()) * self.std + self.mean
        clamped_tensor = torch.clamp(noisy_tensor, min=-max_float16, max=max_float16)
        return clamped_tensor

class PartialMasking(object):
    def __init__(self):
        pass
    
    def __call__(self, image):
        if torch.rand(1) <= 0.8:
            size = 0.2
            x, y = image.shape[1], image.shape[2]  # Assuming the input is in the shape [1, 300, 300]
            x_s = int(x * size * (0.5 + torch.rand(1).item()))
            y_s = int(y * size * (0.5 + torch.rand(1).item()))

            x_p = int(torch.rand(1).item() * (x - x_s))
            y_p = int(torch.rand(1).item() * (y - y_s))
            
            image_return = image.clone()  # Clone the original tensor
            image_return[:, x_p:x_p + x_s, y_p:y_p + y_s] = 0
        else:
            image_return = image

        return image_return
    
class CircularMask(object):
    def __init__(self, radius=1.0, invert=False):
        self.mask = None
        self.shape = None
        self.radius = radius
        self.invert = invert
    
    def __call__(self, images):
        if len(images.shape) == 4:
            # If the input is a batch of images ([batch_size, channels, height, width])
            batches, channels, height, width = images.shape
        elif len(images.shape) == 3:
            # If the input is a single RGB image ([channels, height, width])
            channels, height, width = images.shape
        elif len(images.shape) == 2:
            # If the input is a single grayscale image ([height, width])
            height, width = images.shape
        else:
            raise ValueError("Unsupported image shape")

        self._create_grid(height, width, images.device)
        
        if len(images.shape) == 2:
            # For a single grayscale image, apply the mask directly
            return images * self.mask
        elif len(images.shape) in [3, 4]:
            # For RGB images (single or batch), expand the mask to match the dimensions
            expanded_mask = self.mask.unsqueeze(0)  # Add two dimensions: batch and channel
            if len(images.shape) == 4:
                expanded_mask = self.mask.unsqueeze(0)
                expanded_mask = expanded_mask.repeat(images.size(0), channels, 1, 1)  # Repeat for batch and channels
            else:
                expanded_mask = expanded_mask.repeat(channels, 1, 1)  # Repeat for channels only for single image
            return images * expanded_mask
    
    def _create_grid(self, height, width, device):
        if not self.mask or self.shape != (height, width):
            self.shape = (height, width)

            x = torch.linspace(-1, 1, steps=width)
            y = torch.linspace(-1, 1, steps=height)
            grid_x, grid_y = torch.meshgrid(x, y, indexing='ij')

            # Create the mask
            self.mask = ((grid_x ** 2 + grid_y ** 2) <= self.radius ** 2).to(device)

            # Invert the mask if the invert flag is True
            if self.invert:
                self.mask = ~self.mask

class MaxBrightnessToBlackTransform(object):
    def __init__(self):
        pass

    def __call__(self, img):
        """
        img: a PyTorch tensor of the image with shape (C, H, W) and values in [0, 1].
        """
        # Ensure img is a torch tensor
        if not isinstance(img, torch.Tensor):
            raise TypeError("Image must be a PyTorch tensor")

        # Find the maximum pixel value across the entire image tensor
        max_val = img.max()
        
        # Create a mask where the pixel value is equal to the max value
        mask = img == max_val

        # Set those pixels to black (0)
        img[mask] = 0
        
        return img

'''
Concatenation
--------------------------------------------
'''
class ImageTransforms():
    @classmethod
    def compose_transforms(cls, image:np, for_train:bool, dataset_no:int):
        
        #custom_transforms = [
            #IdentityTransform(),
            
            # Distorting filters
            #T.GaussianBlur(kernel_size=(5, 5), sigma=(0.1, 2.0)),
            
            # Padding
            # : randomize
            #T.Pad(padding=0, fill=0, padding_mode='constant'), # other padding modes: edge, reflect, symmetric
            
            # Cropping: Crops the given image at the center. 
            #T.CenterCrop(size=200),
            #T.RandomCrop(size=image.shape, padding=10, fill=0, padding_mode='constant'), # edge, reflect or symmetric
            

            # Positioning
            # -------------------------------------------
            # The following destroys determinism due to different worker seeds
            #T.RandomRotation(degrees=(-180,180), fill=0, interpolation=InterpolationMode.BILINEAR), # try center parameter random, interpolationmode has no effect on b/w image
            #T.RandomPerspective(distortion_scale=0.3, p=1.0, fill=0),
            #T.RandomHorizontalFlip(p=0.5),
            #T.RandomVerticalFlip(p=0.5),

            # Resizing, stretching, squeezing, repositioning
            #  :Randomize
            #T.Resize(size=(200,200)),
            # random crop, random aspect ratio, resize
            #T.RandomResizedCrop(size=image.shape, scale=(0.8, 1.0), ratio=(1.0, 1.0), antialias=True), # also used in Inception networks
            # random rotation, random scale, repositioning
            #T.RandomAffine(degrees=(-180,180), translate=(0, 0), fill=0, interpolation=InterpolationMode.BILINEAR), #, scale=(0.8, 1.2)
            
            
            # Grayscaling
            # -------------------------------------------
            #T.ColorJitter(brightness=1, contrast=0, saturation=0, hue=0), # saturation and hue have no effect on b/w image
            #T.RandomInvert(p=0.5),
            # Posterizing
            #RandomPosterize(bits=3, p=0.5),
            
            # Later maybe
            # GAN
            # RandomApply
            
            # No expected improvement for less data
            #T.RandomErase()
            
            # Not usable
            # Grayscale
            # RandomGrayscale
        #]
        da_before_pre_transform = [
            # DA New
            #MoveCurve(),
            #RandomGuideWire(),
            #WhiteColumnArtefacts(num_arrays = 10),
            #BloodArtefacts(),
            #BloodArtefacts(),
            #BloodArtefacts(),
                        
            # preprocessing
            #GuideWireRemoval(),
            
            #RandomShiftHor(max_amount=173),
            #RandomShiftVert(max_amount=341),
            
            #VerticalFlip(),
            #HorizontalFlip(),
            
            #RandomDist(),
            
            #Davella(),
            #Hussain(),
            #GaussianBlur(),
            
            #Shearing(max_amount=100),
            #Stretching(max_amount=200),
            #Scaling(max_amount=50),
            
            #RemoveGuidewire(),
            #PartialMasking(),
        ]
        pre_transforms = [
            #CLAHE(), #TODO
            #AddGaussianNoise2(),
            
            #CartToPolar(radius=112),
            
            # always
            ToFloat(),
            T.ToTensor(),

            # preprozessing
            #MeanNormalization(),
            #Standardization_zero(),
            #Standardization_zero_five(),
            Rescaling(), # TODO rescale only circle # note: if rescaling is removed, set a scaling factor in create_samples for better visibility
            
        ]
        post_transforms = [
            T.Resize(size=(224, 224), antialias=True),
            #AddGaussianNoise(0, 5000),
            #AddDoubleZeroPadding(),
            #CircularMask(), #TODO
            #CircularMask(0.17, True), #TODO
            ThreeChannelCopy(),
            #Standardization_IN(),
            
        ]

        da_techniques = []
        match dataset_no:
            case 1:
                if for_train: da_techniques = [
                    T.RandomAffine(degrees=(-180,180), translate=(0, 0), fill=0, interpolation=InterpolationMode.BILINEAR),
                    MaxBrightnessToBlackTransform(),
                    T.RandomHorizontalFlip(p=0.5),
                    T.RandomVerticalFlip(p=0.5),
                    PartialMasking(),
                ]
                all_transforms = pre_transforms + da_techniques + post_transforms
            case 2:
                if for_train: da_techniques = [
                    #MaxBrightnessToBlackTransform(),
                    T.RandomHorizontalFlip(p=0.5),
                ]
                all_transforms = pre_transforms + da_techniques + post_transforms

        composed_transforms = T.Compose(all_transforms)
        return composed_transforms
        
    @classmethod
    def transform_image(cls, image, for_train, dataset_no):
        composed_transforms = cls.compose_transforms(image, for_train, dataset_no)
        transformed_image = composed_transforms(image)
        return transformed_image
