import numpy as np
from torch.utils.data import Dataset
import h5py
from image_transforms import ImageTransforms
from tqdm import tqdm
from PIL import Image

class OCT_Dataset(Dataset):
    def __init__(self, ind_set:list, label_data:np.ndarray, all_files_paths, for_train, preload, dataset_no):
        self.for_train = for_train
        self.preload = preload
        self.length = len(ind_set)
        self.dataset_no = dataset_no

        self.chosen_file_paths = [all_files_paths[i] for i in ind_set]
        
        self.chosen_labels = label_data[ind_set].astype(np.float32)

        self.chosen_images = []
        if self.preload:
            for path in tqdm(self.chosen_file_paths, total=len(ind_set), ncols=75, desc="Preloading files..."):
                self.chosen_images.append(self._load_image(path))

    def _load_image(self, image_file_path):
        """Load and preprocess an image from a given path."""
        if image_file_path.endswith('.jpeg') or image_file_path.endswith('.jpg'):
            with Image.open(image_file_path) as img:
                img = img.convert('L')  # Convert to grayscale if not already
                image = np.array(img)
        else:
            with h5py.File(image_file_path, 'r') as h5_file:
                image = np.squeeze(h5_file['raw'][:])
        assert image.ndim == 2, "Image does not have only two dimensions"
        return image

    def __len__(self):
        return self.length
    
    def __getitem__(self, idx):        
        # Get the input data
        if self.preload: #TODO not working for jpg yet
            image = self.chosen_images[idx]
        else:
            image = self._load_image(self.chosen_file_paths[idx])

        image_tensor = ImageTransforms.transform_image(image, self.for_train, self.dataset_no)
        
        label = self.chosen_labels[idx]
        
        return image_tensor, label
