
from torch.utils.data import DataLoader
from dataset import OCT_Dataset

class Dataloaders():
    training = None
    training_eval = None
    validation = None
    test = None

    @classmethod
    def _create_dataloader(cls, indices, cust_data, config, shuffle=False, drop_last=False, for_train=False, imgs_for=None):
        return DataLoader(
            OCT_Dataset(indices, cust_data.label_data, cust_data.all_files_paths, for_train, config['preload'], config['dataset_no']),
            batch_size=config['batch_size'],
            shuffle=shuffle,
            num_workers=1,  #TODO: Adjust this as necessary, potentially len(config['gpus']) * 4
            pin_memory=True,
            drop_last=drop_last
        )
    
    @classmethod
    def setup_data_loader_testset(cls, cust_data, config):
        cls.test = cls._create_dataloader(cust_data.test_ind, cust_data, config)

    @classmethod
    def setup_data_loaders_training(cls, train_ind_for_cv, train_eval_ind_for_cv, valid_ind_for_cv, cust_data, config):
        cls.training = cls._create_dataloader(train_ind_for_cv, cust_data, config, shuffle=True, drop_last=True)
        cls.training_eval = cls._create_dataloader(train_eval_ind_for_cv, cust_data, config, for_train=True)
        cls.validation = cls._create_dataloader(valid_ind_for_cv, cust_data, config)

    
