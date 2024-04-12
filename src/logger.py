import os
import csv

class Logger():
    # Initialize class variables
    commit = {}  # Dictionary to store committed metrics
    enable_wandb = False  # Flag to determine if Weights & Biases logging is enabled

    @classmethod
    def init(cls, cv, checkpoint, config):
        """
        Initialize the logger.

        Args:
            cv (int): Cross-validation fold number.
            checkpoint: Object containing checkpoint information.
            config (dict): Configuration parameters including W&B settings.
        """
        # Check if W&B logging is enabled
        cls.enable_wandb = config['enable_wandb']
        if cls.enable_wandb:
            import wandb

            # Set W&B API key from the configuration
            if config['wandb'] is not None:
                os.environ['WANDB_API_KEY'] = config['wandb']
            else:
                raise AssertionError("W&B is missing API key argument from this program. See docs for more information.")
                
            # Initialize W&B run
            wandb.init(
                project=config['wb_project'],
                entity='lennardkorte',  # Set entity for data privacy
                group=config['group'],
                id=checkpoint.wandb_id,
                resume="allow",
                name=config['name'] + '_cv_' + str(cv),
                reinit=True,
                dir=os.getenv("WANDB_DIR", config.save_path))
    
    @classmethod   
    def get_id(cls):
        """
        Generate a unique ID using Weights & Biases utility function.

        Returns:
            str: Unique ID.
        """
        import wandb
        return wandb.util.generate_id()

    @classmethod   
    def add(cls, metrics:dict, prefix=None):
        """
        Add metrics to the commit dictionary.

        Args:
            metrics (dict): Dictionary containing metrics and their values.
            prefix (str, optional): Prefix to be added to metric keys. Defaults to None.
        """
        pf = ''  # Initialize prefix string
        if prefix is not None:
            pf = prefix + ' '  # Set prefix if provided
        for key, value in metrics.items():
            cls.commit[pf + key] = value  # Add metrics to the commit dictionary
        
    @classmethod
    def push(cls, path):
        """
        Push committed metrics to a CSV file and optionally to Weights & Biases.

        Args:
            path (str): Path to the CSV file.
        """
        # Write committed metrics to CSV file
        with open(path, 'a', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=cls.commit.keys())
            csvfile.seek(0, 2)
            if csvfile.tell() == 0: writer.writeheader()  # Write header if file is empty
            writer.writerow(cls.commit)  # Write metrics to CSV

        # If W&B logging is enabled, log metrics to W&B
        if cls.enable_wandb:    
            import wandb
            wandb.log(cls.commit)  # Log metrics to W&B
        
        # Reset commit dictionary
        cls.commit = {}
