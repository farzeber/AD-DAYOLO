from data.base_dataset import BaseDataset, get_transform
from data.image_folder import make_dataset,fzb_make_dataset
from PIL import Image
import yaml
from pathlib import Path
FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]  # YOLOv5 root directory

class SingleDataset(BaseDataset):
    """This dataset class can load a set of images specified by the path --dataroot /path/to/data.

    It can be used for generating CycleGAN results only for one side with the model option '-model test'.
    """

    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseDataset.__init__(self, opt)
        dataPath = ROOT / 'data/remoteDataConfig.yaml'
        with open(dataPath, errors='ignore') as f:
            hyp = yaml.safe_load(f)
            path = Path(hyp.get('path'))  # optional 'path' default to '
            if isinstance(hyp['strain'], list):
                self.dir_A = []
                for value in hyp['strain']:
                    self.dir_A.append(str(path / value))
            else:
                self.dir_A = str(path / hyp['strain'])
        self.A_paths = sorted(
                    fzb_make_dataset(self.dir_A, opt.max_dataset_size))  # load train2007 from '/path/to/data/trainA'

        #self.A_paths = sorted(make_dataset(opt.dataroot, opt.max_dataset_size))
        input_nc = self.opt.output_nc if self.opt.direction == 'BtoA' else self.opt.input_nc
        self.transform = get_transform(opt, grayscale=(input_nc == 1))

    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index - - a random integer for data indexing

        Returns a dictionary that contains A and A_paths
            A(tensor) - - an image in one domain
            A_paths(str) - - the path of the image
        """
        A_path = self.A_paths[index]
        A_img = Image.open(A_path).convert('RGB')
        A = self.transform(A_img)
        return {'A': A, 'A_paths': A_path}

    def __len__(self):
        """Return the total number of images in the dataset."""
        return len(self.A_paths)
