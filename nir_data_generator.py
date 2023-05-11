import random
from dataclasses import dataclass
from typing import Tuple
import pandas as pd
import numpy as np
import torch
# Write a custom dataset class (inherits from torch.utils.data.Dataset)
from torch.utils.data import Dataset


class NIRSpectralData(Dataset):

    # 2. Initialize with a targ_dir and transform (optional) parameter
    def __init__(self, root_dir: str, label_column_name: str, class_dict: dict = None, transform=None) -> None:

        # setup root_dir to .csv file
        self.root_dir = root_dir
        # setup label columns name
        self.label_col_name = label_column_name
        # setup data and labels
        self.df = pd.read_csv(root_dir, index_col=0)

        self.data = torch.from_numpy(
            self.df.iloc[:, :-1].to_numpy().astype("float32")).type(torch.float32)

        self.labels = torch.from_numpy(
            self.df[self.label_col_name].to_numpy().astype("uint8")).type(torch.uint8)

        # setup class dictionary
        if class_dict:
            self.class_to_idx = class_dict
            self.classes = list(class_dict.keys())
        else:
            self.class_to_idx = {
                i: i for i in range(len(self.labels.unique()))}
            self.classes = list(self.class_to_idx.keys())
        # Setup transforms
        self.transform = transform

    # override __len__ and __getitem__ methods
    def __len__(self) -> int:
        "Returns the total number of samples."
        return len(self.df)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        "Returns one sample of data, data and label (X, y)."
        spectral_data = self.data[index]
        class_idx = self.labels[index].item()

        # Transform if necessary
        if self.transform:
            # return data, label (X, y)
            return self.transform(spectral_data), class_idx
        else:
            return spectral_data, class_idx  # return data, label (X, y)


@dataclass
class NIRDataGenerator():
    df: pd.DataFrame  # dataframe with columns 0:-1 as spectra data with different wavelenghts and last column as class label
    batch_size: int  # number of samples per batch
    class_col_name: str  # name of the column containing the class label

    weighted_sum: bool = True
    roll: bool = True
    # number of shift (needs to be multiplied with nm resolution of the spectral data to obtain the real shift)
    roll_factor: int = 12

    slope: bool = True
    slope_factor: float = 0.2

    noise: bool = True
    noise_range: tuple = (80, 100)

    # if True, labels are sparse (es 0,1,2,...) , otherwise categorical (with one-hot encoding, es: [1,0,0], [0,1,0], [0,0,1], ...])
    sparse_labels: bool = True

    def __post_init__(self):
        # transform to numpy for performance reasons
        self.samples = self.df.iloc[:, :-1].to_numpy().astype("float32")
        self.labels = self.df[self.class_col_name].to_numpy().astype("uint8")
        self.columns = self.df.columns[:-1]

    def __len__(self):
        return int(len(self.df) // self.batch_size)

    def __getitem__(self, index):
        # selection of mini-batch
        BOTTOM = index * self.batch_size
        TOP = (index + 1) * self.batch_size
        batch_samples = self.samples[BOTTOM:TOP]
        batch_labels = self.labels[BOTTOM:TOP].reshape((self.batch_size, 1))

        batch_samples = self._augmentation(batch_samples, batch_labels)
        batch_labels = batch_labels.reshape((self.batch_size,))

        # in case of categorical crossentropy loss, labels are translated
        # form sparse to categorical
        if not self.sparse_labels:
            # TODO: to implement
            pass

        data = pd.DataFrame(batch_samples, columns=self.columns)
        data['class'] = batch_labels
        return (
            data
        )

    def augment_all_once(self):
        # selection of all data
        data_len = len(self.df)
        batch_samples = self.samples
        batch_labels = self.labels.reshape((data_len, 1))
        batch_samples = self._augmentation(
            batch_samples, batch_labels, batch_size=data_len)
        batch_labels = batch_labels.reshape((data_len,))

        # in case of categorical crossentropy loss, labels are translated
        # form sparse to categorical
        if not self.sparse_labels:
            # TODO: to implement
            pass

        data = pd.DataFrame(batch_samples, columns=self.columns)
        data['class'] = batch_labels

        return data

    def augment_n_times(self, iterations: int = 10):

        print(f"Augmenting {iterations} times...")
        data_df_list = []

        for _ in range(iterations):
            data = self.augment_all_once()
            data_df_list.append(data)

        augmented_df = pd.concat(data_df_list)

        print(f"Done!\nDataset is now composed of {len(augmented_df)} samples")

        return augmented_df

    def _augmentation(self, batch_samples, batch_labels, batch_size=None):
        """Compute data augmentation on 'batch_samples', applying 
        weighted sum + roll(horizontal shift) + baseline noise + adittive white gaussian noise

        Args:
            batch_samples (np.array): Batch of spectra (1d array)
            batch_labels (np.array): Batch of label, number that are class identifier

        Returns:
            np.array: Batch of augmented data
        """

        if batch_size is None:
            batch_size = self.batch_size
        else:
            batch_size = batch_size

        if self.weighted_sum:
            alpha = np.random.rand(batch_size)

            other_samples = np.apply_along_axis(
                self._get_random_sample_from_class, 1, batch_labels
            ).reshape(batch_size, batch_samples.shape[1])

            if self.roll:
                other_samples = np.apply_along_axis(
                    self._random_roll, 1, other_samples)

            batch_samples = (
                np.multiply(
                    batch_samples -
                    other_samples, alpha.reshape(batch_size, 1)
                )
                + other_samples
            )

        if self.slope:
            batch_samples = np.apply_along_axis(
                lambda x: self._produce_background_baseline(
                    x, batch_samples.shape[1]),
                1,
                batch_samples,
            )

        if self.noise:
            batch_samples = np.apply_along_axis(
                self._random_noise, 1, batch_samples)

        return batch_samples

    def _get_random_sample_from_class(self, label):
        """Extract a random sample from the datas marked as 'label'

        Args:
            label (int): Number that describe the class identifier of data to select

        Returns:
            np.array: Random sample of the 'label' class
        """
        class_indexes = np.where(self.labels == label)[0]
        CLASS_INDEX = np.random.choice(class_indexes, 1)[0]

        return self.samples[CLASS_INDEX: CLASS_INDEX + 1]

    def _random_noise(self, arr):
        """Apply adittive white gaussian noise to 'arr' of magnitued 'noise_range'

        Args:
            arr (np.array): Sample to wich apply the noise

        Returns:
            np.array: Noise 'arr'
        """
        rnd_snr = random.randint(self.noise_range[0], self.noise_range[1])
        NOISE_FACTOR = 1 / (10 ** (rnd_snr / 10))

        return arr + np.random.normal(0, NOISE_FACTOR, len(arr))

    def _random_roll(self, arr):
        """Apply random roll (numpy way to say horizontal shift) to 'arr' of magnitude 'roll_factor'

        Args:
            arr (np.array): Sample to wich apply the roll

        Returns:
            np.array: Random rolled sample
        """

        SHIFT_FACTOR = self.roll_factor
        random_shift = random.randint(-1 * SHIFT_FACTOR, SHIFT_FACTOR)

        rolled = np.roll(arr, random_shift)
        padded = (
            np.pad(rolled[random_shift:], (random_shift, 0), "edge")
            if random_shift >= 0
            else np.pad(rolled[:random_shift], (0, abs(random_shift)), "edge")
        )

        return padded

    def _produce_background_baseline(self, arr, steps):
        """Apply a random baseline noise to 'arr' of magnitude 'slope_factor' 

        Args:
            arr (np.array): Spectrum to wich apply the baseline noise
            steps (int): Length of the 'arr' argument

        Returns:
            np.array: Noised spectrum
        """

        SLOPE = random.triangular(-1 * self.slope_factor, self.slope_factor)
        line = (
            np.linspace(abs(SLOPE), 0, steps)
            if SLOPE < 0
            else np.linspace(0, SLOPE, steps)
        )

        alpha = random.random()

        return arr * alpha + line * (1 - alpha)
