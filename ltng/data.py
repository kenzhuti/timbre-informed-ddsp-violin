from torch.utils.data import DataLoader, Dataset
from lightning.pytorch import LightningDataModule
import pathlib
import numpy as np
from tqdm import tqdm
import soundfile as sf
from functools import partial
import pandas as pd
import os

# CHRf0 version

class M21ViolinistDataset(Dataset):
    # Read directly from file, hardcoded path
    @staticmethod
    def _load_prefixes(file_path: str):

        print(f"Current working directory: {os.getcwd()}")
        print(f"File exists: {os.path.exists(file_path)}")

        try:
            with open(file_path, 'r') as f:
                return set(line.strip() for line in f if line.strip())
        except FileNotFoundError:
            print(f"Warning: {file_path} not found, using empty set")
            return set()
        
    def __init__(
        self,
        wav_dir: str,
        split: str = "train",
        duration: float = 2.0,
        overlap: float = 1.0,
        f0_suffix: str = ".pv",
        file_suffix = ".wav",
        hmag_suffix = ".tb.csv",
        use_hmag_embedding: bool = True,
    ):
        super().__init__()
        wav_dir = pathlib.Path(wav_dir)
        # train_folder_prefixes = self._load_prefixes("train_data/train_files.txt")
        # test_folder_prefixes = self._load_prefixes("train_data/test_files.txt")
        # valid_folder_prefixes = self._load_prefixes("train_data/valid_files.txt")

        train_folder_prefixes = self._load_prefixes("train_data/train_files_small.txt")
        test_folder_prefixes = self._load_prefixes("train_data/test_files_small.txt")
        valid_folder_prefixes = self._load_prefixes("train_data/valid_files_small.txt")

        print(f"Train folder prefixes: {list(train_folder_prefixes)[:10]} ... (total {len(train_folder_prefixes)})")
        print(f"Test folder prefixes: {list(test_folder_prefixes)[:10]} ... (total {len(test_folder_prefixes)})")
        print(f"Valid folder prefixes: {list(valid_folder_prefixes)[:10]} ... (total {len(valid_folder_prefixes)})")

        test_files = []
        valid_files = []
        train_files = []
        
        print("M21ViolinistDataset: Development")
        # Get all matching files
        all_files = []
        for f in wav_dir.rglob('*'):
            # Print each file being traversed
            #print(f"Found: {f} - Is file: {f.is_file()}")
    
            # Check if it's a file and suffix matches
            if f.is_file():
                # Support both .wav and wav suffix formats
                if (f.suffix == file_suffix or
                    f.suffix == f".{file_suffix}" or
                    str(f).endswith(file_suffix)):

                    all_files.append(f)
                    #print(f'Matched file: {f}')
                    parent_prefix = f.parent.name.split("#")[0]
                    wav_name = f.stem
                    # print(wav_name)
                    if wav_name in test_folder_prefixes:
                        test_files.append(f)
                    if wav_name in valid_folder_prefixes:
                        valid_files.append(f)
                    if wav_name in train_folder_prefixes:
                        train_files.append(f)
                    # else:
                    #     print(f"Warning: {wav_name} not found in any prefix list")
                #else:
                    #print(f"Skipped (suffix not match): {f}")
            #else:
                #print(f"Skipped (not file): {f}")

        # Print result statistics
        print(f"\nTotal files: {len(all_files)}")
        print(f"Train files: {len(train_files)}")
        print(f"Valid files: {len(valid_files)}")
        print(f"Test files: {len(test_files)}")
        #print('train_file is:',train_files)
        
        if split == "train":
            self.files = train_files
        elif split == "valid":
            self.files = valid_files
        elif split == "test":
            self.files = test_files
        else:
            raise ValueError(f"Unknown split: {split}")
        
        self.sample_rate = None

        file_lengths = []
        self.samples = []
        self.f0s = []

        self.use_hmag_embedding = use_hmag_embedding

        # Initialize hmag_clusters based on use_hmag_embedding
        self.hmag_clusters = [] if use_hmag_embedding else None

        # New: Record f0 hop for each file (unit: samples)
        self.f0_hop_num_frames_list = []

        print("Gathering files ...")
        for filename in tqdm(self.files):
            x, sr = sf.read(filename, dtype="float32")
            if x.ndim > 1:
                x = x.mean(axis=1)

            if self.sample_rate is None:
                self.sample_rate = sr
                self.segment_num_frames = int(duration * self.sample_rate)
                self.hop_num_frames = int((duration - overlap) * self.sample_rate)
            else:
                assert sr == self.sample_rate, f"Sample rate mismatch: {sr} != {self.sample_rate}"
            # f0 = np.loadtxt(filename.with_suffix(f0_suffix))
            # if np.isscalar(f0):
            #     f0 = np.array([f0])

            # Load f0 and hmag data from hmag file
            hmag_file = pd.read_csv(filename.with_suffix(hmag_suffix))
            f0 = hmag_file['f0'].values
            if np.isscalar(f0):
                f0 = np.array([f0])
            self.f0s.append(f0)

            if self.use_hmag_embedding:
                hmag_value = hmag_file['label'].to_numpy(dtype=int)
                self.hmag_clusters.append(hmag_value)

            #print("hmag_clusters:", self.hmag_clusters)

            # Adaptively calculate f0 hop for this file (samples)
            if len(f0) > 1:
                hop_samples = max(1, int(round(x.shape[0] / len(f0))))
            else:
                hop_samples = max(1, int(round(0.005 * self.sample_rate)))  # Fallback to 5ms
            
            self.samples.append(x)
            self.f0_hop_num_frames_list.append(hop_samples)

            file_lengths.append(
                max(0, x.shape[0] - self.segment_num_frames) // self.hop_num_frames + 1
            )

        self.file_lengths = np.array(file_lengths)
        self.boundaries = np.cumsum(np.array([0] + file_lengths))

    def __len__(self):
        return self.boundaries[-1]
    
    def __getitem__(self, index):
        bin_pos = np.digitize(index, self.boundaries[1:], right=False)
        x = self.samples[bin_pos]
        f0 = self.f0s[bin_pos]
        f0 = np.atleast_1d(f0)
        assert hasattr(f0, '__len__'), f"f0 is not sized: {type(f0)}, value: {f0}"
        assert hasattr(x, '__len__'), f"x is not sized: {type(x)}, value: {x}"
        f0 = np.where(f0 < 150, 0, f0)

        # Use f0 hop for this file
        f0_hop = self.f0_hop_num_frames_list[bin_pos]

        offset = (index - self.boundaries[bin_pos]) * self.hop_num_frames
        x = x[offset : offset + self.segment_num_frames]

        tp = np.arange(len(f0)) * f0_hop
        t = np.arange(offset, offset + self.segment_num_frames)
        mask = np.interp(t, tp, (f0 == 0).astype(float), right=1) > 0
        interp_f0 = np.where(mask, 0, np.interp(t, tp, f0))

        if x.shape[0] < self.segment_num_frames:
            x = np.pad(x, (0, self.segment_num_frames - x.shape[0]), "constant")
        else:
            x = x[: self.segment_num_frames]

        if self.use_hmag_embedding and self.hmag_clusters is not None:
            # use_hmag_embedding: {self.use_hmag_embedding}")
            # Process hmag data
            hmag_clusters = self.hmag_clusters[bin_pos]
            
            # Align hmag clusters to audio segment
            hmag_tp = np.arange(len(hmag_clusters)) * f0_hop
            t = np.arange(offset, offset + self.segment_num_frames)
            t_indices_in_hmag_tp = np.searchsorted(hmag_tp, t, side='right') - 1
            t_indices_in_hmag_tp = np.clip(t_indices_in_hmag_tp, 0, len(hmag_clusters) - 1)
            hmag_slice = hmag_clusters[t_indices_in_hmag_tp]

            if hmag_slice.shape[0] < self.segment_num_frames:
                pad_width = self.segment_num_frames - hmag_slice.shape[0]
                hmag_slice = np.pad(hmag_slice, (0, pad_width), mode='constant')
            else:
                hmag_slice = hmag_slice[:self.segment_num_frames]
                
            return x.astype(np.float32), interp_f0.astype(np.float32), hmag_slice.astype(np.int64)
        else:
            # Don't return hmag when not using hmag embedding
            # print(f"data不用use_hmag_embedding: {self.use_hmag_embedding}")
            return x.astype(np.float32), interp_f0.astype(np.float32)



class ViolinEtudesDataset(M21ViolinistDataset):
    test_folder_prefixes = set(
        [
            "p360",
            "p361",
            "p362",
            "p363",
            "p364",
            "p374",
            "p376",
            "s5",
        ]
    )

    valid_folder_prefixes = set(
        [
            "p225",
            "p226",
            "p227",
            "p228",
            "p229",
            "p230",
            "p231",
            "p232",
            "p233",
            "p234",
            "p236",
            "p237",
            "p238",
            "p239",
            "p240",
            "p241",
        ]
    )

    file_suffix = "mic1.wav"


class ViolinEtudesInferenceDataset(Dataset):
    # Read directly from file, hardcoded path
    @staticmethod
    def _load_prefixes(file_path: str):
        print(f"Current working directory: {os.getcwd()}")
        print(f"File exists: {os.path.exists(file_path)}")

        try:
            with open(file_path, 'r') as f:
                return set(line.strip() for line in f if line.strip())
        except FileNotFoundError:
            print(f"Warning: {file_path} not found, using empty set")
            return set()
    
    file_suffix = ".wav"

    def __init__(
            self, 
            wav_dir: str, 
            split: str = "train", 
            f0_suffix: str = ".pv",
            hmag_suffix: str = '.tb.csv',
            use_hmag_embedding: bool = True
        ):
        super().__init__()
        self.wav_dir = pathlib.Path(wav_dir)
        wav_dir = pathlib.Path(wav_dir)
        self.f0_suffix = f0_suffix
        self.hmag_suffix = hmag_suffix
        self.use_hmag_embedding = use_hmag_embedding

        test_folder_prefixes = self._load_prefixes("train_data/test_files_small.txt")
        valid_folder_prefixes = self._load_prefixes("train_data/valid_files_small.txt")
        train_folder_prefixes = self._load_prefixes("train_data/train_files_small.txt")
        # print(f"Test folder prefixes: {list(test_folder_prefixes)[:10]} ... (total {len(test_folder_prefixes)})")
        # print(f"Valid folder prefixes: {list(valid_folder_prefixes)[:10]} ... (total {len(valid_folder_prefixes)})")
        # print(f"Train folder prefixes: {list(train_folder_prefixes)[:10]} ... (total {len(train_folder_prefixes)})")
        test_files = []
        valid_files = []
        train_files = []

        print("ViolinEtudesInferenceDataset: Development")
        # Get all matching files
        all_files = []
        for f in wav_dir.rglob('*'):
            # Print each file being traversed
            #print(f"Found: {f} - Is file: {f.is_file()}")
    
            # Check if it's a file and suffix matches
            
            if f.is_file():
                # Support both .wav and wav suffix formats
                if (f.suffix == "wav" or 
                    f.suffix == ".wav" or
                    str(f).endswith("wav")):

                    all_files.append(f)
                    #print(f'Matched file: {f}')
                    parent_prefix = f.parent.name.split("#")[0]
                    wav_name = f.stem
                    print(wav_name)
                    if wav_name in test_folder_prefixes:
                        test_files.append(f)
                    if wav_name in valid_folder_prefixes:
                        valid_files.append(f)
                    if wav_name in train_folder_prefixes:
                        train_files.append(f)

        # Print result statistics
        print(f"\nTotal files: {len(all_files)}")
        print(f"Train files: {len(train_files)}")
        print(f"Valid files: {len(valid_files)}")
        print(f"Test files: {len(test_files)}")
        #print('train_file is:',train_files)

        if split == "train":
            self.files = train_files
        elif split == "valid":
            self.files = valid_files
        elif split == "test":
            self.files = test_files
        else:
            raise ValueError(f"Unknown split: {split}")
         
    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        filename: pathlib.Path = self.files[index]
        y, sr = sf.read(filename, dtype="float32")
        if y.ndim > 1:
            y = y.mean(axis=1)

        # Read and clean F0
        # f0_path = filename.with_suffix(self.f0_suffix)
        # f0 = np.loadtxt(f0_path)
        # f0 = np.atleast_1d(f0).astype(float)
        # f0 = np.where(f0 < 150, 0.0, f0)

        hmag_file = pd.read_csv(filename.with_suffix(self.hmag_suffix))
        f0 = hmag_file['f0'].values
        if np.isscalar(f0):
            f0 = np.array([f0])
        f0 = np.atleast_1d(f0).astype(float)
        f0 = np.where(f0 < 150, 0.0, f0)


        # Adaptively calculate hop for this file (samples: interval between F0 frames); fallback to 5ms when unable to infer
        # Total audio length = F0 frame count × frame interval i.e. y.shape[0] = len(f0) × hop_samples
        if len(f0) > 1:
            hop_samples = max(1, int(round(y.shape[0] / len(f0))))
        else:
            hop_samples = max(1, int(round(0.005 * sr)))

        # Linearly interpolate F0 to sample level and propagate silence mask
        if len(f0) == 0:
            interp_f0 = np.zeros_like(y, dtype=np.float32)
        else:
            tp = np.arange(len(f0)) * hop_samples
            t = np.arange(y.shape[0])
            mask = np.interp(t, tp, (f0 == 0).astype(float), right=1) > 0
            interp_f0 = np.where(mask, 0.0, np.interp(t, tp, f0)).astype(np.float32)

        rel_path = filename.relative_to(self.wav_dir)

        # Unified return format, return None when hmag doesn't exist
        if self.use_hmag_embedding:
            hmag_path = filename.with_suffix(self.hmag_suffix)
            if hmag_path.exists():
                hmag_df = pd.read_csv(hmag_path)
                if "label" in hmag_df.columns and len(hmag_df) > 0:
                    hmag_clusters = hmag_df["label"].to_numpy(dtype=int)
                    # Use same hop nearest neighbor upsampling to sample level
                    hmag_tp = np.arange(len(hmag_clusters)) * hop_samples
                    t = np.arange(y.shape[0])
                    idx = np.searchsorted(hmag_tp, t, side="right") - 1
                    idx = np.clip(idx, 0, len(hmag_clusters) - 1)
                    hmag_value = hmag_clusters[idx].astype(np.int64)
                else:
                    return y.astype(np.float32), interp_f0, str(rel_path)
            else:
                return y.astype(np.float32), interp_f0, str(rel_path)
        else:
            return y.astype(np.float32), interp_f0, str(rel_path)

        return y.astype(np.float32), interp_f0, hmag_value, str(rel_path)

class M21Violinist(LightningDataModule):
    def __init__(
        self,
        batch_size: int,
        wav_dir: str,
        duration: float = 2,
        overlap: float = 0.5,
        f0_suffix: str = ".pv",
        use_hmag_embedding: bool = True,  # New parameter
    ):
        super().__init__()
        self.save_hyperparameters()
        
    def setup(self, stage=None):
        factory = partial(
            M21ViolinistDataset,
            wav_dir=self.hparams.wav_dir,
            duration=self.hparams.duration,
            overlap=self.hparams.overlap,
            f0_suffix=self.hparams.f0_suffix,
            use_hmag_embedding=self.hparams.use_hmag_embedding,  # Pass parameter
        )
        
        if stage == "fit":
            self.train_dataset = factory(split="train")

        if stage == "validate" or stage == "fit":
            self.valid_dataset = factory(split="valid")

        if stage == "test":
            self.test_dataset = factory(split="test")

    def train_dataloader(self):
         
        return DataLoader(
            self.train_dataset,
            batch_size=self.hparams.batch_size,
            num_workers=4,
            shuffle=True,
            drop_last=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.valid_dataset,
            batch_size=self.hparams.batch_size,
            num_workers=4,
            shuffle=False,
            drop_last=False,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.hparams.batch_size,
            num_workers=1,
            shuffle=False,
            drop_last=False,
        )


class ViolinEtudes(M21Violinist):
    def setup(self, stage=None):
        factory = partial(
            ViolinEtudesDataset,
            wav_dir=self.hparams.wav_dir,
            duration=self.hparams.duration,
            overlap=self.hparams.overlap,
            f0_suffix=self.hparams.f0_suffix,
            use_hmag_embedding=self.hparams.use_hmag_embedding,  # Pass parameter
        )
        #print(factory)
        if stage == "fit":
            self.train_dataset = factory(split="train")

        if stage == "validate" or stage == "fit":
            self.valid_dataset = factory(split="valid")

        if stage == "test":
            self.test_dataset = factory(split="test")

        if stage == "predict":
            self.predict_dataset = ViolinEtudesInferenceDataset(
                wav_dir=self.hparams.wav_dir,
                split="test",
                f0_suffix=self.hparams.f0_suffix,
                use_hmag_embedding=self.hparams.use_hmag_embedding,  # Pass parameter
            )

    def predict_dataloader(self):
        return DataLoader(
            self.predict_dataset,
            batch_size=1,
            num_workers=0,
            shuffle=False,
            drop_last=False,
        )
