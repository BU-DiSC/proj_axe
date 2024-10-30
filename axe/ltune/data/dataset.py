import glob
import numpy as np
import os
import pyarrow.parquet as pa
import torch
import torch.utils.data

from axe.ltune.data.input_features import kINPUT_FEATS, kINPUT_FEATS_ROBUST


class LTuneDataSet(torch.utils.data.IterableDataset):
    def __init__(
        self,
        folder: str,
        shuffle: bool = False,
        robust: bool = False,
    ) -> None:
        self._format = format
        self._fnames = glob.glob(os.path.join(folder, "*.parquet"))
        self._shuffle = shuffle
        self._robust = robust

    def _get_input_cols(self):
        if self._robust:
            return kINPUT_FEATS_ROBUST
        return kINPUT_FEATS

    def _load_data(self, fname):
        df = pa.read_table(fname).to_pandas()

        return df

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        files = self._fnames
        if self._shuffle:
            np.random.shuffle(files)

        if worker_info is not None:
            file_bins = np.array_split(files, worker_info.num_workers)
            files = file_bins[worker_info.id]

        for file in files:
            df = self._load_data(file)
            inputs = torch.from_numpy(df[self._get_input_cols()].values).float()
            for input in inputs:
                # We return input twice, as the loss function for our learned
                # tuner will be taking in the workload as a label, and the
                # design prediciton to calculate estimate IO cost
                yield input, input
            del df  # attempt to release dataframe memory
