import os
import pandas as pd
import torch
import numpy as np
from Utilities.Datasets.base import DatasetBase

class myDatasetBase(DatasetBase):
    """ A dataset class that takes a csv file (e.g. from pandas).
        Split has to be defined as a colum if any mode except 'all' should be used
        :param csv_file: path to the csv file
        remaining params see base class
        """
    def __init__(self,
                 csv_file,
                 *args,
                 **kwargs):
        base_path = os.path.dirname(csv_file)
        self.csv_file = csv_file
        super().__init__(base_path, *args, **kwargs)

    def read_annotations(self):
        # load csv file
        df = pd.read_csv(self.csv_file)
        """ Drop entries without full information """
        df = df.dropna()
        """ Drop gray valued entries """
        # for x in df.index:
        #     if df.loc[x, "modality"] == 'gray':
        #         df.drop(x, inplace=True)
        listOfCounts = df['person_id'].value_counts()
        """ Drop all entries from dataframe, which dont have a certain amount of occurences.
            Example: minOcc = 5. All entries, which dont occur 5 times in dataframe are dropped. """
        if self.csv_file == 'D:\\Projects\\FaceRecognition\\colorferet.csv':
            #minOcc = 10
            minOcc = 5
            maxOcc = max(listOfCounts)
        else:
            minOcc = 100
            maxOcc = 180
            minOcc = 1
            maxOcc = max(listOfCounts)
        occurence = range(minOcc, maxOcc+1)
        indicesOfCounts = []
        for i in occurence:
            indicesOfCounts.append(list(np.where(np.array(listOfCounts) == i)[0]))
        indicesOfCounts = [item for sublist in indicesOfCounts for item in sublist]
        """ occurenceList: List of person_id's which occur [occurence] times """
        occurenceList = listOfCounts.axes[0][indicesOfCounts].to_list()
        """ Drop all entries which are not in IDlist """
        df = df[df['person_id'].isin(occurenceList)]
        """ IDmap showing list of available person_id's """
        IDmap = list(set(df.iloc[:, 3]))
        data = list(df.T.to_dict().values())

        # convert bbox and landmark values to arrays
        for idx in range(len(data)):
            if "face_x" in data[idx]:
                data[idx]["boundingbox"] = [int(data[idx]["face_x"]), int(data[idx]["face_y"]), int(data[idx]["face_w"]),
                                     int(data[idx]["face_h"])]
                data[idx].pop("face_x");
                data[idx].pop("face_y");
                data[idx].pop("face_w");
                data[idx].pop("face_h")
            data[idx]["person_id"] = IDmap.index(data[idx]["person_id"])
            #if "sdk_landmark_x0" in data[idx]:
            #    lm_array = np.zeros((75, 2))
            #    for i in range(75):
            #        lm_array[i][0] = data[idx].get("sdk_landmark_x" + str(i), -1)
            #        lm_array[i][1] = data[idx].get("sdk_landmark_y" + str(i), -1)
            #        data[idx].pop("sdk_landmark_x" + str(i));
            #        data[idx].pop("sdk_landmark_y" + str(i))
            #    data[idx]["landmarks"] = lm_array.tolist()  # json can't story numpy arrays
        self.data = data
    # we have to change the default annotations.json filename to dataset_name.json
    # as there might be multiple csv files in one folder. The remaining behavior stays the same

    # def getSplits(self, data, ratios=None):
    #     """
    #     :param ratios: list of 3 floats adding up to 1, ratios of train, val, test splits.
    #         may be overridden if the dataset already has established splits
    #     :return: dict of splits as lists of dicts
    #     """
    #     if ratios is None:
    #         ratios = [1, 0, 0]
    #     random.Random(123456789).shuffle(data)
    #     assert(sum(ratios) == 1)
    #     splits = {}
    #     offset = 0
    #     for mode, ratio in zip(['train', 'validation', 'test'], ratios):
    #         o = int(len(data) * offset)
    #         split = data[o: o + int(len(data) * ratio)]
    #         splits[mode] = split
    #         offset += ratio
    #     return splits


def collate(verbose=True, device='cpu'):
    def inner(batch):
        batch = list(filter(lambda x: x is not None, batch))
        if len(batch) < 1:
            return
        if torch.is_tensor(batch[0]['computed']['image']):
            img = torch.stack([s['computed']['image'].to(device) for s in batch], 0).type(torch.float32)
            gt_labels = torch.stack([torch.tensor(s["person_id"], dtype=torch.int64).to(device) for s in batch], dim=0)
        else:
            img = np.stack([s['computed']['image'] for s in batch], 0).round().astype(np.uint8)
            gt_labels =np.stack([s["person_id"] for s in batch], 0).astype(np.int64)
        if verbose:
            meta = []
            for s in batch:
                sample_meta = {k: v for k, v in s.items() if k not in ['computed']}
                meta.append(sample_meta)

            return img, gt_labels, meta
        else:
            return img, gt_labels, None
    return inner