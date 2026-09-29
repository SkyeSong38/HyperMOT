from torch.utils.data import Dataset
import numpy as np

import os
import glob


class HyperDataset(Dataset):
    def __init__(self, path, config=None):
        self.config = config
        self.interval = self.config.interval
        self.pad = 300

        self.nS = 0  # 轨迹 数据集 总的 时序采样数量 --- 数据集中 所有轨迹的 时序采样数量 之和
        self.frames = {}
        self.nds = {}
        self.cds = {}  # 整个 数据集中，所有轨迹的 时序采样数量 的累加
        if os.path.isdir(path):
            self.seqs = os.listdir(path)
            self.seqs.sort()
            for seq in self.seqs:
                trackerPath = os.path.join(path + "/" + seq, "*.txt")
                self.frames[seq] = sorted(glob.glob(trackerPath))
                track_of_len = len(self.frames[seq])

                self.nds[seq] = track_of_len
                self.cds[seq] = self.nS
                self.nS += track_of_len

        print('=' * 80)
        print('dataset summary')
        print(self.nS)
        print('=' * 80)

    def __getitem__(self, files_index):  # 某一个视频中 的某一个轨迹 的某一时刻的索引

        for i, seq in enumerate(self.cds):  # 遍历视频序列
            if files_index >= self.cds[seq]:  # 时刻索引 是否大于 该视频序列的 第一条轨迹 -- 对应的开始时刻索引
                ds = seq  # 如果大于 则说明 提取的 时刻索引 --- 属于 这一 视频序列 范围内 ds：确定 提取时刻索引 对应的 视频序列
                start_index = self.cds[seq]
            else:
                break

        init_index = files_index - start_index
        track_path = self.frames[ds][init_index]  # 提取 对应视频中 某一条轨迹的 对应标注的txt文件
        track_gt = np.loadtxt(track_path, dtype=np.float32)  # 读取该轨迹的标注
        object_num = len(track_gt) // (self.interval + 2)

        track_gt = track_gt[np.lexsort((track_gt[:, 1], track_gt[:, 0]))]  # 先按照fid排序，在内部按照tid排序

        cur_bbox = np.zeros((self.interval + 1, object_num, 4), dtype=np.float32)
        gt_bbox = np.zeros((1, object_num, 4), dtype=np.float32)

        # 这里直接按照顺序组成二位数组就行了。
        for i in range(self.interval + 1):
            cur_bbox[i] = track_gt[i*object_num:(i+1)*object_num, 2:6]
        gt_bbox[0] = track_gt[(self.interval+1)*object_num:, 2:6]

        # 差分运动
        delta_bbox = np.zeros((self.interval, object_num, 4), dtype=np.float32)
        for i in range(len(cur_bbox)-1):
                delta_bbox[i] = cur_bbox[i+1] - cur_bbox[i]

        cur_bbox = cur_bbox[1:]

        # 填充到300
        if object_num > self.pad:
            cur_bbox = cur_bbox[:,:self.pad,:]
            delta_bbox = delta_bbox[:, :self.pad, :]
            gt_bbox = gt_bbox[:, :self.pad, :]
            print("cut the overFlow objects: "+ track_path)
        elif object_num < self.pad:
            pad_width = ((0,0),(0,self.pad-object_num),(0,0))
            cur_bbox=np.pad(cur_bbox, pad_width=pad_width, mode='constant', constant_values=0)
            delta_bbox = np.pad(delta_bbox, pad_width=pad_width, mode='constant', constant_values=0)
            gt_bbox = np.pad(gt_bbox, pad_width=pad_width, mode='constant', constant_values=0)

        conds = np.concatenate((cur_bbox, delta_bbox), axis=2)  # 。前5个时刻的 bbox ｜ 前5个时刻的 差分运动

        valid_mask = np.arange(self.pad) < min(object_num, self.pad)
        ret = {"gt_bbox": gt_bbox, "cur_bbox": cur_bbox, "condition": conds,
               "delta_bbox": delta_bbox, "valid_mask": valid_mask}
        return ret

    def __len__(self):
        return self.nS

# if __name__ == "__main__":
#     data_path = '../DanceTrack/trackers_gt_t'
#     a = DiffMOTDataset_longterm(data_path)
#     b = a[700]
#     pass
