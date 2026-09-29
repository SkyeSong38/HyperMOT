import os.path as osp
import os
import numpy as np


def mkdirs(d):
    if not osp.exists(d):
        os.makedirs(d)

project_root = osp.abspath(osp.join(osp.dirname(__file__), '..'))
mot17_root = osp.abspath(osp.join(project_root, '..', 'MOT17'))
seqs_root = [osp.join(mot17_root, 'train')]

label_root = osp.join(mot17_root, 'MOT_union', 'trackers_gt')

mkdirs(label_root)

interval = 5

for seq_root in seqs_root:
    seqs = [s for s in os.listdir(seq_root)]

    tid_curr = 0
    tid_last = -1
    for seq in seqs:
        if 'MOT17' in seq and 'DPM' not in seq:
            print(seq+"-------overlooked!")
            continue

        # if seq in os.listdir(label_root):
        #     print(seq+"-------already exists!")
        #     continue

        print(seq+"-------processing...")
        seq_gt_path = os.path.join(seq_root, seq, 'gt/gt.txt')
        seq_info_path = os.path.join(seq_root, seq, 'seqinfo.ini')

        # 读取序列信息
        seq_info = open(seq_info_path).readlines()
        im_width = next(int(s.split('=')[1].strip()) for s in seq_info if 'imWidth=' in s)
        im_height = next(int(s.split('=')[1].strip()) for s in seq_info if 'imHeight=' in s)
        seq_length = next(int(s.split('=')[1].strip()) for s in seq_info if 'seqLength=' in s)

        # 读取gt信息
        gt = np.loadtxt(seq_gt_path, dtype=np.float64, delimiter=',')
        idx = np.lexsort(gt.T[:2, :])
        gt = gt[idx, :]

        # 找到每一帧中的object编号
        object_matrix = [[] for _ in range(seq_length)]

        for fid, tid, x, y, w, h, mark, cls, vis in gt[:,:9]:
            if fid < 1 or mark == 0:
                continue
            if 'MOTSynth' not in seq_gt_path and not cls == 1:
                continue
            fid = int(fid)
            tid = int(tid)
            object_matrix[fid - 1].append(tid)

        # 计算每一帧的以及后面interval帧中的object交集
        common_object_matrix = [[] for _ in range(seq_length - interval - 1)]
        for i in range(len(common_object_matrix)):
            common_elements = set(x for x in object_matrix[i] if x != -1)
            for row in object_matrix[i + 1:i + interval + 2]:
                common_elements &= set(x for x in row if x != -1)
            common_object_matrix[i] = common_elements

        # 将对应的gt找出来，并存入文件
        label_str = [[] for _ in range(len(common_object_matrix))]
        for fid, tid, x, y, w, h, mark, cls, vis in gt[:,:9]:
            if fid < 1 or mark == 0:
                continue
            if 'MOTSynth' not in seq_gt_path and not cls == 1:
                continue
            fid = int(fid)
            tid = int(tid)
            x += w / 2
            y += h / 2
            gt_str = '{:d} {:d} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f}\n'.format(
                fid, tid, x / im_width, y / im_height, w / im_width, h / im_height, vis)
            for cur_fid in range(len(common_object_matrix)):
                if fid in range(cur_fid + 1, cur_fid + interval + 3) and tid in common_object_matrix[cur_fid]:
                    label_str[cur_fid].append(gt_str)

        # 存文件
        seq_label_root = osp.join(label_root, seq)
        mkdirs(seq_label_root)
        for i in range(len(label_str)):
            label_fpath = osp.join(seq_label_root, '{:06d}.txt'.format(i))
            if label_str[i]:
                with open(label_fpath, 'w') as f:
                    f.writelines(label_str[i])
