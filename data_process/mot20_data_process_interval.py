import os
import os.path as osp

import numpy as np


def mkdirs(path):
    if not osp.exists(path):
        os.makedirs(path)


project_root = osp.abspath(osp.join(osp.dirname(__file__), '..'))
mot20_root = osp.abspath(osp.join(project_root, '..', 'MOT20'))
seq_root = osp.join(mot20_root, 'train')
label_root = osp.join(mot20_root, 'MOT_union', 'trackers_gt')

mkdirs(label_root)

interval = 5

for seq in sorted(os.listdir(seq_root)):
    if not seq.startswith('MOT20-'):
        continue

    print(seq + '-------processing...')
    seq_gt_path = osp.join(seq_root, seq, 'gt', 'gt.txt')
    seq_info_path = osp.join(seq_root, seq, 'seqinfo.ini')

    seq_info = open(seq_info_path).readlines()
    im_width = next(int(line.split('=')[1].strip()) for line in seq_info if 'imWidth=' in line)
    im_height = next(int(line.split('=')[1].strip()) for line in seq_info if 'imHeight=' in line)
    seq_length = next(int(line.split('=')[1].strip()) for line in seq_info if 'seqLength=' in line)

    gt = np.loadtxt(seq_gt_path, dtype=np.float64, delimiter=',')
    gt = gt[np.lexsort(gt.T[:2, :]), :]

    object_matrix = [[] for _ in range(seq_length)]
    for fid, tid, x, y, w, h, mark, cls, vis in gt[:, :9]:
        if fid < 1 or mark == 0 or cls != 1:
            continue
        object_matrix[int(fid) - 1].append(int(tid))

    common_object_matrix = [[] for _ in range(seq_length - interval - 1)]
    for i in range(len(common_object_matrix)):
        common_elements = set(tid for tid in object_matrix[i] if tid != -1)
        for row in object_matrix[i + 1:i + interval + 2]:
            common_elements &= set(tid for tid in row if tid != -1)
        common_object_matrix[i] = common_elements

    labels = [[] for _ in range(len(common_object_matrix))]
    for fid, tid, x, y, w, h, mark, cls, vis in gt[:, :9]:
        if fid < 1 or mark == 0 or cls != 1:
            continue

        fid = int(fid)
        tid = int(tid)
        x += w / 2
        y += h / 2
        label = '{:d} {:d} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f}\n'.format(
            fid, tid, x / im_width, y / im_height, w / im_width, h / im_height, vis)
        for cur_fid in range(len(common_object_matrix)):
            if fid in range(cur_fid + 1, cur_fid + interval + 3) \
                    and tid in common_object_matrix[cur_fid]:
                labels[cur_fid].append(label)

    seq_label_root = osp.join(label_root, seq)
    mkdirs(seq_label_root)
    for i, frame_labels in enumerate(labels):
        if not frame_labels:
            continue
        label_path = osp.join(seq_label_root, '{:06d}.txt'.format(i))
        with open(label_path, 'w') as file:
            file.writelines(frame_labels)
