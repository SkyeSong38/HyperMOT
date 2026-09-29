import os
import os.path as osp

import numpy as np


def mkdirs(path):
    if not osp.exists(path):
        os.makedirs(path)


project_root = osp.abspath(osp.join(osp.dirname(__file__), '..'))
mot20_root = osp.abspath(osp.join(project_root, '..', 'MOT20'))
seq_root = osp.join(mot20_root, 'train')
label_root = osp.join(mot20_root, 'trackers_gt')

mkdirs(label_root)

for seq in sorted(os.listdir(seq_root)):
    if not seq.startswith('MOT20-'):
        continue

    print(seq)
    seq_info_path = osp.join(seq_root, seq, 'seqinfo.ini')
    seq_info = open(seq_info_path).readlines()
    seq_width = next(int(line.split('=')[1].strip()) for line in seq_info if 'imWidth=' in line)
    seq_height = next(int(line.split('=')[1].strip()) for line in seq_info if 'imHeight=' in line)

    gt_txt = osp.join(seq_root, seq, 'gt', 'gt.txt')
    gt = np.loadtxt(gt_txt, dtype=np.float64, delimiter=',')
    gt = gt[np.lexsort(gt.T[:2, :]), :]

    seq_label_root = osp.join(label_root, seq, 'img1')
    mkdirs(seq_label_root)
    written_paths = set()

    for fid, tid, x, y, w, h, mark, cls, vis in gt[:, :9]:
        if fid < 1 or mark == 0 or cls != 1:
            continue

        fid = int(fid)
        tid = int(tid)
        x += w / 2
        y += h / 2
        label_path = osp.join(seq_label_root, '{:06d}.txt'.format(tid))
        label = '0 {:d} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f}\n'.format(
            fid, x / seq_width, y / seq_height, w / seq_width, h / seq_height, vis)
        with open(label_path, 'a' if label_path in written_paths else 'w') as file:
            file.write(label)
        written_paths.add(label_path)
