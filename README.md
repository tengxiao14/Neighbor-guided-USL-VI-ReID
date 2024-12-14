# Relieving Universal Label Noise for Unsupervised Visible-Infrared Person Re-Identification by Inferring from Neighbors

## Dataset Preprocessing

Convert the dataset format (like Market1501).
```shell
python prepare_sysu.py   # for SYSU-MM01
python prepare_regdb.py  # for RegDB
```
You need to change the file path in the `prepare_sysu(regdb).py`.

Then put them under the directory like

```
data/USL-VI-ReID
├── SYSU-MM01
└── RegDB
```

## Environment

numpy, torch, torchvision,

six, h5py, Pillow, scipy,

scikit-learn, metric-learn, 

faiss_gpu

## Training

We utilize 4 Tesla A100 GPUs for training. Follwing PGM, our method also includes two stages for training:
 

**examples:**

#for SYSU-MM01:

CUDA_VISIBLE_DEVICES=0,1,2,3 python train_sysu.py -b 256 -a agw -d  sysu_all --num-instances 16 --data-dir 'data/USL-VI-ReID' --eps 0.6 --logs-dir 'results_sysu'  --stage s1

CUDA_VISIBLE_DEVICES=0,1,2,3 python train_sysu.py -b 256 -a agw -d  sysu_all --num-instances 16 --data-dir 'data/USL-VI-ReID' --eps 0.6 --logs-dir 'results_sysu' --lambda1 3.0 --balance 0.7 --coe 10.0 --neighbour 30 --stage s2

#for RegDB:

#trial: 1,2,3,4,5,6,7,8,9,10

CUDA_VISIBLE_DEVICES=0,1,2,3 python train_regdb.py -b 256 -a agw -d  regdb_rgb --num-instances 16 --data-dir 'data/USL-VI-ReID' --eps 0.2 --logs-dir 'results_regdb' --stage s1 --trial 1

CUDA_VISIBLE_DEVICES=0,1,2,3 python train_regdb.py -b 256 -a agw -d  regdb_rgb --num-instances 16 --data-dir 'data/USL-VI-ReID' --eps 0.2 --logs-dir 'results_regdb' --lambda1 3.0 --balance 0.7 --coe 10.0 --neighbour 20 --stage s2 --trial 1



## Evaluation

**examples:**

#for SYSU-MM01:

CUDA_VISIBLE_DEVICES=0,1,2,3 python test_sysu.py -b 256 -a agw -d  sysu_all  --num-instances 16 --logs-dir "results_sysu"

#for RegDB:

CUDA_VISIBLE_DEVICES=0,1,2,3 python test_regdb.py -b 256 -a agw -d  regdb_rgb --num-instances 16 --logs-dir "results_regdb"

**The code is implemented based on PGM.**



