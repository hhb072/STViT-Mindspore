NCCL_P2P_LEVEL=NVL python -m torch.distributed.launch --nproc_per_node=8 --use_env --master_port 29555 main.py  --data-path /home/huaibo.huang/data/imagenet/ILSVRC2015/Data/CLS-LOC  --batch-size 64  --drop-path 0.6 --epoch 50  --finetune ckpt/best.pth  --lr 5e-6 --min-lr 5e-6  --warmup-epochs 0 --weight-decay 1e-8  --input-size 384  --dist-eval  --output_dir ckpt384 2>&1 | tee -a train384.txt
