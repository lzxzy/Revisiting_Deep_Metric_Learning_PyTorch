python main.py --loss triplet --batch_mining distance --project DML_Project --group Margin_with_Distance --seed 0 --gpu 0 --bs 64 --data_sampler class_random --samples_per_class 4 --arch resnet50_frozen_normalize --source /home/nfs/nfsstorage_tmp/ai_research/public/ReID_Group/data --dataset cars196 --n_epochs 150 --lr 0.00001 --embed_dim 128 --evaluate_on_gpu
