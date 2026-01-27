


python -u main.py \
  --num_emb_list 256 256 256 256 \
  --sk_epsilons 0.0 0.0 0.0 0.003 \
  --device cuda:1 \
  --data_path /home/xinyulin/context/data/amazon/info/Toys_and_Games.emb-t5-tdcb.npy \
  --batch_size 480

