1. decoder only的训练 （晚上就能跑，明早收菜）
2. t5 small换成memory layer的训练  （晚上开发，明早训一下）
3. t5 small比如说train到D2的ckpt，然后中间一个FFN换成memory（初始化）再finetune，其余冻住 （等t5 small出来）
4. 试一下把k的topk选择 等于 K和V的词表大小