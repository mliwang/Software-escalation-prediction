for((i=0;i<5;i++));  
do  
  python Fusionlayer/mymain.py \
      --kfold=5 \
      --index=$i \
      --train_batch_size=256 \
      --eval_steps=5000 \
      --max_len_text=128 \
      --epoch=5 \
      --lr=1e-4 \
      --output_path=saved_models \
      --pretrained_model_path=BERT/bert-base \
      --eval_batch_size=512 2>&1 | tee saved_models/log/$i.txt
done  