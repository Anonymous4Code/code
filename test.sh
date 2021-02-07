OPTION_NUM=6
STATE_DIM=312
ACTION_DIM=35
TRAJ_ITER=100
OPTION_DIM=10
META_TRAIN_EPOCHS=1200
ADAPTATION=20
WARM_START_EPOCH=100
SAVE_FOLDER='meta_gail/'
ALPHA=0.1 
INTERVAL=5
META_TRAIN=4
DEMO=5
ROOM_NUM=4
for ITERATOR in 5
do
for META_LR in 0.0001 
do
for INNER in 5
do
python meta_learner.py --CPU\
                --option_num $OPTION_NUM --state_dim $STATE_DIM\
                --action_dim $ACTION_DIM --save_folder $SAVE_FOLDER\
                --traj_iter $TRAJ_ITER --option_dim $OPTION_DIM\
                --meta_train_epochs $META_TRAIN_EPOCHS --adaptation_epochs $ADAPTATION\
                --warm_start_epoch $WARM_START_EPOCH\
                --alpha $ALPHA --LSTM_seg_interval $INTERVAL\
                --meta_train_num $META_TRAIN --demos $DEMO\
                --room_num $ROOM_NUM --meta_iterator $ITERATOR\
                --meta_lr $META_LR --meta_inner $INNER

done
done
done