for i in 5 10 15 25 50 60 70 80 90 100 
do
	echo $i;
	lr=0.001
	f="res_12/all_data/retrain/all_data_lr_${lr}_epocs_${i}"
	mkdir -p $f
	mkdir -p ${f}/history
	mkdir -p ${f}/models
	mkdir -p ${f}/predictions
	nohup /usr/bin/time  /data/sofiaa/sofi_env_3/bin/python3.6 -u main_transfer_learning_b1h_pos.py -d_add /data/sofiaa/data/ -add ${f} -lr $lr -e $i -res_num 12 -r 0 -t_v retrain -ac_x False >> ${f}_out &
            #break

done
