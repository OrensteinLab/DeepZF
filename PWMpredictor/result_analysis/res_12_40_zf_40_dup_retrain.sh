f="/data/sofiaa/pipeline_transfer_learning/b1h_7_pos/res_12/40_zf_40/dup"

for file in ${f}/*_out
do
	mkdir -p $f/outputs
	# echo $file;
	mv ${file}  $f/outputs
done 

for dir in ${f}/*_lr_*
do
	echo $dir;
	mkdir -p $f/mosbat_input

	pred_add=$dir/predictions
	nohup /data/sofiaa/sofi_env_2/bin/python3.6 create_mosbat_input_protein_bert.py -p_add ${pred_add} -c_rc_add /data/sofiaa/data/c_rc_df.csv -zf_add /data/sofiaa/data/zf_pred_40_zf_40_dup.csv  -s_add ${f}/mosbat_input/ -exp_name ${dir} >> ${f}/mosbat_input/out_mosbat &	
done
sleep 5

cd /data/sofiaa/MoSBAT/MoSBAT-master/

for file in $f/mosbat_input/*epocs*
do

	echo $file;
	#echo ${file: 13:-8}
        bash MoSBAT.sh ${file: 13:-8} ${file} ${f}/mosbat_input/gt_pwm.txt dna 100 5000
	mkdir -p $f/mosbat_output/${file: -30:-4}

	cp /data/sofiaa/MoSBAT/MoSBAT-master/out/${file: 13:-8}/results.energy.correl.txt ${f}/mosbat_output/${file: -30:-4}/
	cp /data/sofiaa/MoSBAT/MoSBAT-master/out/${file: 13:-8}/results.affinity.correl.txt ${f}/mosbat_output/${file: -30:-4}/
	sleep 10

done
cd /data/sofiaa/pipeline_transfer_learning/b1h_7_pos/mosbat
sleep 10

for res in ${f}/mosbat_output/*epocs*
do
	nohup /usr/bin/time /data/sofiaa/sofi_env_3/bin/python3.6 eval_mosbat.py -a_add ${res}/results.affinity.correl.txt -e_add ${res}/results.energy.correl.txt -s_add ${res}/ >> ${res}/out_eval_mosbat &

done






