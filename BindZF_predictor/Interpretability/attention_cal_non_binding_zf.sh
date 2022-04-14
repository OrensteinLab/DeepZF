for i in 40
do
	data_name="${i}_zf_${i}_b"
	f="attention_results/non_binding_zf"
             mkdir -p $f
	mkdir -p $f/pre_train
	mkdir -p $f/fine_tuned
	nohup /usr/bin/time /data/sofiaa/sofi_env_3/bin/python3.6 -u main_imp_zf.py -b_n ${data_name} -r 1 -aten_add ${f} -exp_n /data/sofiaa/proteinBERT/attention_calculation/viz_df/non_binding_zfs.csv >> ${f}/out &

done

