
data_name="${i}_zf_${i}_b"
f="loo_bc/with_dup/gpu/${data_name}"
mkdir -p $f
mkdir -p ${f}/predictions
python3.6 main_bindzfpredictor.py -b_n ${data_name} -r 1 -p_add ${f} >> out



