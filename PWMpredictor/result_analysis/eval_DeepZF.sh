f="update main folder path: main directory path pf pwm predictor"

dir=${f}/pwm_predict_folder

echo $dir;
mkdir -p $f/mosbat_input
pred_add=$dir/predictions
python3.6 create_mosbat_input_protein_bert.py -p_add ${pred_add} -c_rc_add 'update_path/'c_rc_df.csv -zf_add 'update_path/'zf_pred_df.csv  -s_add ${f}/mosbat_input/ -exp_name ${dir} >> ${f}/mosbat_input/out_mosbat


cd /MoSBAT/MoSBAT-master/

file=$f/mosbat_input/*epocs*
echo $file;
bash MoSBAT.sh ${file: 13:-8} ${file} ${f}/mosbat_input/gt_pwm.txt dna 100 5000
mkdir -p $f/mosbat_output/${file: -30:-4}
cp /MoSBAT/MoSBAT-master/out/${file: 13:-8}/results.energy.correl.txt ${f}/mosbat_output/${file: -30:-4}/
cp /MoSBAT/MoSBAT-master/out/${file: 13:-8}/results.affinity.correl.txt ${f}/mosbat_output/${file: -30:-4}/
cd 'update path to where you have eval_mosbat.py script'
res={f}/mosbat_output/*epocs*
python3.6 eval_mosbat.py -a_add ${res}/results.affinity.correl.txt -e_add ${res}/results.energy.correl.txt -s_add ${res}/ >> ${res}/out_eval_mosbat






