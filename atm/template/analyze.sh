#!/bin/bash

source ${atm_dev_env}
UWHAT_SCRIPT=${uwhat_script_pathname}
cd $free_energy_dir
echo "name0,name1,ddG_convergence,error,start_frame,end_frame" >> ../ddG_convergence.dat
echo "name0,name1,ddG_value,error,start_frame,end_frame" >> ../atm_results.dat
for pair in *
do
	if [ -d $pair ]; then
		cd $pair
		
		
		name1=${pair%%~*}
		name2=${pair##*~}
		ddGs=()
		# get 10 data point to check convergence for each pair
		step=$(($final_frame_index/10))
		for i in  {1..11}
		do
			step_frame_index=$(($step*$i))
			R CMD BATCH -complex -$start_frame_index -$step_frame_index $UWHAT_SCRIPT
			result=($(grep -r "^DDGb =" uwham_analysis.Rout))
			ddG=$(echo ${result[2]} | xargs printf "%.2f")
			ddGs+=($ddG)
		done

		uncertanity=$(echo ${result[4]} | xargs printf "%.2f")
		end_frame=$(echo ${result[7]})
		printf -v ddGs_str '>%s' "${ddGs[@]}"
		ddGs_str=${ddGs_str:1} 
		echo "$name1,$name2,$ddGs_str,$uncertanity,$start_frame_index,$end_frame" >> ../../ddG_convergence.dat
		echo "$name1,$name2,$ddG,$uncertanity,$start_frame_index,$end_frame" >> ../../atm_results.dat
		cd ..
	fi
done