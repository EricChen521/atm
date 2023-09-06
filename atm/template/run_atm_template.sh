#/bin/bash


cd $work_dir

if [ ! -f $work_dir/complex_0.xml ]; then

   ~/miniconda3/envs/atm-dev/bin/python $work_dir/${fep_type}_structprep.py  $work_dir/atom_${fep_type}.cntl|| exit

fi

if [ $gpu_num_per_pair -eq 1 ]
then
   echo -e "localhost,0:${device_index},1,CUDA,,/tmp" > $work_dir/nodefile
else

   echo -e "localhost,0:0,1,CUDA,,/tmp\nlocalhost,0:1,1,CUDA,,/tmp\nlocalhost,0:2,1,CUDA,,/tmp\nlocalhost,0:3,1,CUDA,,/tmp\nlocalhost,0:4,1,CUDA,,/tmp"  > $work_dir/nodefile
fi

~/miniconda3/envs/atm-dev/bin/python $atom_build_path/${fep_type}_explicit.py atom_${fep_type}.cntl || exit
