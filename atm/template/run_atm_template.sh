#!/bin/bash

#SBATCH -J $pair_name
#SBATCH --output=atm.log
#SBATCH --error=atm.err
#SBATCH --partition=${partition}
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:${gres}
#SBATCH --cpus-per-task=2
#SBATCH --no-requeue
#SBATCH -t 72:00:00


cd $work_dir

if [ ! -f $work_dir/complex_0.xml ]; then

   ${atm_pythonpath} $work_dir/${fep_type}_structprep.py  $work_dir/atom_${fep_type}.cntl|| exit

fi

if [ -f $work_dir/nodefile ]; then
   rm $work_dir/nodefile
fi

if [ $gpu_num_per_pair -eq 1 ]
then
   echo -e "localhost,0:${device_index},1,CUDA,,/tmp" > $work_dir/nodefile

else
   i=0
   while [ $dollar_i -lt $gpu_num_per_pair ]
   do
      echo -e "localhost,0:$dollar_i,1,CUDA,,/tmp"  >> $work_dir/nodefile
      i=$dollar((i + 1 ))
   done
fi

${atm_pythonpath} $atom_build_path/${fep_type}_production.py atom_${fep_type}.cntl || exit
