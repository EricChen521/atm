load complex.pdb
load $traj,complex
remove solvent
remove inorganic
extract L1, resname L1
extract L2, resname L2
translate $reversed_displ, L2, camera=0, state=0
util.cbac L2
zoom