load complex.pdb, eq
load complex_mdlambda_wrap.xtc, eq
delete_state eq,1
extract eq_L1, resname L1 and eq
extract eq_L2, resname L2 and eq
util.cbac eq_L2
load complex.pdb, r0
load complex_r0_wrap.xtc, r0
delete_state r0,1
extract r0_L1, resname L1 and r0
extract r0_L2, resname L2 and r0
label r0_L1 and id $p1_id, "p1"
label r0_L1 and id $p2_id, "p2"
label r0_L1 and id $p3_id, "p3"
label r0_L2 and id $P1_id, "P1"
label r0_L2 and id $P2_id, "P2"
label r0_L2 and id $P3_id, "P3"
util.cbay r0_L2
remove inorganic
remove solvent
smooth
zoom