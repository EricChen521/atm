import logging
import shutil
import subprocess

import multiprocessing
from multiprocessing import Process, Pool
from pathlib import Path
from string import Template
from typing import Dict, List
from time import gmtime, strftime


import numpy as np

from Bio.PDB import PDBParser
from rdkit import Chem
from scipy.spatial import distance_matrix
from scipy.spatial.distance import cdist
from openmm.app import PDBFile
from openmm.unit import angstrom, nanometer
from atm.config import AtmConfig
from atm.forcefield import Gaff
from atm.utility import (
    ReverseAT,
    create_amber_system,
    create_xml_from_amber,
    create_xml_from_openff,
    multi_run_wrapper,
    generate_morph_graph,
    template_dir,
    tmp_cd,
)

LOGGER = logging.getLogger(__name__)


def calc_displ_vec(
    config: AtmConfig = None,
):
    """
    Return the optimal translational vector (x,y,z).

    Step 1: Find the smallest area center(point_1),
    Step 2: Move point_1 along the third direction with 9 A to point_2,
    Step 3: find the the smallest point in the third direction, point_3,
    Finally, the displacement_vec is obtained point_2 - point_3
    """
    protein_fpath = Path(config.protein_fpathname)
    ligand_dpath = Path(config.ligand_dpathname)

    protein_coords = _get_solute_coords(solute_fpath=protein_fpath)
    x_range = np.array([0, min(protein_coords[:, 0]), max(protein_coords[:, 0])])
    y_range = np.array([1, min(protein_coords[:, 1]), max(protein_coords[:, 1])])
    z_range = np.array([2, min(protein_coords[:, 2]), max(protein_coords[:, 2])])

    #print(f"system size: X {x_range}, Y {y_range}, Z {z_range}")

    small_area_center = np.array([0.0, 0.0, 0.0])

    axes = sorted([x_range, y_range, z_range], key=lambda v: v[2] - v[1])
    small_area_center[int(axes[0][0])] = np.mean(axes[0][1:])
    small_area_center[int(axes[1][0])] = np.mean(axes[1][1:])
    small_area_center[int(axes[2][0])] = axes[2][2] # max u coordinate
    u = int(axes[2][0])
    # point_1
    #print(f"small_area_center coordiante: {small_area_center}, with u axis: {u}")
    # point_2
    small_area_center[u] += 10.0 # max u coordinate + 10

    # find the smallest U in all ligands.

    ligandset_coords = _ligset_coords(
        ligands_dir=ligand_dpath,
    )

    ligands_coords = np.empty((1, 3), dtype=float)

    for _k, v in ligandset_coords.items():
        ligands_coords = np.concatenate((ligands_coords, v), axis=0)
     
    #print(ligands_coords)
    ligands_coords=np.array(ligands_coords).reshape((len(ligands_coords),3))

    sorted_u_index = ligands_coords[:, u].argsort()
    sorted_u_coords = ligands_coords[sorted_u_index]
    distant_lig_atom_coords = sorted_u_coords[0, :] # minimum u of all ligands
    #print(f"sorted ligand atom coordiates by {u}: {sorted_u_coords}")

    displ_vec = np.round((small_area_center - distant_lig_atom_coords), 2)
    print(f"displacemnet vector: {displ_vec}")
   
    return displ_vec



def _rename_ligname(source_fmol2: Path, dst_fmol2: Path, new_name: str) -> None:
    """
    Read `source_fmol2`, rename `LIG` to `new_name` and write to `dst_fmol2`
    """
    with open(dst_fmol2, "w") as fh:
        content_in = open(source_fmol2, "r").read()
        content_out = content_in.replace("LIG", new_name)
        fh.write(content_out)


def setup_atm_dir(  # noqa: C901
    config: AtmConfig,
) -> None:
    """
    Generate the xml/pdb unput for `free_energy` dir
    """
    LOGGER.info(f"setup atm dir for {config.atm_type}.")
    protein_fpath = Path(config.protein_fpathname).resolve()
    ligand_dpath = Path(config.ligand_dpathname).resolve()
    cofactor_fpath = (
        Path(config.cofactor_fpathname).resolve() if config.cofactor_fpathname else None
    )
    
    translation_vec = config.displ_vec

    with tmp_cd(config.work_dir):
        Path("free_energy").mkdir(parents=True, exist_ok=True)
        with tmp_cd("free_energy"):
            # use multiprocess as openff is slow
            procs = []
            parameters = []
            if config.atm_type == "rbfe":
                # check whether Morph.in is provided
                if not config.morph_fpathname:

                    assert config.morph_type in [
                        "star",
                        "lomap",
                    ], f"{config.morph_type} is not implemented yet."
                    morph_fpath = generate_morph_graph(
                        morph_type=config.morph_type,
                        ref_ligname=config.ref_ligname,
                        ligand_dpath=ligand_dpath,
                        morph_fpath=(Path(config.work_dir) / "Morph.in").resolve(),
                    )
                    config.morph_fpathname = str(morph_fpath)
                    config.write_to_yaml()

                pair_names = [
                    e
                    for e in open(config.morph_fpathname).read().split("\n")
                    if e.strip()
                ]
                # only do the pairs not done before by checking the existence of 'complex.pdb' file
                for pair_name in [e for e in pair_names if not (Path(e)/"complex.pdb").is_file()]:

                    Path(pair_name).resolve().mkdir(parents=True, exist_ok=True)
                    left_ligand_name, right_ligand_name = pair_name.split("~")
                    with tmp_cd(pair_name):
                        if config.forcefield_option == 'gaff2':
                            forcefield_dpath = Path(config.forcefield_dpathname)
                            lig_idx = 1
                            for lig in [left_ligand_name, right_ligand_name]:
                                lig_fpath = forcefield_dpath / lig / "vacuum.mol2"
                                lig_renamed_fpath = (
                                    forcefield_dpath / lig / f"L{lig_idx}.mol2"
                                )
                                lig_renamed_fpath.is_file() or _rename_ligname(
                                    lig_fpath, lig_renamed_fpath, f"L{lig_idx} "
                                )
                                shutil.copyfile(
                                    lig_fpath.parent / "vacuum.frcmod",
                                    lig_fpath.parent / f"L{lig_idx}.frcmod",
                                )
                                lig_idx += 1

                            solute_paths = (
                                [
                                    protein_fpath,
                                    forcefield_dpath / left_ligand_name / "L1.mol2",
                                    ReverseAT(
                                        forcefield_dpath / right_ligand_name / "L2.mol2"
                                    ),
                                    cofactor_fpath.parent / "cofactor.mol2",
                                ]
                                if cofactor_fpath
                                else [
                                    protein_fpath,
                                    forcefield_dpath / left_ligand_name / "L1.mol2",
                                    ReverseAT(
                                        forcefield_dpath / right_ligand_name / "L2.mol2"
                                    ),
                                ]
                            )
                            create_amber_system(
                                output_top_path="complex.prmtop",
                                output_crd_path="complex.inpcrd",
                                buffer_size=10,
                                translate=(2, translation_vec),
                                solute_paths=solute_paths,
                            )
                
                            create_xml_from_amber(
                                amber_top_fpath=Path("complex.prmtop"),
                                amber_crd_fapth=Path("complex.inpcrd"),
                                xml_out_fpath=Path("complex_sys.xml"),
                                pdb_out_fpath=Path("complex.pdb"),
                                is_hmr=True if config.dt == 0.04 else False,
                            )

                        elif config.forcefield_option == "openff":
                                if not Path("./complex_sys.xml").is_file():
                                    parameters.append(
                                        [
                                            protein_fpath,
                                            ligand_dpath / left_ligand_name / "ligand.sdf",
                                            translation_vec,
                                            Path("./complex_sys.xml").resolve(),
                                            Path("./complex.pdb").resolve(),
                                            ligand_dpath / right_ligand_name / "ligand.sdf",
                                            cofactor_fpath if cofactor_fpath else None,
                                            True if config.dt == 0.04 else False,
                                        ]
                                    )

            elif config.atm_type == "abfe":
                ligand_names = [
                    e.name for e in ligand_dpath.iterdir() if e.is_dir()
                ]
                for ligand_name in ligand_names:
                    Path(ligand_name).mkdir(parents=True, exist_ok=True)
                    with tmp_cd(ligand_name):

                        if config.forcefield_option in ["gaff2"]:
                            forcefield_dpath = Path(config.forcefield_dpathname)
                            lig_fpath = forcefield_dpath / ligand_name / "vacuum.mol2"
                            lig_renamed_fpath = (
                                forcefield_dpath / ligand_name / f"L1.mol2"
                            )
                            lig_renamed_fpath.is_file() or _rename_ligname(
                                lig_fpath, lig_renamed_fpath, f"L1"
                            )
                            shutil.copyfile(
                                lig_fpath.parent / "vacuum.frcmod",
                                lig_fpath.parent / f"L1.frcmod",
                            )

                            create_amber_system(
                                output_top_path="complex.prmtop",
                                output_crd_path="complex.inpcrd",
                                buffer_size=10,
                                solute_paths=[
                                    protein_fpath,
                                    forcefield_dpath / ligand_name / "L1.mol2",
                                    cofactor_fpath.parent / "cofactor.mol2",
                                ]
                                if cofactor_fpath
                                else [
                                    protein_fpath,
                                    forcefield_dpath / ligand_name / "L1.mol2",
                                ],
                                # Translate the ligand to solvent
                                # pass the negative translation_vec to downstream
                                translate=(1, translation_vec),
                            )
                            create_xml_from_amber(
                                amber_top_fpath=Path("complex.prmtop"),
                                amber_crd_fapth=Path("complex.inpcrd"),
                                xml_out_fpath=Path("complex_sys.xml"),
                                pdb_out_fpath=Path("complex.pdb"),
                                is_hmr = True if config.dt == 0.04 else False,
                            )
                        elif config.forcefield_option in ["openff"]:
                            if not Path("complex_sys.xml").is_file():

                                create_xml_from_openff(
                                    protein_fpath=protein_fpath,
                                    lig1_fpath=ligand_dpath
                                    / ligand_name
                                    / "ligand.sdf",
                                    translation_vec=translation_vec,
                                    xml_out_fpath="./complex_sys.xml",
                                    pdb_out_fpath="./complex.pdb",
                                    cofactor_fpath=cofactor_fpath
                                    if cofactor_fpath
                                    else None,
                                    is_hmass=True if config.dt == 0.004 else False,
                                )
            if parameters:

                cpu_number = multiprocessing.cpu_count()
                proc_num = len(parameters)
                print(f"cpu number: {cpu_number}, pair number: {proc_num}")
                print(f"start openff calculation at: {strftime('%Y-%m-%d %H:%M:%S', gmtime())}")
                with Pool(min(cpu_number,proc_num)) as pool:
                    result=pool.map(multi_run_wrapper,[(p[0],p[1],p[2],p[3],p[4],p[5],p[6]) for p in parameters])
                print(f"complete openff calculation at: {strftime('%Y-%m-%d %H:%M:%S', gmtime())}")
                #pool = multiprocessing.Pool(cpu_number)
                #pool.map(func=create_xml_from_openff,interable=parameters)
                """
                for params in parameters:
                    # logging.info(params)
                    logging.info(f"running openff for pair {params[1].parent.name}~{params[5].parent.name}.")
                    proc = Process(
                        target=create_xml_from_openff,
                        args=(
                            params[0],
                            params[1],
                            params[2],
                            params[3],
                            params[4],
                            params[5],
                            params[6],
                        ),
                    )
                    procs.append(proc)
                    proc.start()
                for proc in procs:
                    proc.join()
                """



def _get_solute_coords(solute_fpath: Path):  # pdb or sdf file format
    """
    Return N*3 array for solute coordinates
    """
    assert solute_fpath.suffix in [".sdf", ".pdb"], f"{solute_fpath} is not supported."
    if solute_fpath.suffix == ".sdf":
        mol = Chem.SDMolSupplier(str(solute_fpath), removeHs=False)[0]
    else:
        mol = Chem.rdmolfiles.MolFromPDBFile(str(solute_fpath), removeHs=False)

    conf = mol.GetConformer()
    N_atoms = mol.GetNumAtoms()
    coords = np.zeros((N_atoms, 3))
    for row in range(N_atoms):
        coords[row] = np.array(list(conf.GetAtomPosition(row)))

    return coords

    
def _ligset_coords(ligands_dir: Path) -> Dict:
    """
    Return {ligand_name(str): coords (np.array)} for all ligands under `ligands_dir`.
    """
    set_coords = {}
    ligand_fpaths = [e / "ligand.sdf" for e in ligands_dir.iterdir() if e.is_dir()]
    for ligand_fpath in ligand_fpaths:
        coords = _get_solute_coords(ligand_fpath)
        set_coords.update({ligand_fpath.parent.name: coords})

    return set_coords


def parse_protein(complex_pdb_fpath: Path= None,
                  relaxed_res_ids: List[int]=None ,
                  vsite_radius: float = 6.0,
                  #in abfe, L1 needs to transfer back to get the viste residues
                  displ_vec: List[float]=[0,0,0]) -> Dict:
    
    """
        Parse the complex.pdb to return:
            "CA_ids": the CA atom index (0 indexed)
            "restrianed_CA_ids": the CA atom index to be restrianed in alchemical MD
            "vsite_CA_ids": The CA atom index in vsite
            "N_atom": The first atom index of L1
            "ligoffset": The vector from ligand 1 center to vsite center
    
    """
    # get the first ligand coordinate

    pdb = PDBFile(str(complex_pdb_fpath))
    positions = pdb.positions
    ommtopology = pdb.topology
    if relaxed_res_ids:
        print(f"relaxed_res_ids: {relaxed_res_ids}")
    

    CA_ids = []
    CA_positions = []
    vsite_positions =[]
    vsite_CA_ids = []
    L1_ids =[]
    L1_positions =[]
    relax_CA_ids=[]
    restrained_CA_ids =[]

    # Get the atom index for 'L1' ligand
    for atom in ommtopology.atoms():
        if atom.residue.name == "L1":
            L1_ids.append(atom.index)
            
            L1_positions.append([(positions[atom.index]/angstrom)[0],
                                 (positions[atom.index]/angstrom)[1],
                                (positions[atom.index]/angstrom)[2]])
            
            
        elif atom.name == "CA":
            CA_ids.append(atom.index)
            CA_positions.append([(positions[atom.index]/angstrom)[0],
                                 (positions[atom.index]/angstrom)[1],
                                 (positions[atom.index]/angstrom)[2]])
            
            if relaxed_res_ids:
                if int(atom.residue.id) in relaxed_res_ids:
                    relax_CA_ids.append(atom.index)
          
    L1_coords = np.array(L1_positions)  
    for i in range(len(CA_ids)):
        CA_coords = np.array(CA_positions[i]).reshape((1,3))
        min_distance = np.amin(distance_matrix(CA_coords,L1_coords))
        #print(min_distance)
        
        if min_distance <= vsite_radius:
            vsite_CA_ids.append(i)
            vsite_positions.append(CA_positions[i])
    
    vsite_coords = np.array(vsite_positions)
    
    dx = round(np.mean(L1_coords[:,0]) - np.mean(vsite_coords[:,0]),2)
    dy = round(np.mean(L1_coords[:,1]) - np.mean(vsite_coords[:,1]),2)
    dz = round(np.mean(L1_coords[:,2]) - np.mean(vsite_coords[:,2]),2)
    
    
    ligoffset = [dx,dy,dz]
    
    print(f"ligoffset vector: {ligoffset}")
    
    restrained_CA_ids=sorted(list(set(CA_ids) - set(relax_CA_ids)))
    print(f"relax CA ids: {relax_CA_ids}")       
    result = {"CA_ids": CA_ids,"restrained_CA_ids": restrained_CA_ids, "vsite_CA_ids": vsite_CA_ids, "N_atoms": L1_ids[0],
                    "ligoffset": ligoffset}

    return result

def get_alignment(config: AtmConfig) -> Dict:
    """
    Return Dict with key = 'ligand_name`,
        value= dict {'aligned_atom_ids': [id-1,id_2,id_3], 'N_atoms': N}
    """

    result = {}
    ligands_coords = _ligset_coords(ligands_dir=Path(config.ligand_dpathname))
    
    if config.atm_type == "rbfe":
        LOGGER.info(f"all ligand names: {ligands_coords.keys()}, ref ligand name: {config.ref_ligname}.")
        ref_lig_coords = ligands_coords[str(config.ref_ligname)]
        ref_lig_N_atoms = ref_lig_coords.shape[0]
        result.update(
            {
                config.ref_ligname: {
                    "align_atom_ids": config.ref_alignidx,
                    "N_atoms": ref_lig_N_atoms,
                }
            }
        )

        for lig_name, lig_coords in ligands_coords.items():
            if lig_name != config.ref_ligname:
                dist_matrix = cdist(ref_lig_coords, lig_coords)
                nearest_ids = np.argmin(dist_matrix, axis=1)
                # the ref_align_idx starts from 1, be careful about the index
                lig_aligment_ids = [nearest_ids[i - 1] + 1 for i in config.ref_alignidx]
                result.update(
                    {
                        lig_name: {
                            "align_atom_ids": lig_aligment_ids,
                            "N_atoms": lig_coords.shape[0],
                        }
                    }
                )
    if config.atm_type == "abfe":
        
        for lig_name, lig_coords in ligands_coords.items():
            result.update(
                {lig_name:
                    {
                        "align_atoms_ids": [1,2,3], # not used in abfe 
                        "N_atoms": lig_coords.shape[0],
                    }
                }
            )
        
        
        
    return result


def update_scripts(
    config: AtmConfig,
    protein_info: Dict,
    alignment_result: Dict,
):
    """
    Update the strucprep*.py and run*.py scripts for each perterbuation.

    """
    free_energy_dpath = Path(f"{config.work_dir}/free_energy")
    N_protein = protein_info["N_atoms"]
    CA_ids = protein_info["CA_ids"]
    restrained_CA_ids = protein_info["restrained_CA_ids"]
    vsite_CA_ids = protein_info["vsite_CA_ids"]

    # sanity check to ensure the CA_ids match complex.pdb CA atoms

    complex_fpath = next(free_energy_dpath.iterdir())/"complex.pdb"
    complex_stuct = open(complex_fpath,"r").readlines()
    for line in complex_stuct:
        if "HETAM" in line or "ATOM" in line:
            if line.split()[2] == "CA":
                CA_id = int(line.split()[1]) - 1 
                assert CA_id in CA_ids, f"CA_id: {CA_id} of {complex_fpath} not found in atom.cntl file,"+ \
                    " double check input sturcture file for missing atoms!"
        
    LOGGER.info("CA ids check passed")
    job_id = 0
    max_sample_num = int(
        (config.sim_time * 1000) / (config.dt * config.print_energy_interval)
    )

    #update the analyze template file
    with open(free_energy_dpath.parent/"analyze.sh", "w") as fh:
        analyze_template = Template(
            open(template_dir()/"analyze.sh","r").read()
        ).safe_substitute(
            uwhat_script_pathname=str(template_dir()/"uwham_analysis.R"),
            diffnet_script_pathname=str(template_dir()/"diffnet_atm.py"),
            free_energy_dir = str(free_energy_dpath),
            start_frame_index = int(max_sample_num/3),
            final_frame_index = max_sample_num,
            atm_dev_env = f"{Path(config.atm_pythonpathname).parents[3]}/bin/activate atm-dev"
        )
        fh.write(str(analyze_template))

    for perturbation_dirname in [
        e.name for e in free_energy_dpath.iterdir() if e.is_dir()
    ]:

        with open(
            free_energy_dpath
            / perturbation_dirname
            / f"{config.atm_type}_structprep.py",
            "w",
        ) as fh:
            structprep_template = Template(
                open(template_dir() / f"{config.atm_type}_structprep.py", "r").read()
            ).substitute(
                device_index=config.gpu_devices[job_id % len(config.gpu_devices)],
                atom_build_path = config.atom_build_pathname,
            )
            fh.write(str(structprep_template))

        with open(free_energy_dpath / perturbation_dirname / "run_atm.sh", "w") as fh:
            run_atm_template = Template(
                open(template_dir() / "run_atm_template.sh", "r").read()
            ).substitute(
                pair_name = perturbation_dirname,
                work_dir=str(free_energy_dpath / perturbation_dirname),
                fep_type=config.atm_type,
                partition=config.partition,
                gres=config.gres,
                gpu_num_per_pair=1,
                atom_build_path=config.atom_build_pathname,
                atm_pythonpath = config.atm_pythonpathname,
                device_index=config.gpu_devices[job_id % len(config.gpu_devices)],
            )
            fh.write(str(run_atm_template))

            if config.atm_type == "abfe":
                lig_atom_ids = list(
                    range(
                        N_protein,
                        N_protein + alignment_result[perturbation_dirname]["N_atoms"],
                    )
                )
                with open(
                    free_energy_dpath / perturbation_dirname / "atom_abfe.cntl", "w"
                ) as fh:
                    atom_template = Template(
                        open(template_dir() / "atom_abfe_template.cntl", "r").read()
                    ).substitute(
                        work_dir=str(free_energy_dpath / perturbation_dirname),
                        ligoffset=",".join(map(str,protein_info["ligoffset"])),
                        # abfe with gaff2, ligand already in the solvent, use the oppsite displacemnet to go back to binding site
                        # abfe with openff, ligand is in the binding site, use the normal displacement to go to solvent
                        displ=",".join(map(str, [-i for i in config.displ_vec] if config.forcefield_option =="gaff2" else config.displ_vec)),
                        lig_atoms=",".join(map(str, lig_atom_ids)),
                        rcpt_cm_atoms=",".join(map(str, vsite_CA_ids)),
                        restrained_atoms=",".join(map(str, restrained_CA_ids)),
                        max_sample_num=max_sample_num,
                        exchange_interval=config.exchange_interval,
                        print_energy_interval=config.print_energy_interval,
                        print_traj_interval=config.print_traj_interval,
                        kforce_vsite=config.kforce_vsite,
                        vsite_radius=config.vsite_radius,
                        kforce_displ=config.kforce_displ,
                        kforce_theta=config.kforce_theta,
                        kforce_psi=config.kforce_psi,
                    )
                    fh.write(atom_template)

            if config.atm_type == "rbfe":
                lig1_name, lig2_name = perturbation_dirname.split("~")
                lig1_ids = list(
                    range(N_protein, N_protein + alignment_result[lig1_name]["N_atoms"])
                )
                lig1_alignment_ids = [
                    int(e) - 1 for e in alignment_result[lig1_name]["align_atom_ids"]
                ]
                lig2_ids = list(
                    range(
                        lig1_ids[-1] + 1,
                        lig1_ids[-1] + 1 + alignment_result[lig2_name]["N_atoms"],
                    )
                )
                lig2_alignment_ids = [
                    int(e) - 1 for e in alignment_result[lig2_name]["align_atom_ids"]
                ]

                with open(
                    free_energy_dpath / perturbation_dirname / "atom_rbfe.cntl", "w"
                ) as fh:
                    atom_template = Template(
                        open(template_dir() / "atom_rbfe_template.cntl", "r").read()
                    ).substitute(
                        work_dir=str(free_energy_dpath / perturbation_dirname),
                        displ=",".join(map(str, config.displ_vec)),
                        lig1_atoms=",".join(map(str, lig1_ids)),
                        lig2_atoms=",".join(map(str, lig2_ids)),
                        # local ligand index for alignment
                        refatoms_lig1=",".join(map(str, lig1_alignment_ids)),
                        refatoms_lig2=",".join(map(str, lig2_alignment_ids)),
                        rcpt_cm_atoms=",".join(map(str, vsite_CA_ids)),
                        restrained_atoms=",".join(map(str, restrained_CA_ids)),
                        max_sample_num=max_sample_num,
                        exchange_interval=config.exchange_interval,
                        print_energy_interval=config.print_energy_interval,
                        print_traj_interval=config.print_traj_interval,
                        kforce_vsite=config.kforce_vsite,
                        vsite_radius=config.vsite_radius,
                        kforce_displ=config.kforce_displ,
                        kforce_theta=config.kforce_theta,
                        kforce_psi=config.kforce_psi,
                    )
                    fh.write(atom_template)
        job_id += 1


def submit_job(is_slurm: bool, free_energy_dpath: Path) -> None:
    """
    Submit the atm job to the locoalhost or slurm cluster.
    """
    jobs = {}
    for perturbation_dir in [e for e in free_energy_dpath.iterdir() if e.is_dir()]:

        with open(f"{perturbation_dir}/atm.log", "w") as flog:
            if is_slurm:
                cmdline = ["sbatch", "--parsable",f"{perturbation_dir}/run_atm.sh"]
            else:
                cmdline = ["/bin/bash", f"{perturbation_dir}/run_atm.sh"]
            job = subprocess.Popen(
                cmdline,
                stderr=subprocess.PIPE,
                stdout=flog,
                cwd=perturbation_dir,
            )
            jobs.update({perturbation_dir: job})

    LOGGER.info(f"Running ATM calculation...")
    for pertub_dir, procs in jobs.items():
        stdout, stderr = procs.communicate()
     
        if stderr:
            with open(pertub_dir / "err.log", "w") as fh:
                fh.write(stderr.decode("utf-8"))
        if stdout:
            with open(pertub_dir/"stdout.log","w") as fh:
                fh.write(stdout.decode("utf-8"))


        return stdout, stderr
