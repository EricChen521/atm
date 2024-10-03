"""
Define useful decorators and functions
"""

import contextlib
import glob
import os
import re
import shutil
import subprocess
import time

from pathlib import Path
from typing import List, Optional, Tuple, Union

from openff.toolkit.topology import Molecule
from openmm import XmlSerializer, Vec3
from openmm.app import (
    PME,
    AmberInpcrdFile,
    AmberPrmtopFile,
    ForceField,
    HBonds,
    Modeller,
    PDBFile,
)
from openmm.unit import amu, angstrom, nanometer
from openmmforcefields.generators import SMIRNOFFTemplateGenerator


@contextlib.contextmanager
def tmp_cd(dir: Optional[Union[str, Path]] = None):
    origin = os.getcwd()
    try:
        os.chdir(dir)
        yield
    finally:
        os.chdir(origin)


def tprint(message: str = None):
    print(f"{time.ctime(): {message}}")


def skip_if(files_exist: Tuple[str] = (), dirs_exist: Tuple[str] = ()):
    """
    A decrator to skip function if certeain files/dirs are found.
    """

    def decorated_func(func):

        for file in files_exist:
            if not Path(f"./{file}").is_file():
                return func
        for dir in dirs_exist:
            if not Path(f"./{dir}").is_dir():
                return func
        print("function is skipped because files/dirs are found.")

    return decorated_func


def str_duration(time_duration: Optional[float]) -> str:
    """
    Return the human readiable durtion in format of <xh y' z''>
    """
    if time_duration is None:
        return "N/A"
    hour = int(time_duration / 3600.0)
    minute = int((time_duration - hour * 3600) % 60.0)
    second = int(time_duration - hour * 3600.0 - minute * 60.0)
    return f"{hour}h {minute}' {second}\""


def subrun(
    cmd: List[str],
    stop_on_failure: bool = True,
    **kwarg,
) -> subprocess.CompletedProcess:

    tprint(f"Running subprocess command: {cmd}")
    start_time = time.time()
    ret = subprocess.run(cmd, **kwarg)
    end_time = time.time()
    tprint(f"Subprocess is done. Duratin: {str_duration(end_time-start_time)}")
    if ret.returncode:
        log = ((ret.stdout or bytes())) + (ret.srderr or bytes()).decode("utf-8")
        tprint(log)
        if stop_on_failure:
            ret.check_returncode()

    return ret


def copy_foldered_files(
    src_path: Union[Path, str],
    dst_path: Union[Path, str],
    fname: str,
    skip_empty_files=True,
):
    """
    "Foldered files" refers to files organized as follows:

      src
      ├── folder0
      │   ├── data.dat
      │   └── ligand.sdf
      │
      ├── folder1
      │   ├── data.dat
      │   └── ligand.sdf
      │
      ├── folder2
      │   ├── data.dat
      │   └── ligand.sdf

    This function will copy files from `src_path` to `dst_path`, preserving the original
    folder structure. For example, a call like the following:

    >>> copy_foldered_files("src", "dst", "ligand.sdf")

    will create a new folder called "dst" with subfolder and files organized as follows:

      dst
      ├── folder0
      │   └── ligand.sdf
      │
      ├── folder1
      │   └── ligand.sdf
      │
      ├── folder2
      │   └── ligand.sdf

    If a subfolder of `src_path` doesn't have the designated file `fname`, it will NOT
    be created under `dst_path`. If `skip_empty_files` is `True`, a zero-sized `fname`
    file will not be copied and the subfolder will not be created under `dst_path`.
    If `fname` doesn't exist in any of the subfolders of `src_path`, `dst_path` will NOT
    be created.
    """
    src_path = Path(src_path)
    dst_fname = str(os.path.abspath(dst_path))
    src_fnames = glob.glob(str(src_path / "*" / fname))
    i = len(str(src_path))
    for src_fname in src_fnames:
        if skip_empty_files and not src_fname.is_file():
            continue
        dst_path = Path(dst_fname + src_fname[i:])
        os.makedirs(dst_path.parent, exist_ok=True)
        shutil.copyfile(src_fname, dst_path)


def data_dir() -> Path:
    return (Path(__file__).parent / "tests" / "data").resolve()


def template_dir() -> Path:
    return (Path(__file__).parent / "template").resolve()


FFQUICK_AT = (
    "aa,ab,ac,ad,ae,af,ag,ah,ai,aj,ak,an,ao,ap,aq,as,at,au,av,aw,ax,ay,az,ba,"
    "bb,bc,bd,be,bf,bg,bh,bi,bj,bk,bl,bm,bn,bo,bp,bq,bs,bt,bu,bv,bw,bx,by,bz,"
    "cb,ci,cj,ck,cm,cn,co,cr,ct,cw,da,db,dc,dd,de,df,dg,dh,di,dj,dk,dl,dm,dn,"
    "do,dp,dq,dr,ds,dt,dv,dw,dx,dy,dz,ea,eb,ec,ed,ee,ef,eg,eh,ei,ej,ek,el,em,"
    "en,eo,eq,er,es,et,eu,ev,ew,ex,ey,ez,fa,fb,fc,fd,ff,fg,fh,fi,fj,fk,fl,fm,"
    "fn,fo,fp,fq,fr,fs,ft,fu,fv,fw,fx,fy,fz,ga,gb,gc,gd,ge,gf,gg,gh,gi,gj,gk,"
    "gl,gm,gn,go,gp,gq,gr,gs,gt,gu,gv,gw,gx,gy,gz,hb,hd,he,hf,hg,hh,hi,hj,hk,"
    "hl,hm,hq,hr,ht,hu,hv,hy,hz,ia,ib,ic,id,ie,if,ig,ih,ii,ij,ik,il,im,in,io,"
    "ip,iq,ir,is,it,iu,iv,iw,ix,iy,iz,ja,jb,jc,jd,je,jf,jg,jh,ji,jj,jk,jl,jm,"
    "jn,jo,jp,jq,jr,js,jt,ju,jv,jw,jx,jy,jz,ka,kb,kc,kd,ke,kf,kg,kh,ki,kj,kk,"
    "kl,km,kn,ko,kp,kq,kr,ks,kt,ku,kv,kw,kx,ky,kz,la,lb,lc,ld,le,lf,lg,lh,lj,"
    "lk,ll,lm,ln,lo,lq,lr,ls,lt,lu,lv,lw,lx,ly,lz,ma,mb,mc,md,me,mf,mh,mi,mj,"
    "mk,ml,mm,mn,mo,mp,mq,mr,ms,mt,mu,mv,mw,mx,my,mz,ng,nr,nw,oa,ob,oc,od,oe,"
    "of,og,oi,oj,ok,ol,om,on,oo,or,ot,ou,ov,ox,oy,oz,pa,pg,ph,pi,pj,pk,pl,pm,"
    "pn,po,pp,pq,pr,ps,pt,pu,pv,pw,pz,qa,qb,qc,qd,qe,qf,qg,qh,qi,qj,qk,ql,qm,"
    "qn,qo,qp,qq,qr,qs,qt,qu,qv,qw,qx,qy,qz,ra,rb,rc,rd,re,rf,rg,rh,ri,rj,rk,"
    "rl,rm,rn,ro,rp,rq,rr,rs,rt,ru,rv,rw,rx,ry,rz,sa,sb,sc,sd,se,sf,sg,sj,sk,"
    "sl,sm,sn,so,sr,st,su,sv,sw,sz,ta,tb,tc,td,te,tf,tg,th,ti,tj,tk,tl,tm,tn,"
    "to,tp,tq,tr,ts,tt,tu,tv,tw,tx,ty,tz,ua,ub,uc,ud,ue,uf,ug,uh,ui,uj,uk,ul,"
    "um,un,uo,up,uq,ur,us,ut,uu,uv,uw,ux,uy,uz,va,vb,vc,vd,ve,vf,vg,vh,vi,vj,"
    "vk,vl,vm,vn,vo,vp,vq,vr,vs,vt,vu,vv,vw,vx,vy,vz,wa,wb,wc,wd,we,wf,wg,wh,"
    "wi,wj,wk,wl,wm,wn,wo,wp,wq,wr,ws,wt,wu,wv,ww,wx,wy,wz,xa,xb,xc,xd,xe,xf,"
    "xg,xh,xi,xj,xk,xl,xm,xn,xo,xp,xq,xr,xs,xt,xu,xv,xw,xx,xy,xz,ya,yb,yc,yd,"
    "ye,yf,yg,yh,yi,yj,yk,yl,ym,yn,yo,yp,yq,yr,ys,yt,yu,yv,yw,yx,yy,yz,za,zb,"
    "zc,zd,ze,zf,zg,zh,zi,zj,zk,zl,zm,zo,zp,zq,zr,zs,zt,zu,zv,zw,zx,zy,zz"
).split(",")

AT_REMAP = dict(zip(FFQUICK_AT, list(reversed(FFQUICK_AT))))


class ReverseAT(Path):
    _flavour = Path()._flavour

    def resolve(self):
        return ReverseAT(super().resolve())


def _reverse_AT(fpath: Union[Path, str]) -> Path:
    """
    Replace the atomtype with the reserved one to avoid collisions in tleap
    """

    content_in = open(fpath, "r").read()
    pattern = re.compile(r"\b(" + "|".join(list(AT_REMAP.keys())) + r")\b")
    content_out = pattern.sub(lambda x: AT_REMAP[x.group()], content_in)
    out_fpath = Path(fpath).parent / f"reversed_{fpath.name}"

    with open(out_fpath, "w") as fh:
        fh.write(content_out)

    return out_fpath


# FIXME Add this function to parmedizer
def _correct_parm(parm_file):
    """
    Correct topology file that generatd from FFquick
    """

    import parmed

    from parmed.periodic_table import AtomicNum

    mol = parmed.load_file(parm_file)

    for a in mol.atoms:
        element_name = parmed.periodic_table.element_by_mass(a.mass)
        if a.atomic_number != AtomicNum[element_name]:
            a.atomic_number = AtomicNum[element_name]
            a.element = AtomicNum[element_name]

    mol.save(parm_file, overwrite=True)


def create_amber_system(
    output_top_path: Union[Path, str],
    output_crd_path: Union[Path, str],
    solute_paths: Union[List[Path], List[str]],
    translate: Tuple[int, Tuple[float, float, float]] = (),
    box_shape: str = "Box",
    solvent_model: str = "TIP3P",
    buffer_size: float = 0.0,
    ions: Tuple = ("K+", "Cl-"),
) -> Tuple[Path, Path]:
    """
    Return:
        Tuple[Path,Path]:
            the file paths for topology and coordinate (amber format) of the system
    """
    tleap_in = [
        "source leaprc.protein.ff14SB",
        "source leaprc.phosaa14SB",
        "source leaprc.DNA.OL15",
        "source leaprc.RNA.OL3",
        "source leaprc.lipid21",
        f"source leaprc.water.{solvent_model.lower()}",
        "source leaprc.gaff2",
    ]
    to_be_cleanedup = []
    for i, solute_path in enumerate(solute_paths):
        solute_path = Path(solute_path) if isinstance(solute_path, str) else solute_path
        solute_path = solute_path.resolve()
        leaprc_path = solute_path.parent / "leaprc_header"
        frcmod_path = solute_path.with_suffix(".frcmod")

        if isinstance(solute_path, ReverseAT):
            solute_path = _reverse_AT(solute_path)
            leaprc_path = _reverse_AT(leaprc_path)
            frcmod_path = _reverse_AT(frcmod_path)
            to_be_cleanedup += [solute_path, leaprc_path, frcmod_path]

        leaprc_path.is_file() and tleap_in.append(f"source {leaprc_path}")
        frcmod_path.is_file() and tleap_in.append(f"loadAmberParams {frcmod_path}")
        fname_suffix = solute_path.suffix
        assert fname_suffix in [
            ".pdb",
            ".mol2",
        ], f"Structure file format is not support for {solute_path}."
        tleap_in.append(f"solute_{i}=load{fname_suffix[1:]} {solute_path}")

    if translate:
        solute_id, (x, y, z) = translate
        tleap_in.append(f"translate solute_{solute_id} {{{x} {y} {z}}}")

    solutes = "solute_" + " solute_".join(map(str, range(len(solute_paths))))
    tleap_in.append(f"sys=combine {{{solutes}}}")

    if buffer_size:
        tleap_in.append(f"solvate{box_shape} sys {solvent_model}BOX {buffer_size}")

    for ion in ions:
        # addions2 places ions in edges but might cause crashes in MD
        tleap_in.append(f"addions sys {ion} 0")

    tleap_in.append("check sys")
    tleap_in.append(f"saveamberparm sys {output_top_path} {output_crd_path}")
    output_crd_path = Path(output_crd_path).resolve()
    output_top_path = Path(output_top_path).resolve()
    tleap_in.append(
        f"savePdb sys "
        f"{output_top_path.parent/output_top_path.name.split('.')[0]}.pdb"
    )
    tleap_in.append("quit")

    with open(f"{output_top_path.parent}/tleap.in", "w") as fh:
        fh.write("\n".join(tleap_in))

    cmd = ["tleap", "-f", f"{output_top_path.parent}/tleap.in"]
    with tmp_cd(output_top_path.parent):
        subprocess.run(cmd, check=True, text=True, capture_output=True)
        _correct_parm(str(output_top_path))

    for tmp_path in to_be_cleanedup:
        tmp_path.unlink()

    return output_top_path, output_crd_path


def create_xml_from_amber(
    amber_top_fpath: Path,
    amber_crd_fapth: Path,
    xml_out_fpath: Path,
    pdb_out_fpath: Path,
    is_hmr: bool = False,
) -> Tuple[Path]:
    """
    Create xml file from the amber system input.

    Return (xml_out_fpath, pdb_out_fpath) for the system
    """

    prmtop = AmberPrmtopFile(str(amber_top_fpath))
    inpcrd = AmberInpcrdFile(str(amber_crd_fapth))
    system = prmtop.createSystem(
        nonbondedMethod=PME,
        nonbondedCutoff=0.9 * nanometer,
        constraints=HBonds,
        hydrogenMass=1.5 * amu if is_hmr else 1 * amu,
    )

    with open(xml_out_fpath, "w") as fh:
        fh.write(XmlSerializer.serialize(system))

    with open(pdb_out_fpath, "w") as fh:
        PDBFile.writeFile(prmtop.topology, inpcrd.positions, fh)


def create_xml_from_openff(
    protein_fpath: Path,  # pdb format
    lig1_fpath: Path,  # sdf format
    translation_vec: Tuple[float],
    xml_out_fpath: Path,
    pdb_out_fpath: Path,
    lig2_fpath: Optional[Path]=None,  # sdf format
    cofactor_fpath: Optional[Path] = None,
    protein_ff: str = "amber14-all.xml",
    solvent_ff: str = "amber14/tip3p.xml",
    ligand_ff: str = "openff-2.2.0",
    is_hmass: bool = False,
) -> Tuple[Path]:
    """
    Create xml file using amber14(protein)/openff(small molecules) forcefield.
    Retrun: Tuple (xml_out_fpath, pdb_out_fpath)
    """
    ff = ForceField(protein_ff, solvent_ff)

    protein_pdb = PDBFile(str(protein_fpath))
    protein_coords = protein_pdb.positions
    protein_mmtop = protein_pdb.topology

    system = Modeller(protein_mmtop, protein_coords)

    small_mols = []  # ligands and cofactors
    lig1_mol = Molecule.from_file(
        str(lig1_fpath), file_format="SDF", allow_undefined_stereo=True
    )
    lig1_mmtop = lig1_mol.to_topology().to_openmm(ensure_unique_atom_names=True)
    # print(next(lig1_mmtop.residues()).name)
    next(lig1_mmtop.residues()).name = "L1"
    lig1_coords = lig1_mol.conformers[0].to("angstrom").magnitude
    small_mols.append(lig1_mol)
    # abfe
    if not lig2_fpath:
        translated_lig1_coords = lig1_coords + translation_vec
        system.add(lig1_mmtop, translated_lig1_coords * angstrom)

    # rbfe
    else:
        system.add(lig1_mmtop, lig1_coords * angstrom)
        lig2_mol = Molecule.from_file(
            str(lig2_fpath), file_format="SDF", allow_undefined_stereo=True
        )
        small_mols.append(lig2_mol)
        lig2_mmtop = lig2_mol.to_topology().to_openmm(ensure_unique_atom_names=True)
        next(lig2_mmtop.residues()).name = "L2"
        lig2_coords = lig2_mol.conformers[0].to("angstrom").magnitude
        translated_lig2_coords = lig2_coords + translation_vec
        system.add(lig2_mmtop, translated_lig2_coords * angstrom)

    # add cofator if exists
    if cofactor_fpath:
        cofactor_mol = Molecule.from_file(
            str(cofactor_fpath), file_format="SDF", allow_undefined_stereo=True
        )
        small_mols.append(cofactor_mol)
        cofactor_mmtop = cofactor_mol.to_topology().to_openmm(
            ensure_unique_atom_names=True
        )
        cofactor_coords = cofactor_mol.conformers[0].to("angstrom").magnitude
        system.add(cofactor_mmtop, cofactor_coords * angstrom)

    
    smirnoff = SMIRNOFFTemplateGenerator(molecules=small_mols, forcefield=ligand_ff)
    ff.registerTemplateGenerator(smirnoff.generator)

    x_coords =[i[0].value_in_unit(nanometer) for i in system.positions]
    y_coords = [i[1].value_in_unit(nanometer) for i in system.positions]
    z_coords = [i[2].value_in_unit(nanometer) for i in system.positions]
    x_range = (max(x_coords) - min(x_coords))
    y_range = max(y_coords) - min(y_coords)
    z_range = max(z_coords) - min(z_coords)
    boxSize = Vec3(x_range +2, y_range+2, z_range+2)*nanometer # 2 nM padding 
    # add solvent to full boxSize
    system.addSolvent(
        ff,
        boxSize=boxSize,
    )

    sys = ff.createSystem(
        system.topology,
        nonbondedMethod=PME,
        nonbondedCutoff=0.9 * nanometer,
        constraints=HBonds,
        rigidWater=True,
        removeCMMotion=False,
        hydrogenMass=1.5 * amu if is_hmass else 1 * amu,
    )

    with open(xml_out_fpath, "w") as fh:
        fh.write(XmlSerializer.serialize(sys))

    with open(pdb_out_fpath, "w") as fh:
        PDBFile.writeFile(system.topology, system.positions, fh)

    return (xml_out_fpath, pdb_out_fpath)


def multi_run_wrapper(args):
    return create_xml_from_openff(*args)

def generate_morph_graph(
    morph_type: str,  # star or lomap
    ref_ligname: str,
    ligand_dpath: Path,
    morph_fpath: Path,
) -> Path:

    """
    Return the path to `Morph.in` file
    """
    with open(morph_fpath, "w") as fh:
        if morph_type == "star":
            for ligname in [e.name for e in ligand_dpath.iterdir() if e.is_dir()]:
                if ligname != ref_ligname:
                    fh.write(f"{ref_ligname}~{ligname}\n")

        elif morph_type == "lomap":
            Path("./morph_graph").mkdir(parents=True, exist_ok=True)
            with tmp_cd("morph_graph"):
                for lig in [e for e in ligand_dpath.iterdir() if e.is_dir()]:
                    shutil.copyfile(lig / "ligand.sdf", f"./{lig.name}.sdf")
                cmdline = ["lomap", "."]
                p = subprocess.run(cmdline, capture_output=True)
                if p.stderr:
                    with open("lomap.err", "w") as fh:
                        fh.write(p.stderr)
                if p.stdout:
                    with open("lomap.log", "w") as fh:
                        fh.write(p.stdout)

                name_dict = {}
                for line in open("./out.txt", "r").readlines()[1:]:
                    idx = line.strip().split()[0]
                    lig_name = line.strip().split()[1].strip(".sdf")
                    name_dict.update({idx: lig_name})

                with open(morph_fpath, "w") as fh:
                    for line in [
                        line
                        for line in open("out.dot", "r").readlines()
                        if "--" in line
                    ]:
                        temp = line.split("[")[0]
                        lig1_idx = temp.strip().split("--")[0].strip()
                        lig2_idx = temp.strip().split("--")[1].strip()
                        fh.write(f"{name_dict[lig1_idx]}~{name_dict[lig2_idx]}\n")

    return morph_fpath

def check_atm_input(config):
    """
    Check the input files for ATM
    """

    assert Path(config.ligand_dpathname).is_dir(), "not ligand dir found"
    assert Path(config.protein_fpathname).is_file(),"no protein file found"
    
    if Path(config.cofactor_fpathname).is_file():
        assert Path(config.cofactor_fpathname).name == "cofactor.sdf", "you have to put cofactor as '[name]/cofactor.sdf'"
