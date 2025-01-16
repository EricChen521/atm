"""Forcefield generation class."""
import glob
import os
import shutil
import subprocess

from dataclasses import dataclass
from pathlib import Path

from rdkit import Chem

from atm.utility import tmp_cd


@dataclass
class Gaff:
    """a Gaff engine."""

    ligands_dpath: Path = None
    cofactor_dpath: Path = None
    forcefield_dpath: Path = None

    def produce(
        self,
    ) -> None:
        """Run gaff kernel."""
        procs = {}
        self.forcefield_dpath.mkdir(parents=True, exist_ok=True)
        # copy_foldered_files(ligand_dir,forcefield_dir,"ligand.sdf")
        for ligand_dir in [e for e in self.ligands_dpath.iterdir() if e.is_dir()]:
            ligand_fpath = ligand_dir / "ligand.sdf"
            ligand_ff_dir = self.forcefield_dpath / ligand_dir.name
            ligand_ff_dir.mkdir(parents=True, exist_ok=True)
            mol = Chem.SDMolSupplier(str(ligand_fpath), removeHs=False)[0]
            charge = Chem.rdmolops.GetFormalCharge(mol)
            cmdline = [
                f"antechamber -i {ligand_fpath} -fi sdf -o "
                f"{ligand_ff_dir}/vacuum.mol2 -fo "
                f"mol2 -rn LIG -c bcc -at gaff2 -nc {str(charge)}; "
                f"parmchk2 -i {ligand_ff_dir}/vacuum.mol2 -f mol2"
                f" -o {ligand_ff_dir}/vacuum.frcmod"
            ]
            # execuate if the vaccum.mol2 file is not there
            if not (ligand_ff_dir/"vacuum.mol2").is_file():
                
                print(f"Run gaff for {ligand_ff_dir}...")
                child_process = subprocess.Popen(
                    cmdline,
                    stderr=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    shell=True,
                    cwd=ligand_ff_dir,
                )
                procs.update({ligand_ff_dir: child_process})

        if (
            self.cofactor_dpath
            and not (self.cofactor_dpath / "cofactor.mol2").is_file()
        ):
            cofactor_fpath = self.cofactor_dpath / "cofactor.sdf"
            mol = Chem.SDMolSupplier(str(cofactor_fpath), removeHs=False)[0]
            charge = Chem.rdmolops.GetFormalCharge(mol)
            cmdline = [
                f"antechamber -i {cofactor_fpath} -fi sdf -o "
                f"{cofactor_fpath.parent}/cofactor.mol2 -fo "
                f"mol2 -rn COF -c bcc -at gaff2 -nc {str(charge)}; "
                f"parmchk2 -i {cofactor_fpath.parent}/cofactor.mol2 -f mol2"
                f" -o {cofactor_fpath.parent}/cofactor.frcmod"
            ]
            child_process = subprocess.Popen(
                cmdline,
                stderr=subprocess.PIPE,
                stdout=subprocess.PIPE,
                shell=True,
                cwd=self.cofactor_dpath,
            )
            procs.update({self.cofactor_dpath: child_process})

        for ligand_ff_dir, child_proc in procs.items(): 
            stdout, stderr = child_proc.communicate()
            with open(ligand_ff_dir / "gaff.log", "w") as fh:
                fh.write(stdout.decode("utf-8"))

            if stderr:
                with open(ligand_ff_dir / "gaff.err", "w") as fh:
                    fh.write(stderr.decode("utf-8"))

        # remove failed ligand ff and clean files
        for ligand_ff_dir in [e for e in self.forcefield_dpath.iterdir() if e.is_dir()]:
            if (ligand_ff_dir / "vacuum.mol2").is_file():
                with tmp_cd(ligand_ff_dir):
                    (ligand_ff_dir / "leaprc_header").touch()
                    for fname in (
                        glob.glob("ANTECHAMBER*") + glob.glob("sqm*") + ["ATOMTYPE.INF"]
                    ):
                        try:
                            os.remove(fname)
                        except OSError:
                            continue
            else:
                print(f"gaff failed for {ligand_ff_dir.name}")
                shutil.rmtree(ligand_ff_dir)
        print("Gaff forcefield is genereated successfully!")


@dataclass
class Quickgaff:
    """A quickgaff engine."""

    ligands_dpath: Path = None
    forcefield_dpath: Path = None
    cofactor_dpath: Path = None
    miniconda3_pathname: str = '/home/eric.chen/miniconda3'

    def produce(
        self,
    ) -> None:
        """Run quickff kernel."""
        cmdline = f". {self.miniconda3_pathname}/bin/activate parmedizer-dev;"+\
        f" parmedizer apply {self.ligands_dpath}/*/ligand.sdf --retyper amber_legacy"   
        print(cmdline)
        p = subprocess.run(cmdline, shell=True,executable="/usr/bin/bash", capture_output=True, text=True)

        if self.cofactor_dpath:

            cofactor_fpath=self.cofactor_dpath/"cofactor.sdf"
            mol = Chem.SDMolSupplier(str(cofactor_fpath), removeHs=False)[0]
            charge = Chem.rdmolops.GetFormalCharge(mol)

            cmdline = [
                f"antechamber -i {cofactor_fpath} -fi sdf -o "
                f"{cofactor_fpath.parent}/cofactor.mol2 -fo "
                f"mol2 -rn COF -c bcc -nc {str(charge)}; "
                f"parmchk2 -i {cofactor_fpath.parent}/cofactor.mol2 -f mol2"
                f" -o {cofactor_fpath.parent}/cofactor.frcmod"
            ]
            subprocess.Popen(
                cmdline,
                stderr=subprocess.PIPE,
                stdout=subprocess.PIPE,
                shell=True,
                cwd=self.cofactor_dpath,
            )
        if p.stderr:
            with open("quickgaff.log", "w") as fh:
                fh.write(p.stderr)
