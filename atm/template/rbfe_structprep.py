#! python

from __future__ import print_function
from __future__ import division
import sys
import time
from pathlib import Path
import logging
import os

from sys import stdout
sys.path.append("${atom_build_path}")
from openmm.app import *
from openmm import *
from openmm.unit import *

import logging
from configobj import ConfigObj

from ommsystem import *
from utils.AtomUtils import AtomUtils, residue_is_solvent


class OMMSystemRBFEnoATM(OMMSystemRBFE):
    def create_system(self):

        self.load_system()
        self.atm_utils = AtomUtils(self.system)
        self.set_ligand_atoms()
        self.set_displacement()
        self.set_vsite_restraints()
        #set orientation restraints
        self.set_orientation_restraints()
        #set reference atoms for alignment force
        self.set_alignmentForce()
        #indexes of the atoms whose position is restrained near the initial positions
        #by a flat-bottom harmonic potential.
        self.set_positional_restraints()
        
        #target temperature
        temp = self.keywords.get("TEMPERATURES")
        if temp == None:
            self.temperature = 300.0 * kelvin
        elif isinstance(temp, list):
            self.temperature = float(temp[0]) * kelvin
        else:
            self.temperature = float(temp) * kelvin

        self.set_torsion_metaDbias(self.temperature)

        #do not include ATM Force, instead place the nonbonded
        #forces in what it would be the the ATMForce group
        #for the integrator
        #self.set_atmforce()
        self.atmforcegroup = self.free_force_group()
        import re
        nbpattern = re.compile(".*Nonbonded.*")
        for i in range(self.system.getNumForces()):
            if nbpattern.match(str(type(self.system.getForce(i)))):
                nbforce = self.system.getForce(i)
                nbforce.setForceGroup(self.atmforcegroup)
                break
        
        #add barostat
        pressure=1*bar
        self.set_barostat(self.temperature,pressure,25)

        self.set_integrator(self.temperature, self.frictionCoeff, self.MDstepsize)

def do_mintherm(keywords, logger):
    basename = keywords.get('BASENAME')
    jobname = basename
    
    pdbtopfile = basename + ".pdb"
    systemfile = basename + "_sys.xml"

    #OpenMM system for minimization, thermalization, NPT, NVT
    #does not include ATM Force
    syst = OMMSystemRBFEnoATM(basename, keywords, pdbtopfile, systemfile, logger)
    syst.create_system()
    
    platform_properties = {}
    platform_name = keywords.get("OPENMM_PLATFORM")
    if platform_name == None:
        platform_name = "CUDA"
    if platform_name == "CUDA" or platform_name == "OpenCL" or platform_name == "HIP":
        platform_properties["DeviceIndex"] = '${device_index}'
        platform_properties["Precision"] = "mixed"
    platform = Platform.getPlatformByName(platform_name)
    
    simulation = Simulation(syst.topology, syst.system, syst.integrator, platform, platform_properties)
    simulation.context.setPositions(syst.positions)
    if syst.boxvectors is not None:
        simulation.context.setPeriodicBoxVectors(syst.boxvectors[0], syst.boxvectors[1], syst.boxvectors[2] )

    print (f"Using platform {simulation.context.getPlatform().getName()} on gpu: {platform_properties['DeviceIndex']}")
        
    print("Potential energy before minimization =", simulation.context.getState(getEnergy = True).getPotentialEnergy())
    print("Energy minimizing the system ...")
    simulation.minimizeEnergy()
    print("Potential energy after minimization =", simulation.context.getState(getEnergy = True).getPotentialEnergy())

    #saves minimization checkpoint
    simulation.saveState(jobname + '_min.xml')
    #saves a pdb file
    positions = simulation.context.getState(getPositions=True).getPositions()
    boxsize = simulation.context.getState().getPeriodicBoxVectors()
    simulation.topology.setPeriodicBoxVectors(boxsize)
    with open(jobname + '_min.pdb', 'w') as output:
        PDBFile.writeFile(simulation.topology, positions, output)

    print("Thermalization ...")
 
    #FIX ME - get from control file
    totalSteps = 150000
    steps_per_cycle = 5000
    number_of_cycles = int(totalSteps/steps_per_cycle)
    simulation.reporters.append(StateDataReporter(stdout, steps_per_cycle, step=True, potentialEnergy = True, temperature=True, volume=True))    
    
    # initial temperature
    initial_temp = keywords.get("INITIAL_TEMPERATURE")
    if initial_temp == None:
        initial_temperature = 50.0 * kelvin
    else:
        initial_temperature = float(initial_temp) * kelvin

    final_temperature = syst.temperature
    delta_temperature = (final_temperature - initial_temperature)/number_of_cycles

    syst.barostat.setFrequency(0)#disabled

    #MD with temperature ramp
    temperature = initial_temperature
    syst.integrator.setTemperature(temperature)
    for i in range(number_of_cycles):
        simulation.step(steps_per_cycle)
        #prepare system for new temperature
        temperature = temperature + delta_temperature
        syst.integrator.setTemperature(temperature)

    #saves thermalized checkpoint
    simulation.saveState(jobname + '_therm.xml')
    #saves a pdb file
    positions = simulation.context.getState(getPositions=True).getPositions()
    boxsize = simulation.context.getState().getPeriodicBoxVectors()
    simulation.topology.setPeriodicBoxVectors(boxsize)
    with open(jobname + '_therm.pdb', 'w') as output:
        PDBFile.writeFile(simulation.topology, positions, output)

    print("NPT equilibration ...")
    
    syst.barostat.setFrequency(25)

    for i in range(number_of_cycles):
        simulation.step(steps_per_cycle)

    #saves checkpoint
    simulation.saveState(jobname + '_npt.xml')
    #saves a pdb file
    positions = simulation.context.getState(getPositions=True).getPositions()
    boxsize = simulation.context.getState().getPeriodicBoxVectors()
    simulation.topology.setPeriodicBoxVectors(boxsize)
    with open(jobname + '_npt.pdb', 'w') as output:
        PDBFile.writeFile(simulation.topology, positions, output)

    print("NVT equilibration ...")
    syst.barostat.setFrequency(999999999)#disabled

    #MD at constant volume
    for i in range(number_of_cycles):
        simulation.step(steps_per_cycle)

    #saves checkpoint
    simulation.saveState(jobname + '_equil.xml')
    #saves a pdb file
    positions = simulation.context.getState(getPositions=True).getPositions()
    boxsize = simulation.context.getState().getPeriodicBoxVectors()
    simulation.topology.setPeriodicBoxVectors(boxsize)
    with open(jobname + '_equil.pdb', 'w') as output:
        PDBFile.writeFile(simulation.topology, positions, output)

def do_lambda_annealing(keywords, logger):
    basename = keywords.get('BASENAME')
    jobname = basename
    
    pdbtopfile = basename + ".pdb"
    systemfile = basename + "_sys.xml"
    
    syst = OMMSystemRBFE(basename, keywords, pdbtopfile, systemfile, logger)
    syst.create_system()
    
    platform_properties = {}
    platform_name = keywords.get("OPENMM_PLATFORM")
    if platform_name == None:
        platform_name = "CUDA"
    if platform_name == "CUDA" or platform_name == "OpenCL" or platform_name == "HIP":
        platform_properties["Precision"] = "mixed"
        platform_properties["DeviceIndex"] = '${device_index}'
    platform = Platform.getPlatformByName(platform_name)
    
    simulation = Simulation(syst.topology, syst.system, syst.integrator, platform, platform_properties)
    simulation.context.setPositions(syst.positions)
    if syst.boxvectors is not None:
        simulation.context.setPeriodicBoxVectors(syst.boxvectors[0], syst.boxvectors[1], syst.boxvectors[2] )


    print (f"Using platform {simulation.context.getPlatform().getName()} on gpu: {platform_properties['DeviceIndex']}")
    
    syst.barostat.setFrequency(0)#disabled

    #target temperature
    temp = keywords.get("TEMPERATURES")
    if temp == None:
        temperature = 300.0 * kelvin
    elif isinstance(temp, list):
        temperature = float(temp[0]) * kelvin
    else:
        temperature = float(temp) * kelvin
    syst.integrator.setTemperature(temperature)
    lmbd = 0.0
    lambda1 = lmbd
    lambda2 = lmbd
    alpha = 0.0 / kilocalorie_per_mole
    uh = 0.0 * kilocalorie_per_mole
    w0coeff = 0.0 * kilocalorie_per_mole
    umsc =  1000.0 * kilocalorie_per_mole
    ubcore = 500.0 * kilocalorie_per_mole
    acore = 0.062500
    direction = 1
    uoffset = 0.0 * kilocalorie_per_mole

    print( "LoadState ...")
    simulation.loadState(jobname + '_equil.xml')

    #override ATM parameters
    simulation.context.setParameter(syst.atmforce.Lambda1(), lambda1)
    simulation.context.setParameter(syst.atmforce.Lambda2(), lambda2)
    simulation.context.setParameter(syst.atmforce.Alpha(), alpha *kilojoules_per_mole)
    simulation.context.setParameter(syst.atmforce.Uh(), uh /kilojoules_per_mole)
    simulation.context.setParameter(syst.atmforce.W0(), w0coeff /kilojoules_per_mole)
    simulation.context.setParameter(syst.atmforce.Umax(), umsc /kilojoules_per_mole)
    simulation.context.setParameter(syst.atmforce.Ubcore(), ubcore /kilojoules_per_mole)
    simulation.context.setParameter(syst.atmforce.Acore(), acore)
    simulation.context.setParameter(syst.atmforce.Direction(), direction)


    if syst.doMetaD:
        fgroups = {0,syst.metaDforcegroup,syst.atmforcegroup}
    else:
        fgroups = {0,syst.atmforcegroup}

    state = simulation.context.getState(getEnergy = True, groups = fgroups)
    #print("Potential Energy =", state.getPotentialEnergy())

    print("Annealing to lambda = 1/2 ...")

    #FIX ME: get from keywords
    totalSteps = 250000
    steps_per_cycle = 5000
    number_of_cycles = int(totalSteps/steps_per_cycle)
    deltalambda = (0.5 - 0.0)/float(number_of_cycles)
    simulation.reporters.append(StateDataReporter(stdout, steps_per_cycle, step=True, potentialEnergy = True, temperature=True))
    if os.path.exists(jobname+"_mdlambda.xtc"):
        os.remove(jobname+"_mdlambda.xtc")
    
    simulation.reporters.append(XTCReporter(jobname + "_mdlambda.xtc", steps_per_cycle))

    state = simulation.context.getState(getEnergy = True, groups = fgroups)
    print("Potential Energy =", state.getPotentialEnergy())
    
    binding_file = jobname + '_mdlambda.out'
    f = open(binding_file, 'w')
    
    for i in range(number_of_cycles):

        simulation.step(steps_per_cycle)
        state = simulation.context.getState(getEnergy = True, groups = fgroups )
        pot_energy = state.getPotentialEnergy()
        (u1, u0, ebias) = syst.atmforce.getPerturbationEnergy(simulation.context)
        umcore = simulation.context.getParameter(syst.atmforce.Umax())* kilojoules_per_mole
        ubcore = simulation.context.getParameter(syst.atmforce.Ubcore())* kilojoules_per_mole
        acore = simulation.context.getParameter(syst.atmforce.Acore())
        direction = simulation.context.getParameter(syst.atmforce.Direction())
        if direction > 0:
            pert_energy = syst.atm_utils.softCorePertE(u1 - (u0+uoffset), umcore, ubcore, acore)
        else:
            pert_energy = syst.atm_utils.softCorePertE((u0+uoffset) - u1, umcore, ubcore, acore)
        l1 = simulation.context.getParameter(syst.atmforce.Lambda1())
        l2 = simulation.context.getParameter(syst.atmforce.Lambda2())
        a = simulation.context.getParameter(syst.atmforce.Alpha()) / kilojoules_per_mole
        umid = simulation.context.getParameter(syst.atmforce.Uh()) * kilojoules_per_mole
        w0 = simulation.context.getParameter(syst.atmforce.W0()) * kilojoules_per_mole
        print("%f %f %f %f %f %f %f %f %f" % (temperature/kelvin,lmbd, l1, l2, a*kilocalorie_per_mole, umid/kilocalorie_per_mole, w0/kilocalorie_per_mole, pot_energy/kilocalorie_per_mole, pert_energy/kilocalorie_per_mole), file=f )
        f.flush()
        lmbd += deltalambda
        lambda1 += deltalambda
        lambda2 += deltalambda
        simulation.context.setParameter(syst.atmforce.Lambda1(), lambda1)
        simulation.context.setParameter(syst.atmforce.Lambda2(), lambda2)

    f.close()
        
    print( "SaveState ...")
    simulation.saveState(jobname + "_mdlambda.xml")

    #saves a pdb file
    positions = simulation.context.getState(getPositions=True).getPositions()
    boxsize = simulation.context.getState().getPeriodicBoxVectors()
    simulation.topology.setPeriodicBoxVectors(boxsize)
    with open(jobname + '_mdlambda.pdb', 'w') as output:
        PDBFile.writeFile(simulation.topology, positions, output)

def do_equil(keywords, logger):
    basename = keywords.get('BASENAME')
    jobname = basename
    
    pdbtopfile = basename + ".pdb"
    systemfile = basename + "_sys.xml"

    syst = OMMSystemRBFE(basename, keywords, pdbtopfile,  systemfile, logger)
    syst.create_system()
    
    platform_properties = {}
    platform_name = keywords.get("OPENMM_PLATFORM")
    if platform_name == None:
        platform_name = "CUDA"
    if platform_name == "CUDA" or platform_name == "OpenCL" or platform_name == "HIP":
        platform_properties["Precision"] = "mixed"
        platform_properties["DeviceIndex"] = '${device_index}'
    platform = Platform.getPlatformByName(platform_name)
    
    simulation = Simulation(syst.topology, syst.system, syst.integrator, platform, platform_properties)
    simulation.context.setPositions(syst.positions)
    if syst.boxvectors is not None:
        simulation.context.setPeriodicBoxVectors(syst.boxvectors[0], syst.boxvectors[1], syst.boxvectors[2] )

    print (f"Using platform {simulation.context.getPlatform().getName()} on gpu: {platform_properties['DeviceIndex']}")
    
    syst.barostat.setFrequency(0)#disabled

    #target temperature
    temp = keywords.get("TEMPERATURES")
    if temp == None:
        temperature = 300.0 * kelvin
    elif isinstance(temp, list):
        temperature = float(temp[0]) * kelvin
    else:
        temperature = float(temp) * kelvin
    syst.integrator.setTemperature(temperature)

    lmbd = 0.5
    lambda1 = lmbd
    lambda2 = lmbd
    alpha = 0.0 / kilocalorie_per_mole
    uh = 0.0 * kilocalorie_per_mole
    w0coeff = 0.0 * kilocalorie_per_mole
    umsc =  1000.0 * kilocalorie_per_mole
    ubcore = 500.0 * kilocalorie_per_mole
    acore = 0.062500
    direction = 1
    uoffset = 0.0 * kilocalorie_per_mole

    print( "LoadState ...")
    simulation.loadState(jobname + '_mdlambda.xml')

    #override ATM parameters
    simulation.context.setParameter(syst.atmforce.Lambda1(), lambda1)
    simulation.context.setParameter(syst.atmforce.Lambda2(), lambda2)
    simulation.context.setParameter(syst.atmforce.Alpha(), alpha *kilojoules_per_mole)
    simulation.context.setParameter(syst.atmforce.Uh(), uh /kilojoules_per_mole)
    simulation.context.setParameter(syst.atmforce.W0(), w0coeff /kilojoules_per_mole)
    simulation.context.setParameter(syst.atmforce.Umax(), umsc /kilojoules_per_mole)
    simulation.context.setParameter(syst.atmforce.Ubcore(), ubcore /kilojoules_per_mole)
    simulation.context.setParameter(syst.atmforce.Acore(), acore)
    simulation.context.setParameter(syst.atmforce.Direction(), direction)
    


    if syst.doMetaD:
        fgroups = {0,syst.metaDforcegroup,syst.atmforcegroup}
    else:
        fgroups = {0,syst.atmforcegroup}

    state = simulation.context.getState(getEnergy = True, groups = fgroups)
    print("Potential Energy =", state.getPotentialEnergy())

    print("Equilibration at lambda = 1/2 ...")

    #FIX ME: get from keywords
    totalSteps = 150000
    steps_per_cycle = 5000
    simulation.reporters.append(StateDataReporter(stdout, steps_per_cycle, step=True, potentialEnergy = True, temperature=True))
    
    if os.path.exists(jobname + "_0.xtc"):
        os.remove(jobname+"_0.xtc")
    simulation.reporters.append(XTCReporter(jobname + "_0.xtc", steps_per_cycle))

    state = simulation.context.getState(getEnergy = True, groups = fgroups)

    simulation.step(totalSteps)
        
    print( "SaveState ...")
    simulation.saveState(jobname + "_0.xml")

    #saves a pdb file
    positions = simulation.context.getState(getPositions=True).getPositions()
    boxsize = simulation.context.getState().getPeriodicBoxVectors()
    simulation.topology.setPeriodicBoxVectors(boxsize)
    with open(jobname + '_0.pdb', 'w') as output:
        PDBFile.writeFile(simulation.topology, positions, output)
        

def massage_keywords(keywords, restrain_solutes = True):

    #use 1 fs time step
    keywords['TIME_STEP'] = 0.001

    #temporarily restrain all non-solvent atoms
    if restrain_solutes:
        basename = old_keywords.get('BASENAME')
        pdbtopfile = basename + ".pdb"
        pdb = PDBFile(pdbtopfile)
        non_ion_wat_atoms = []
        for res in pdb.topology.residues():
            if not residue_is_solvent(res):
                for atom in res.atoms():
                    non_ion_wat_atoms.append(atom.index)
        keywords['POS_RESTRAINED_ATOMS'] = non_ion_wat_atoms

if __name__ == '__main__':

    # Parse arguments:
    usage = "%prog <ConfigFile>"

    if len(sys.argv) != 2:
        print("Please specify ONE input file")
        sys.exit(1)

    commandFile = sys.argv[1]

    print("")
    print("========================================")
    print("AToM RBFE Structure Preparation         ")
    print("========================================")
    print("")
    print("Started at: " + str(time.asctime()))
    print("Input file:", commandFile)
    print("")
    sys.stdout.flush()
    
    keywords = ConfigObj(commandFile)
    logger = logging.getLogger("rbfe_structprep")

    restrain_solutes = True

    old_keywords = keywords.copy()
    massage_keywords(keywords, restrain_solutes)
    
    do_mintherm(keywords, logger)
    do_lambda_annealing(keywords, logger)

    #reestablish the restrained atoms
    if restrain_solutes:
        keywords['POS_RESTRAINED_ATOMS'] = old_keywords.get('POS_RESTRAINED_ATOMS') 

    do_equil(keywords, logger)