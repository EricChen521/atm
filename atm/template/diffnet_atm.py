#!/usr/bin/env python
import argparse
import collections
import math
import os
import sys

from pathlib import Path

import numpy as np
import pandas as pd
import scipy

# Awful horrible hack for now, but I can't see this needing to change.
kT = 0.001987192 * 298.15  # in kcal/mol
# DEFAULT_ERROR = None
DEFAULT_ERROR = 0.43  # kcal/mol, based on meta-analysis of pKi


def diffnet(x, cov, x0=None, cov0=None):
    """
    This has been excised and re-written from at least 3 functions in the
    original DiffNet package. Results have been validated for the normal
    usage here (a single reference ligand, etc.).

    x : 2D array-like
      Square matrix of difference measurements x_ij = x_i - x_j. Missing
      values can be either None or nan.
    cov : 2D array-like (same shape as x)
      Square matrix of squared errors (covariance) of x.
    x0 : 1D array-like (optional)
      Known values of any or all diagonal elements in x.
      Unknown values can be either None or nan.
    cov0 : 1D array-like (optional)
      Squared uncertainties of values in x0. Values corresponding to unknown
      x0 values are ignored.
    """
    # nb `None` entries in an array-like object are converted to np.nan when
    # cast as a float array (this is not the default!). We exploit this by
    # only operating on the indices that return isnan False.
    #
    # Prepare estimated values and uncertainties.
    # Only consider entries where x is not nan (indexed as i).
    #
    x = np.array(x, np.float64)
    cov = np.array(cov, np.float64)
    assert x.ndim == 2 and cov.ndim == 2
    n = x.shape[0]
    assert x.shape[1] == n
    assert cov.shape == x.shape
    i = np.where(~np.isnan(x))
    j = np.where(np.isnan(x))
    x[j] = 0.0  # Needed for vectorization. Any number here would work.
    # By construction, all missing x values have infinite uncertainty and thus
    # invcov of zero.
    #
    invcov = np.zeros(cov.shape)
    invcov[i] = 1.0 / cov[i]

    # Prepare optional known values and uncertainties.
    # Only agument entries where x0 is not nan (indexed as i0).
    #
    if x0 is None:
        x0 = [None for i in range(n)]
    x0 = np.array(x0, np.float64)
    assert x0.ndim == 1
    assert x0.size == n
    if cov0 is None:
        cov0 = [None for i in range(n)]
    cov0 = np.array(cov0, np.float64)
    assert cov0.ndim == 1
    assert cov0.size == n
    i0 = np.where(~np.isnan(x0))[0]
    j0 = np.where(np.isnan(x0))[0]
    x0[j0] = 0.0  # Needed for vectorization. Any number here would work.
    invcov0 = np.zeros(cov0.shape)
    if i0.size > 0:
        invcov0[i0] = 1.0 / cov0[i0]

    # Begin actual computations.
    # nb(BKR) It's easier to construct the Fisher information matrix as
    # component diagonal and off-diagonal matrices.
    #
    z = (invcov * x).sum(axis=1) + invcov0 * x0
    F = -invcov
    np.fill_diagonal(F, 0.0)
    F += np.diag(invcov.sum(axis=1) + invcov0)
    xfit, residuals, rank, _ = np.linalg.lstsq(F, z, rcond=None)
    v = scipy.linalg.null_space(F)
    assert rank + v.shape[1] == n
    # j0.size == n implies that all x0 are unknown.
    if not (v.shape[1] == 0 or j0.size == n):
        dx = x0[i0] - xfit[i0]
        vp = v[i0]
        lv, _, _, _ = np.linalg.lstsq(vp, dx, rcond=None)
        xfit += v.dot(lv)

    # Now compute the covariance matrix (inverse of F).
    #
    if np.all(np.abs(np.diag(invcov)) < 1e-9) and j0.size in [n, n - 1]:
        # The Fisher information matrix is singular. Solve under constraint on
        # the mean of x. This happens if we have no reference (j0.size == n) or
        # an arbitrary shift (j0.size == n - 1).
        #
        F1 = np.pad(F, ((1, 0), (0, 1)), constant_values=1)
        F1[0, n] = 0.0
        I0 = np.zeros((n + 1, n))
        I0[1:, :] = np.identity(n)
        C1 = np.linalg.solve(F1, I0)
        C = C1[:n, :]
    else:
        C = np.linalg.inv(F)
    return (xfit, C)


parser = argparse.ArgumentParser()

parser.add_argument(
    "--atm-dat",
    type=str,
    default='',
    help="True if input file is generated by atm."
)
parser.add_argument(
    "--ref-lig",
    type=str,
    default=None,
    nargs="*",
    help="formatted as: <lig name> [<ref dG> [<ref err>]]",
)
parser.add_argument(
    "--ref-file",
    type=str,
    default=None,
    help="Simple space delimited file for multiple ref. ligand specfications",
)
parser.add_argument(
    "--result-dir",
    type=str,
    default=".",
    help="pathname to save result files",
)
parser.add_argument(
    "--omit", type=str, default=[], nargs="*", help="List of ligands to ignore"
)
parser.add_argument(
    "--uniform-error",
    action="store_true",
    help="Ignore all error values and set weights to be uniform",
)
parser.add_argument("--no-delta", action="store_true")
parser.add_argument(
    "--err-cutoff",
    type=float,
    default=0.2,
    help=(
        "Put a floor on error values. "
        "Useful to avoid overweighting suspiciously high-precision dG values"
    ),
)
parser.add_argument(
    "--rmse",
    type=float,
    default=0.0,
    help="Assume RMSE bias relative to expt."
    " This only adjusts relative weights in DiffNet, not the precision",
)

parser.add_argument("--no-sort", action="store_true", help="Do not sort the output")
parser.add_argument(
    "--new-csv", action="store_true", help="write a new summary CSV in BFE format"
)

parser.add_argument("--quiet", "-q", action="store_true")
parser.add_argument("--shift", type=float, default=0.0)
parser.add_argument(
    "--conformer-suffix",
    type=str,
    default=None,
    help="Find ligands with naming pattern <name><suffix><n>, where <n> is a"
    " whole number. Consider these conformers of the same ligand.",
)
args = parser.parse_args()

# Read the BFE output.
#
dfs = []

with open("atm.csv","w") as fh:
    fh.write("Index,Ligand1,Ligand2,MBAR_binding_ddG,"
            "MBAR_binding_ddG_miderr,MBAR_binding_ddG_sem,"
            "TI_binding_ddG,TI_binding_ddG_miderr,TI_binding_ddG_sem\n")
    id=0
    for line in open(args.atm_dat,"r").readlines()[1:]:
        ligand1, ligand2, dG, error = line.strip().split(",")[:4]
        fh.write(f"{id},{ligand1},{ligand2},{dG},{error},{error},{dG},{error},{error}\n")
        id+=1
dfs.append(pd.read_csv("atm.csv",dtype="str",index_col=0))
os.remove("atm.csv")
    
# Extract the information we need for DiffNet.
# nb(BKR) We store the standard variance, not the standard error, since the
# estimators are different and variance is what goes into the DiffNet
# equations. miderr, however, is still the error.
#
ligand_names = []
ref_ligand_names = None
mbar_dG_dict = {}
var_mbar_dG_dict = {}
miderr_mbar_dG_dict = {}
ti_dG_dict = {}
var_ti_dG_dict = {}
miderr_ti_dG_dict = {}
ddG_type = "binding"  # may toggle to dehydration
bias2 = args.rmse**2
for df in dfs:
    # nb(BKR) pandas does not read files deterministically when using iterrows,
    # so the order of the ligands will change randomly between runs on the same
    # file.
    #
    for _i, row in df.iterrows():
        ligand1 = str(row["Ligand1"])
        # Hack to fix incorrect names mangled with mapping type
        tokens = str(row["Ligand2"]).split("_")
        if len(tokens) > 1:
            ligand2 = "_".join(str(row["Ligand2"]).split("_")[:-1])
        else:
            ligand2 = str(row["Ligand2"])
        skip = False
        for omit_ligand in args.omit:
            # Match by prefix so that one specification can be used to omit
            # the same ligand with multiple pose suffixes.
            #
            if ligand1.startswith(omit_ligand):
                skip = True
                break
            if ligand2.startswith(omit_ligand):
                skip = True
                break
        if skip:
            continue

        mp = "%s~%s" % (ligand1, ligand2)
        ligand_names.extend([ligand1, ligand2])
        if mp not in mbar_dG_dict or mp not in ti_dG_dict:
            mbar_dG_dict[mp] = []
            ti_dG_dict[mp] = []
            miderr_mbar_dG_dict[mp] = []
            miderr_ti_dG_dict[mp] = []
        mbar_ddG_key = "MBAR_%s_ddG" % ddG_type
        try:
            row[mbar_ddG_key]
        except KeyError:
            ddG_type = "dehydration"
        mbar_ddG_key = "MBAR_%s_ddG" % ddG_type
        mbar_miderr_key = "MBAR_%s_ddG_miderr" % ddG_type
        mbar_sem_key = "MBAR_%s_ddG_sem" % ddG_type
        ti_ddG_key = "TI_%s_ddG" % ddG_type
        ti_miderr_key = "TI_%s_ddG_miderr" % ddG_type
        ti_sem_key = "TI_%s_ddG_sem" % ddG_type

        mbar_dG_dict[mp].append(float(row[mbar_ddG_key]))
        ti_dG_dict[mp].append(float(row[ti_ddG_key]))
        miderr_mbar_dG_dict[mp].append(
            math.sqrt(float(row[mbar_miderr_key]) ** 2 + bias2)
        )
        miderr_ti_dG_dict[mp].append(math.sqrt(float(row[ti_miderr_key]) ** 2 + bias2))
        # If multiple data sets are present, the variance will be overwritten
        # by the observed sample variance and miderr will be overwritten by the
        # mean miderr.
        #
        var_mbar_dG_dict[mp] = float(row[mbar_sem_key]) ** 2 + bias2
        var_ti_dG_dict[mp] = float(row[ti_sem_key]) ** 2 + bias2

ligand_names = list(set(ligand_names))
# Compute sample mean and variance.
for mp in mbar_dG_dict:
    mbar_data = np.asarray(mbar_dG_dict[mp])
    mbar_dG_dict[mp] = mbar_data.mean()
    ti_data = np.asarray(ti_dG_dict[mp])
    ti_dG_dict[mp] = ti_data.mean()
    miderr_mbar_dG_dict[mp] = np.asarray(miderr_mbar_dG_dict[mp]).mean()
    miderr_ti_dG_dict[mp] = np.asarray(miderr_ti_dG_dict[mp]).mean()
    if mbar_data.size > 1:
        var_mbar_dG_dict[mp] = mbar_data.var(ddof=1.5) + bias2
        var_ti_dG_dict[mp] = ti_data.var(ddof=1.5) + bias2

# Convert these to a matrix, use the order of ligand_names to assign indices to
# the entries in the dicts. MISSING VALUES ARE ALWAYS INFERRED FROM THE
# MEASUREMENTS BEING nan. The actual value in the variance matrices do not
# matter, since they are masked by the actual values:
#
# Example:
#   If mbar_vars[0, 2] = 1.0, but mbar_ddGs[0, 2] = np.nan, then the whole
#   calculation is ignored. Same goes for dG0_vars and dG0s.
#
nligands = len(ligand_names)
mbar_ddGs = np.zeros((nligands, nligands), np.float64)
ti_ddGs = np.zeros((nligands, nligands), np.float64)
mbar_ddGs[:, :] = np.nan
ti_ddGs[:, :] = np.nan

mbar_vars = np.zeros((nligands, nligands), np.float64)
ti_vars = np.zeros((nligands, nligands), np.float64)
mbar_midvars = np.zeros((nligands, nligands), np.float64)
ti_midvars = np.zeros((nligands, nligands), np.float64)

var_cutoff = args.err_cutoff**2
if args.uniform_error:

    def compute_var(var):
        return 1.0

else:

    def compute_var(var):
        return max(var_cutoff, var)


num_small_errs = 0
for i, ligand1 in enumerate(ligand_names):
    for j, ligand2 in enumerate(ligand_names):
        mp = "%s~%s" % (ligand1, ligand2)
        if mp in mbar_dG_dict:
            var = compute_var(var_mbar_dG_dict[mp])
            midvar = compute_var(miderr_mbar_dG_dict[mp] ** 2)
            num_small_errs += var <= var_cutoff
            if np.isnan(mbar_ddGs[i, j]):
                dGij = mbar_dG_dict[mp]
            else:
                # hysteresis check with both directions, avg the result
                # This should naturally merge with the averaging due to repeats?
                #
                dGij = 0.5 * (mbar_ddGs[j, i] + mbar_dG_dict[mp])
                var = 0.25 * (mbar_vars[j, i] + var)
                midvar = (mbar_ddGs[j, i] - mbar_dG_dict[mp]) ** 2
            # nb(BKR) ddG[i, j] = dG[i] - dG[j] but we compute dG[j] - dG[i]!
            mbar_ddGs[i, j] = -dGij
            mbar_ddGs[j, i] = +dGij
            mbar_vars[i, j] = mbar_vars[j, i] = var
            mbar_midvars[i, j] = mbar_midvars[j, i] = midvar
        if mp in ti_dG_dict:
            var = compute_var(var_ti_dG_dict[mp])
            midvar = compute_var(miderr_ti_dG_dict[mp] ** 2)
            if np.isnan(ti_ddGs[i, j]):
                dGij = ti_dG_dict[mp]
            else:
                dGij = 0.5 * (ti_ddGs[j, i] + ti_dG_dict[mp])
                var = 0.25 * (ti_vars[j, i] + var)
                midvar = (ti_ddGs[j, i] - ti_dG_dict[mp]) ** 2
            ti_ddGs[i, j] = -dGij
            ti_ddGs[j, i] = +dGij
            ti_vars[i, j] = ti_vars[j, i] = var
            ti_midvars[i, j] = ti_midvars[j, i] = midvar
if not args.quiet:
    print(
        "# Found %d MBAR error estimates below cutoff of %f"
        % (num_small_errs, args.err_cutoff)
    )


def read_ref_data(ligand_names):
    # Extract reference data, if any.
    # This does not play nice with the multiple conformers code, because the
    # reference values are always macroscopic while the conformer specific ddGs
    # are microscopic. In that latter case, do NOT read the reference data
    # here, because we don't know the macroscopic (i.e. unique) ligand names
    # just yet.
    #
    nligands = len(ligand_names)
    dG0s = np.array([None] * nligands, np.float64)
    dG0_vars = np.array([np.inf] * nligands, np.float64)
    ref_ligand_names = []
    dG0_fit = None
    if args.ref_file is not None:
        with open(args.ref_file, "r") as ref_file:
            for line in ref_file:
                if line.lstrip().startswith("#"):
                    continue
                tokens = line.strip().split()
                if len(tokens) < 2:
                    if not args.quiet:
                        print("# Skipping line with only one entry")
                    continue
                elif len(tokens) == 2:
                    ligand, dG = tokens
                    err_dG = DEFAULT_ERROR
                else:
                    ligand, dG, err_dG = tokens[:3]
                if ligand not in ligand_names:
                    if ligand not in args.omit and not args.quiet:
                        print(
                            "# Skipping reference data for unknown ligand %s" % ligand
                        )
                    continue
                index = ligand_names.index(ligand)
                dG0s[index] = float(dG)
                if err_dG is not None:
                    dG0_vars[index] = float(err_dG) ** 2
                ref_ligand_names.append(ligand)

        num_not_nan = (~np.isnan(dG0s)).sum()
        num_not_inf = (~np.isinf(dG0_vars)).sum()
        if num_not_nan == num_not_inf == nligands:
            # We have all reference values - do not use these to constrain,
            # instead do a linear regression for assessment purposes.
            print("# Performing benchmark, not constrained fitting")
            dG0_fit = dG0s
            dG0s = np.array([None] * nligands, np.float64)
    elif args.ref_lig is not None:
        if len(args.ref_lig) == 1:
            ref_lig = args.ref_lig[0]
            dG_ref = 0.0
            err_dG_ref = DEFAULT_ERROR
        elif len(args.ref_lig) == 2:
            ref_lig, dG_ref = args.ref_lig
            err_dG_ref = DEFAULT_ERROR
        elif len(args.ref_lig) == 3:
            ref_lig, dG_ref, err_dG_ref = args.ref_lig
        else:
            print("Too many values for --ref-lig!")
            sys.exit(1)
        index = ligand_names.index(ref_lig)
        dG0s[index] = float(dG_ref)
        if err_dG_ref is not None:
            dG0_vars[index] = float(err_dG_ref) ** 2
        ref_ligand_names.append(ref_lig)
    return (dG0s, dG0_vars, dG0_fit, ref_ligand_names)


if args.conformer_suffix is None:
    dG0s, dG0_vars, dG0_fit, ref_ligand_names = read_ref_data(ligand_names)
else:
    dG0s, dG0_vars = None, None  # no effect on DiffNet

# Run DiffNet
# Note that we run twice in order to adjust the miderr values based on the
# same Fisher information matrix, but only report dGs adjusted by the
# variance from repeat measurements.
#
num_not_nan = (~np.isnan(mbar_ddGs)).sum()
num_not_nan_diag = (~np.isnan(np.diag(mbar_ddGs))).sum()
all_bfe_is_absolute = False
if num_not_nan != num_not_nan_diag:
    mbar_dG_mles, mbar_dG_cov = diffnet(mbar_ddGs, mbar_vars, dG0s, dG0_vars)
    _, mbar_dG_cov_miderr = diffnet(mbar_ddGs, mbar_midvars, dG0s, dG0_vars)

    ti_dG_mles, ti_dG_cov = diffnet(ti_ddGs, ti_vars, dG0s, dG0_vars)
    _, ti_dG_cov_miderr = diffnet(ti_ddGs, ti_midvars, dG0s, dG0_vars)
else:
    # Shortcut if all of the input is ABFE.
    all_bfe_is_absolute = True
    mbar_dG_mles, mbar_dG_cov = np.diag(mbar_ddGs).copy(), mbar_vars
    mbar_dG_cov_miderr = mbar_midvars
    ti_dG_mles, ti_dG_cov = np.diag(ti_ddGs).copy(), ti_vars
    ti_dG_cov_miderr = ti_midvars

mbar_dG_mles += args.shift
ti_dG_mles += args.shift

mbar_err_dG_mles = np.sqrt(np.diag(mbar_dG_cov))
mbar_miderr_dG_mles = np.sqrt(np.diag(mbar_dG_cov_miderr))

ti_err_dG_mles = np.sqrt(np.diag(ti_dG_cov))
ti_miderr_dG_mles = np.sqrt(np.diag(ti_dG_cov_miderr))

# If present, combine estimates for multiple poses.
# nb(BKR) This is ONLY correct in the DiffNet ABFE-like frame, since all dGs
# are known up to a single constant. The combining rule for multiple poses is:
#
#   dG_b = sum_i p_i dG_i + kT sum_i p_i ln pi_i
#
# where the sum runs over all poses and
#
#   p_i = exp(-dG_i/kT) / sum_j exp(-dG_j/kT)
#
# The definition of p_i is invariant to an arbitrary shift in the dG_i, but
# dG_b will inherit any such shift. Since this applies to all ligands, the
# effect is no different from a normal DiffNet calculation.
#
mbar_dG_ents = np.array([None] * nligands, np.float64)  # dummy values
ti_dG_ents = np.array([None] * nligands, np.float64)
dominant_poses = ["" for i in range(nligands)]
if args.conformer_suffix is not None:
    ligands2indices = collections.defaultdict(list)
    ligands2poses = collections.defaultdict(list)
    for i, ligand in enumerate(ligand_names):
        tokens = ligand.split(args.conformer_suffix)
        #        if 3 < len(tokens):
        #            print(
        #                'Warning - found conformer pattern for ligand %s,'%ligand, \
        #                '  but could not determine pose label'
        #            )
        if len(tokens) > 1:
            # This extra join statement is for really short suffixes
            # e.g. mol-10-1 and mol-10-2 would have a separator of "-"
            # but clearly the unique name is mol-10, not mol
            unique_ligand = args.conformer_suffix.join(tokens[:-1])
            pose = tokens[-1]
            ligands2poses[unique_ligand].append(pose)
        else:
            unique_ligand = ligand
        ligands2indices[unique_ligand].append(i)
    nunique_ligands = len(ligands2indices)

    unique_mbar_dGs = np.zeros(nunique_ligands, np.float64)
    mbar_dG_ents = np.zeros(nunique_ligands, np.float64)
    unique_mbar_err_dGs = np.zeros(nunique_ligands, np.float64)
    unique_mbar_miderr_dGs = np.zeros(nunique_ligands, np.float64)

    unique_ti_dGs = np.zeros(nunique_ligands, np.float64)
    ti_dG_ents = np.zeros(nunique_ligands, np.float64)
    unique_ti_err_dGs = np.zeros(nunique_ligands, np.float64)
    unique_ti_miderr_dGs = np.zeros(nunique_ligands, np.float64)

    def weighted_dG(dGs, var_dGs, midvar_dGs):
        # for easier Boltzmann weighting, use df = dG / kT
        dfs = np.array(dGs) / kT
        var_dGs = np.array(var_dGs)
        midvar_dGs = np.array(midvar_dGs)
        norm = scipy.special.logsumexp(-dfs)
        dG = -kT * norm  # Boltzmann weighted dGs
        p_i = np.exp(-(dfs + norm))
        p_i[p_i < 1e-6] = 1e-6
        dG_ent = kT * (p_i * np.log(p_i)).sum()  # "entropic" term
        if p_i.size > 1:
            dG_ent /= -np.log(p_i.size)
        err_dG = np.sqrt((p_i**2 * var_dGs).sum())
        miderr_dG = np.sqrt((p_i**2 * midvar_dGs).sum())
        return (dG, dG_ent, err_dG, miderr_dG, p_i)

    unique_ligand_names = []
    for i, (unique_ligand, idxs) in enumerate(ligands2indices.items()):
        unique_ligand_names.append(unique_ligand)
        poses = ligands2poses[unique_ligand]
        # Combine MBAR results.
        dGs = [mbar_dG_mles[j] for j in idxs]
        var_dGs = [np.diag(mbar_dG_cov)[j] for j in idxs]
        midvar_dGs = [np.diag(mbar_dG_cov_miderr)[j] for j in idxs]
        dG, dG_ent, err_dG, miderr_dG, p = weighted_dG(dGs, var_dGs, midvar_dGs)
        if len(poses) > 0:
            dominant_poses[i] = poses[p.argmax()]

        unique_mbar_dGs[i] = dG
        mbar_dG_ents[i] = dG_ent
        unique_mbar_err_dGs[i] = err_dG
        unique_mbar_miderr_dGs[i] = miderr_dG
        # Combine TI results.
        dGs = [ti_dG_mles[j] for j in idxs]
        var_dGs = [np.diag(ti_dG_cov)[j] for j in idxs]
        midvar_dGs = [np.diag(ti_dG_cov_miderr)[j] for j in idxs]
        dG, dG_ent, err_dG, miderr_dG, p = weighted_dG(dGs, var_dGs, midvar_dGs)
        unique_ti_dGs[i] = dG
        ti_dG_ents[i] = dG_ent
        unique_ti_err_dGs[i] = err_dG
        unique_ti_miderr_dGs[i] = miderr_dG
    # Replace the non-unique values with the (combined) unique values.
    ligand_names = unique_ligand_names
    mbar_dG_mles = unique_mbar_dGs
    mbar_err_dG_mles = unique_mbar_err_dGs
    mbar_miderr_dG_mles = unique_mbar_miderr_dGs
    ti_dG_mles = unique_ti_dGs
    ti_err_dG_mles = unique_ti_err_dGs
    ti_miderr_dG_mles = unique_ti_miderr_dGs

if args.conformer_suffix is not None:
    dG0s, dG0_vars, dG0_fit, ref_ligand_names = read_ref_data(ligand_names)
    if not np.all(np.isnan(dG0s)):
        # Re-run DiffNet to adjust for experimental values
        mbar_ddGs = np.zeros((mbar_dG_mles.size, mbar_dG_mles.size))
        mbar_ddGs[:, :] = np.nan
        mbar_vars = np.zeros((mbar_dG_mles.size, mbar_dG_mles.size))
        mbar_midvars = np.zeros((mbar_dG_mles.size, mbar_dG_mles.size))

        mbar_var_mles = mbar_err_dG_mles**2
        mbar_midvar_mles = mbar_miderr_dG_mles**2
        for i, (dGi, vari, midvari) in enumerate(
            zip(mbar_dG_mles, mbar_var_mles, mbar_midvar_mles)
        ):
            for j, (dGj, varj, midvarj) in enumerate(
                zip(mbar_dG_mles, mbar_var_mles, mbar_midvar_mles)
            ):
                if i == j:
                    continue
                mbar_ddGs[i, j] = dGi - dGj
                mbar_vars[i, j] = vari + varj
                mbar_midvars[i, j] = midvari + midvarj

        mbar_dG_mles, mbar_dG_cov = diffnet(mbar_ddGs, mbar_vars, dG0s, dG0_vars)
        _, mbar_dG_cov_miderr = diffnet(mbar_ddGs, mbar_midvars, dG0s, dG0_vars)

        # TODO(BKR) repeat for TI

# Undo variance shifts that to reflect accuracy, not precision.
mbar_err_dG_mles = np.sqrt(mbar_err_dG_mles**2 - bias2)
mbar_miderr_dG_mles = np.sqrt(mbar_miderr_dG_mles**2 - bias2)
ti_err_dG_mles = np.sqrt(ti_err_dG_mles**2 - bias2)
ti_miderr_dG_mles = np.sqrt(ti_miderr_dG_mles**2 - bias2)

# For csv reporting, use both TI and MBAR, but only use one for simple
# reporting.
#
report_dGs = mbar_dG_mles
report_dG_ents = mbar_dG_ents
report_err_dGs = mbar_err_dG_mles
if not args.no_sort:
    if dG0_fit is None:
        sort_indices = report_dGs.argsort()
    else:
        sort_indices = dG0_fit.argsort()
        dG0_fit = dG0_fit[sort_indices]
    ligand_names = np.asarray(ligand_names)[sort_indices]
    mbar_dG_mles = mbar_dG_mles[sort_indices]
    mbar_dG_ents = mbar_dG_ents[sort_indices]
    mbar_err_dG_mles = mbar_err_dG_mles[sort_indices]
    mbar_miderr_dG_mles = mbar_miderr_dG_mles[sort_indices]
    ti_dG_mles = ti_dG_mles[sort_indices]
    ti_dG_ents = ti_dG_ents[sort_indices]
    ti_err_dG_mles = ti_err_dG_mles[sort_indices]
    ti_miderr_dG_mles = ti_miderr_dG_mles[sort_indices]
    dG0s = dG0s[sort_indices]
    # Account for sorting!
    report_dGs = report_dGs[sort_indices]
    report_dG_ents = report_dG_ents[sort_indices]
    report_err_dGs = report_err_dGs[sort_indices]

# Optional benchmark linear regression when all dGs have a reference value.
#
if dG0_fit is not None:
    # Running in benchmark mode - do linear regression and shift.
    ntrials = 5000  # bootstrap samples

    def fit(x, y):
        r = np.corrcoef(y, x)[0][1]
        slp = r * y.std() / x.std()
        intrcpt = y.mean() - slp * x.mean()
        yest = slp * x + intrcpt
        rmsr = np.sqrt(((y - yest) ** 2).mean())
        rmse = np.sqrt(((y - x) ** 2).mean())
        mue = (np.abs(y - x)).mean()
        return (r, slp, intrcpt, rmsr, rmse, mue)

    # Adjustment of data to minimize RMSE. This only affects the slope of the
    # linear fit.
    #
    shift = dG0_fit.mean() - report_dGs.mean()
    report_dGs += shift
    mbar_dG_mles += shift
    ti_dG_mles += shift
    r, slp, intrcpt, rmsr, rmse, mue = fit(dG0_fit, report_dGs)
    null_rmse = dG0_fit.std()

    n = dG0_fit.size
    size = n * ntrials
    bs_data = np.zeros((6, ntrials))
    # nb It's faster to generate prngs upfront rather than in the loop.
    masks = np.random.randint(0, n, size=size).reshape(ntrials, n)
    noise = np.random.normal(size=size).reshape(ntrials, n)
    for trial, (maskt, noiset) in enumerate(zip(masks, noise)):
        x = dG0_fit[maskt]
        y = report_dGs[maskt] + report_err_dGs[maskt] * noiset
        bs_data[:, trial] = fit(x, y)
    r_err, slp_err, intrcpt_err, rmsr_err, rmse_err, mue_err = bs_data.std(
        axis=1, ddof=1.5
    )
    print("# Linear fit of MBAR dGs to reference")
    print("# results shifted by %4.2f to minimize RMSE" % shift)
    print("# slope   % 6.2f +/- %4.2f" % (slp, slp_err))
    print("# intrcpt % 6.2f +/- %4.2f" % (intrcpt, intrcpt_err))
    print("# r        % 5.2f +/- %4.2f" % (r, r_err))
    print("# RMSR   %7.2f +/- %4.2f" % (rmsr, rmsr_err))
    print("# MUE    %7.2f +/- %4.2f" % (mue, mue_err))
    print("# RMSE   %7.2f +/- %4.2f" % (rmse, rmse_err))
    print("# RMSE(null) %7.2f" % null_rmse)
    # Switch back values for printing (a bit of a hack, sorry).
    dG0s = dG0_fit

# Print a human-readable summary. Also good for plotting.
#
for ligand, dG, dG_err, ref, dG_ent, pose in zip(
    ligand_names, report_dGs, report_err_dGs, dG0s, report_dG_ents, dominant_poses
):

    ent_str = "" if np.isnan(dG_ent) else " % 6.2f" % dG_ent
    ref_str = "" if np.isnan(ref) else " % 6.2f" % ref
    pose_str = " " if pose == "" else " %s" % pose
    print("%-25s % 6.2f %6.2f%s%s%s" % (ligand, dG, dG_err, ent_str, pose_str, ref_str))

# Copy the results to a new DataFrame with the relevant columns.
#
if not args.new_csv:
    sys.exit(0)

sorted_cols = [
    "Ligand",
    "MBAR_binding_ddG",
    "MBAR_binding_ddG_miderr",
    "MBAR_binding_ddG_sem",
    "TI_binding_ddG",
    "TI_binding_ddG_miderr",
    "TI_binding_ddG_sem",
]
abfe_data = []
for i, ligand in enumerate(ligand_names):
    tmp_dict = {}
    tmp_dict["Ligand"] = ligand
    tmp_dict["MBAR_binding_ddG"] = mbar_dG_mles[i]
    tmp_dict["MBAR_binding_ddG_miderr"] = mbar_miderr_dG_mles[i]
    tmp_dict["MBAR_binding_ddG_sem"] = mbar_err_dG_mles[i]
    tmp_dict["TI_binding_ddG"] = ti_dG_mles[i]
    tmp_dict["TI_binding_ddG_miderr"] = ti_miderr_dG_mles[i]
    tmp_dict["TI_binding_ddG_sem"] = ti_err_dG_mles[i]
    abfe_data.append(tmp_dict)
abfe_df = pd.DataFrame(abfe_data)
abfe_df = abfe_df[sorted_cols]
abfe_df = abfe_df.round(decimals=2)

if args.atm_dat:
    atm_df = abfe_df[['Ligand','MBAR_binding_ddG','MBAR_binding_ddG_miderr']].copy()
    
    atm_df.rename(columns={"Ligand":"ligand", 'MBAR_binding_ddG': "dG", 'MBAR_binding_ddG_miderr':"error"},inplace=True)
    with open(f"{args.result_dir}/atm_diffnet_results.csv","w+") as csv:
        atm_df.to_csv(csv)
else:
    with open(f"{args.result_dir}/bfe_diffnet_results.csv", "w+") as csv:
        abfe_df.to_csv(csv)
