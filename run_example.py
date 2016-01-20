# Before running this:
# - add pyMS,pyIMS and pySpatialMetabolomics to the sys.path
# - edit the .json config file 
#       * check data file is correct
#       * set output directory
#       * set directory to save isotope patterns if you don't want to generate each time
# - change the json_filename varible to the correct file
import sys
sys.path.append('/path/to/clone')


# Run Pipeline
from pySpatialMetabolomics import spatial_metabolomics
json_filename = "evaluation/decoy_dataset_chemnoise_real_adducts_fdr_2ppm.json"
spatial_metabolomics.run_pipeline(json_filename)
# View results
from  pySpatialMetabolomics import spatial_metabolomics, fdr_measures
target_adducts = ["H","Na","K"]   
decoy_adducts = ["He", "Li", "Be", "B", "C", "N", "O", "F", "Ne", "Mg", "Al", "Si", "P", "S", "Cl", "Ar", "Ca", "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn", "Ga", "Ge", "As", "Se", "Br", "Kr", "Rb", "Sr", "Y", "Zr", "Nb", "Mo", "Ru", "Rh", "Pd", "Ag", "Cd", "In", "Sn", "Sb", "Te", "I", "Xe", "Cs", "Ba", "La", "Ce", "Pr", "Nd", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Ir", "Th", "Pt", "Pu", "Os", "Yb", "Lu", "Bi", "Pb", "Re", "Tl", "Tm", "U", "W", "Au", "Er", "Hf", "Hg", "Ta"]   
config = spatial_metabolomics.get_variables(json_root+json_filename)
results_fname = spatial_metabolomics.generate_output_filename(config,[],'spatial_all_adducts')
fdr = fdr_measures.decoy_adducts(results_fname,target_adducts,decoy_adducts)
fdr_target=0.1
fdr.decoy_adducts_get_pass_list(fdr_target,n_reps,col='msm')
