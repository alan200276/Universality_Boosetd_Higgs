#/bin/bash


date

echo "Start Submission"

# echo "  "

# echo "Downsize"

# bash -e herwig_angular_downsize.sh > herwig_ang_downsize.log 
# sleep 3
# bash -e pythia_default_downsize.sh > pythia_def_downsize.log 
# sleep 3
# bash -e pythia_vincia_downsize.sh > pythia_vin_downsize.log 
# sleep 3
# bash -e pythia_dipole_downsize.sh > pythia_dip_downsize.log 
# sleep 3
# bash -e sherpa_downsize.sh > sherpa_def_downsize.log

echo "  "

echo "Preprocessing"

bash -e herwig_angular_preprocess.sh > herwig_ang_preprocess_.log 
sleep 3
bash -e pythia_default_preprocess.sh > pythia_def_preprocess_.log 
sleep 3
bash -e pythia_vincia_preprocess.sh > pythia_vin_preprocess.log
sleep 3
bash -e pythia_dipole_preprocess.sh > pythia_dip_preprocess.log
# sleep 3
# bash -e sherpa_preprocess.sh > sherpa_def_preprocess.log


echo "Job Submitted"

date