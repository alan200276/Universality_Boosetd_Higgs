#!/bin/bash

PROCESS="Sherpa_default"
sherpapath="/root/SHERPA/bin"
HOMEPATH="/home/alan/ML_Analysis/Universality_Boosetd_Higgs"
mcdatapath="/home/u5/Universality_Boosetd_Higgs"
outpath_H="Log/"$PROCESS"_H"
outpath_QCD="Log/"$PROCESS"_QCD"
process="sherpa_def"

preprocesspath="/home/alan/ML_Analysis/Universality_Boosetd_Higgs/Preprocess"

echo "Start Running"
echo "============================================"
date


# Sherpa Part

rand=$RANDOM
rand="$rand"

nevent=100

pt_range="250_550"

echo "Random Seed =  $rand "

mkdir $HOMEPATH/$PROCESS/ppjj_tmp_"$rand"
cp $HOMEPATH/$PROCESS/ppjj.dat  $HOMEPATH/$PROCESS/ppjj_tmp_"$rand"/ppjj_"$rand".dat

sed -i -e "s/randomseed/"$rand"/g" $HOMEPATH/$PROCESS/ppjj_tmp_"$rand"/ppjj_"$rand".dat

cd $HOMEPATH/$PROCESS/ppjj_tmp_"$rand"/
$sherpapath/Sherpa -f ./ppjj_"$rand".dat -R $rand EVENTS=$nevent > $HOMEPATH/$outpath_QCD/ppjj_"$rand".log 

gunzip < $mcdatapath/Sherpa/ppjj_"$process"_"$pt_range"_"$rand".hepmc2g > $mcdatapath/Sherpa/ppjj_"$process"_"$pt_range"_"$rand".hepmc


# Delphes Part
cd /root/Delphes-3.4.2
./DelphesHepMC $HOMEPATH/Cards/delphes_card_HLLHC.tcl $mcdatapath/Sherpa/ppjj_"$process"_"$pt_range"_"$rand".root $mcdatapath/Sherpa/ppjj_"$process"_"$pt_range"_"$rand".hepmc > $HOMEPATH/$outpath_QCD/ppjj_"$process"_"$pt_range"_"$rand".txt


# Downsize Part
python3 $preprocesspath/downsize.py $mcdatapath/Sherpa/ppjj_"$process"_"$pt_range"_"$rand".root $rand $mcdatapath/Sherpa > $HOMEPATH/$outpath_QCD/downsize_ppjj_"$process"_"$pt_range"_"$rand".log


# Preprocess Part
python3 $preprocesspath/preprocess_v2.py $mcdatapath/Sherpa/EventList_QCD_"$process"_"$pt_range"_"$rand".h5 0 $rand > $HOMEPATH/$outpath_QCD/preprocess_ppjj_"$process"_"$pt_range"_"$rand".log


rm -rf $mcdatapath/Sherpa/ppjj_"$process"_"$pt_range"_"$rand".hepmc2g
rm -rf $mcdatapath/Sherpa/ppjj_"$process"_"$pt_range"_"$rand".hepmc
rm -rf $mcdatapath/Sherpa/ppjj_"$process"_"$pt_range"_"$rand".root
rm -rf $HOMEPATH/$PROCESS/ppjj_tmp_"$rand"

echo "Finish"

date