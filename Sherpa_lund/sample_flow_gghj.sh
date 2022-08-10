#!/bin/bash

PROCESS="Sherpa_lund"
sherpapath="/root/SHERPA/bin"
HOMEPATH="/home/alan/ML_Analysis/Universality_Boosetd_Higgs"
mcdatapath="/home/u5/Universality_Boosetd_Higgs"
outpath_H="Log/"$PROCESS"_H"
outpath_QCD="Log/"$PROCESS"_QCD"
process="sherpa_lun"

preprocesspath="/home/alan/ML_Analysis/Universality_Boosetd_Higgs/Preprocess"

echo "Start Running"
echo "============================================"
date


# Sherpa Part

rand=$RANDOM
rand="$rand"

nevent=10000

pt_range="250_550"

echo "Random Seed =  $rand "

mkdir $HOMEPATH/$PROCESS/gghj_tmp_"$rand"
cp $HOMEPATH/$PROCESS/gghj.dat  $HOMEPATH/$PROCESS/gghj_tmp_"$rand"/gghj_"$rand".dat

sed -i -e "s/randomseed/"$rand"/g" $HOMEPATH/$PROCESS/gghj_tmp_"$rand"/gghj_"$rand".dat

cd $HOMEPATH/$PROCESS/gghj_tmp_"$rand"/
$sherpapath/Sherpa -f ./gghj_"$rand".dat -R $rand EVENTS=$nevent > $HOMEPATH/$outpath_H/gghj_"$rand".log 

gunzip < $mcdatapath/Sherpa/ggHj_"$process"_"$pt_range"_"$rand".hepmc2g > $mcdatapath/Sherpa/ggHj_"$process"_"$pt_range"_"$rand".hepmc


# Delphes Part
cd /root/Delphes-3.4.2
./DelphesHepMC $HOMEPATH/Cards/delphes_card_HLLHC.tcl $mcdatapath/Sherpa/ggHj_"$process"_"$pt_range"_"$rand".root $mcdatapath/Sherpa/ggHj_"$process"_"$pt_range"_"$rand".hepmc > $HOMEPATH/$outpath_H/ggHj_"$process"_"$pt_range"_"$rand".txt


# Downsize Part
python3 $preprocesspath/downsize.py $mcdatapath/Sherpa/ggHj_"$process"_"$pt_range"_"$rand".root $rand $mcdatapath/Sherpa > $HOMEPATH/$outpath_H/downsize_ggHj_"$process"_"$pt_range"_"$rand".log


# Preprocess Part
python3 $preprocesspath/preprocess_v2.py $mcdatapath/Sherpa/EventList_H_"$process"_"$pt_range"_"$rand".h5 0 $rand > $HOMEPATH/$outpath_H/preprocess_ggHj_"$process"_"$pt_range"_"$rand".log


rm -rf $mcdatapath/Sherpa/ggHj_"$process"_"$pt_range"_"$rand".hepmc2g
rm -rf $mcdatapath/Sherpa/ggHj_"$process"_"$pt_range"_"$rand".hepmc
rm -rf $mcdatapath/Sherpa/ggHj_"$process"_"$pt_range"_"$rand".root
rm -rf $HOMEPATH/$PROCESS/gghj_tmp_"$rand"

echo "Finish"

date