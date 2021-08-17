


* Signal (gluon gluon production of Higgs bosons in $b\bar{b}b\bar{b}$ final state)  
    * UFO [The Higgs Characterisation model](https://feynrules.irmp.ucl.ac.be/wiki/HiggsCharacterisation)  
    * process:   
     ```
    import model HC_NLO_X0_UFO-heft
    generate p p > x0 /t [QCD] @0
    ```
    
    * decay in Madspin: 
    ```
    set spinmode none
    decay x0 > b b~
    ```
    
    * run_card setting:  
    ```
    set run_card nevents 100000
    set run_card ebeam1 7000.0
    set run_card ebeam2 7000.0

    set pdlabel lhapdf
    set lhaid 91500 #PDF4LHC15_nnlo_mc

    set pt_min_pdg {25:250}
    set pt_max_pdg {25:500}
    ```
    
    
    * process card:
    ```
    proc_ggH.txt
    ```




* Background (QCD multijet )   
    * process:   
    ```
    define p = p b b~
    define j = j b b~

    generate p p > j j
    ```  
    
    * run_card setting:
    ```
    set run_card nevents 100000
    set run_card ebeam1 7000.0
    set run_card ebeam2 7000.0
    set run_card pdlabel lhapdf 
    set run_card lhaid 260000  #NNPDF30_nlo_as_0118

    set run_card ihtmin 250
    set run_card ihtmax 500
    ```
    * process card:
    ```
    proc_ppjj.txt
    ```