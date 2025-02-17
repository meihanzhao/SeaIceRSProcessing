#!/bin/bash

cd /mnt/jqhyqx/algorithm/icecon12500/

docker run --rm -v /mnt:/mnt osice:v1.2 /anaconda/bin/python /mnt/jqhyqx/algorithm/icecon25000/NSIDC-Bootstrap_pre.py
docker run --rm -v /mnt:/mnt osice:v1.2 /anaconda/bin/python /mnt/jqhyqx/algorithm/icecon25000/OISST_pre.py
docker run --rm -v /mnt:/mnt osice:v1.2 /anaconda/bin/python /mnt/jqhyqx/algorithm/icecon25000/OSI-401-d_pre.py
docker run --rm -v /mnt:/mnt osice:v1.2 /anaconda/bin/python /mnt/jqhyqx/algorithm/icecon25000/NISE_pre.py
docker run --rm -v /mnt:/mnt osice:v1.2 /anaconda/bin/python /mnt/jqhyqx/algorithm/icecon25000/FY-3D_pre.py
docker run --rm -v /mnt:/mnt osice:v1.2 /anaconda/bin/python /mnt/jqhyqx/algorithm/icecon25000/FY-3C_pre.py

wait
docker run --rm -v /mnt:/mnt osice:v1.2 /anaconda/bin/python /mnt/jqhyqx/algorithm/icecon25000/NSIDC-AA_PDFcorrected.py

wait
docker run --rm -v /mnt:/mnt osice:v1.2 /anaconda/bin/python /mnt/jqhyqx/algorithm/icecon25000/Kalman.py

wait
docker run --rm -v /mnt:/mnt osice:v1.2 /anaconda/bin/python /mnt/jqhyqx/algorithm/icecon25000/checking12500.py


#----------------------------------------------------------------------------------------------------------------------------
cd /mnt/jqhyqx/algorithm/icecon12500/

/anaconda/bin/python /mnt/jqhyqx/algorithm/icecon12500/NSIDC-AA_pre.py &
/anaconda/bin/python /mnt/jqhyqx/algorithm/icecon12500/OISST_pre.py &
/anaconda/bin/python /mnt/jqhyqx/algorithm/icecon12500/OSTIA_pre.py &
/anaconda/bin/python /mnt/jqhyqx/algorithm/icecon12500/OSI-401-d_pre.py &
/anaconda/bin/python /mnt/jqhyqx/algorithm/icecon12500/UB-AMSR2_pre.py &
/anaconda/bin/python /mnt/jqhyqx/algorithm/icecon12500/NSIDC-Bootstrap_125pre.py 
wait
/anaconda/bin/python /mnt/jqhyqx/algorithm/icecon12500/NSIDC-AA_PDFcorrected.py 
wait
/anaconda/bin/python /mnt/jqhyqx/algorithm/icecon12500/Kalman_V1.1.py


docker run --rm -v /mnt:/mnt osice:v1.2 bash run.all.sh 

