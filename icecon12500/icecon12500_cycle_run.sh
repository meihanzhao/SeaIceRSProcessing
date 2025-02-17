#!/bin/bash

cd /mnt/jqhyqx/algorithm/icecon12500/

docker run --rm -v /mnt:/mnt osice:v1.2 /anaconda/bin/python /mnt/jqhyqx/algorithm/icecon12500/NSIDC-AA_pre.py
docker run --rm -v /mnt:/mnt osice:v1.2 /anaconda/bin/python /mnt/jqhyqx/algorithm/icecon12500/OISST_pre.py
docker run --rm -v /mnt:/mnt osice:v1.2 /anaconda/bin/python /mnt/jqhyqx/algorithm/icecon12500/OSTIA_pre.py
docker run --rm -v /mnt:/mnt osice:v1.2 /anaconda/bin/python /mnt/jqhyqx/algorithm/icecon12500/OSI-401-d_pre.py
docker run --rm -v /mnt:/mnt osice:v1.2 /anaconda/bin/python /mnt/jqhyqx/algorithm/icecon12500/UB-AMSR2_pre.py
docker run --rm -v /mnt:/mnt osice:v1.2 /anaconda/bin/python /mnt/jqhyqx/algorithm/icecon12500/NSIDC-Bootstrap_125pre.py

wait
docker run --rm -v /mnt:/mnt osice:v1.2 /anaconda/bin/python /mnt/jqhyqx/algorithm/icecon12500/NSIDC-AA_PDFcorrected.py

wait
docker run --rm -v /mnt:/mnt osice:v1.2 /anaconda/bin/python /mnt/jqhyqx/algorithm/icecon12500/Kalman.py

wait
docker run --rm -v /mnt:/mnt osice:v1.2 /anaconda/bin/python /mnt/jqhyqx/algorithm/icecon12500/checking12500.py

