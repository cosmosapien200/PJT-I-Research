#!/usr/bin/env bash
# Downloads all raw datasets into data/raw (background; logs to logs/dl)
set -u
R=$(cd "$(dirname "$0")/.." && pwd)
D=$R/data/raw; mkdir -p $D; cd $D
dl() { # url outfile
  [ -s "$2.done" ] && return 0
  wget -q -c -O "$2" "$1" && touch "$2.done" && echo "OK $2" || echo "FAIL $1"
}
P=https://physionet-open.s3.amazonaws.com/challenge-2016/1.0.0
(dl $P/training.zip physionet2016_training.zip; dl $P/annotations.zip physionet2016_annotations.zip) &
dl https://physionet.org/static/published-projects/circor-heart-sound/the-circor-digiscope-phonocardiogram-dataset-1.0.3.zip circor.zip &
(dl https://www.openslr.org/resources/31/dev-clean-2.tar.gz mini_librispeech_dev-clean-2.tar.gz; dl https://www.openslr.org/resources/12/test-clean.tar.gz librispeech_test-clean.tar.gz) &
dl https://github.com/karoldvl/ESC-50/archive/master.zip esc50.zip &
(mkdir -p demand; for e in DKITCHEN DLIVING DWASHING NFIELD NPARK NRIVER OHALLWAY OMEETING OOFFICE PCAFETER PRESTO PSTATION SCAFE SPSQUARE STRAFFIC TBUS TCAR TMETRO; do dl https://zenodo.org/api/records/1227121/files/${e}_16k.zip/content demand/${e}_16k.zip; done) &
wait
echo ALLDONE
