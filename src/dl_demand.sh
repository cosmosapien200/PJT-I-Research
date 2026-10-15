#!/usr/bin/env bash
# DEMAND 16 kHz environments from Zenodo (6 parallel, resumable). Bank 2 test noise.
cd "$(dirname "$0")/../data/raw/demand"
get() {
  e=$1; [ -f ${e}_16k.zip.done ] && return
  for i in 1 2 3 4 5; do
    curl -sL -C - --retry 3 -o ${e}_16k.zip "https://zenodo.org/records/1227121/files/${e}_16k.zip?download=1" \
      && unzip -tq ${e}_16k.zip >/dev/null 2>&1 && touch ${e}_16k.zip.done && echo OK $e && return
  done
  echo FAIL $e
}
export -f get
printf "%s\n" NPARK DKITCHEN DLIVING DWASHING NFIELD NRIVER OHALLWAY OMEETING OOFFICE PCAFETER PRESTO PSTATION SPSQUARE STRAFFIC TBUS TCAR TMETRO | xargs -P 6 -I{} bash -c 'get {}'
echo ALLDONE
