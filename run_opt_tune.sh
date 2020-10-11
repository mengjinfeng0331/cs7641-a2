#/bin/bash

python flipflop-tune.py RHC &
python flipflop-tune.py SA &
python flipflop-tune.py GA &
python flipflop-tune.py MIMIC &
python fourpeaks-tune.py RHC&
python fourpeaks-tune.py SA&
python fourpeaks-tune.py GA&
python fourpeaks-tune.py MIMIC&
python tsp-tune.py RHC &
python tsp-tune.py SA &
python tsp-tune.py GA &
python tsp-tune.py MIMIC &

