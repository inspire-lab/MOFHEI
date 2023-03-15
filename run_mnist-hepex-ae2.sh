python3 main_mlsurgery.py 6 1 -MA False -AD 0 -EO 30 -ET 50 -EF 50 -PE 10 -OE 10 -TS 0.50
python3 main_mlsurgery.py 6 1 -MA True  -AD 0 -EO 30 -ET 50 -EF 50 -PE 10 -OE 10 -TS 0.55
python3 main_mlsurgery.py 6 1 -MA True  -AD 0 -EO 30 -ET 50 -EF 50 -PE 10 -OE 10 -TS 0.60
python3 main_mlsurgery.py 6 1 -MA True  -AD 0 -EO 30 -ET 50 -EF 50 -PE 10 -OE 10 -TS 0.65
python3 main_mlsurgery.py 6 1 -MA True  -AD 0 -EO 30 -ET 50 -EF 50 -PE 10 -OE 10 -TS 0.70
python3 main_mlsurgery.py 6 1 -MA True  -AD 0 -EO 30 -ET 50 -EF 50 -PE 10 -OE 10 -TS 0.75
python3 main_mlsurgery.py 6 1 -MA True  -AD 0 -EO 30 -ET 50 -EF 50 -PE 10 -OE 10 -TS 0.80
python3 main_mlsurgery.py 6 1 -MA True  -AD 0 -EO 30 -ET 50 -EF 50 -PE 10 -OE 10 -TS 0.85
python3 main_mlsurgery.py 6 1 -MA True  -AD 0 -EO 30 -ET 50 -EF 50 -PE 10 -OE 10 -TS 0.90
python3 main_mlsurgery.py 6 1 -MA True  -AD 0 -EO 30 -ET 50 -EF 50 -PE 10 -OE 10 -TS 0.95

git fetch
git pull
rm      ./experiment_mnist-hepex-ae2/data/*
git add ./experiment_mnist-hepex-ae2/
git commit -m "experiment added"
git push 