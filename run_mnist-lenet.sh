python3 main_mlsurgery.py 2 0 -MA False -AD 2 -EO 100 -ET 100 -EF 100 -PE 10 -OE 100 -TS 0.50 -GP 1
python3 main_mlsurgery.py 2 0 -MA True  -AD 2 -EO 100 -ET 100 -EF 100 -PE 10 -OE 100 -TS 0.55 -GP 1
python3 main_mlsurgery.py 2 0 -MA True  -AD 2 -EO 100 -ET 100 -EF 100 -PE 10 -OE 100 -TS 0.60 -GP 1
python3 main_mlsurgery.py 2 0 -MA True  -AD 2 -EO 100 -ET 100 -EF 100 -PE 10 -OE 100 -TS 0.65 -GP 1
python3 main_mlsurgery.py 2 0 -MA True  -AD 2 -EO 100 -ET 100 -EF 100 -PE 10 -OE 100 -TS 0.70 -GP 1
python3 main_mlsurgery.py 2 0 -MA True  -AD 2 -EO 100 -ET 100 -EF 100 -PE 10 -OE 100 -TS 0.75 -GP 1
python3 main_mlsurgery.py 2 0 -MA True  -AD 2 -EO 100 -ET 100 -EF 100 -PE 10 -OE 100 -TS 0.80 -GP 1
python3 main_mlsurgery.py 2 0 -MA True  -AD 2 -EO 100 -ET 100 -EF 100 -PE 10 -OE 100 -TS 0.85 -GP 1
python3 main_mlsurgery.py 2 0 -MA True  -AD 2 -EO 100 -ET 100 -EF 100 -PE 10 -OE 100 -TS 0.90 -GP 1
python3 main_mlsurgery.py 2 0 -MA True  -AD 2 -EO 100 -ET 100 -EF 100 -PE 10 -OE 100 -TS 0.95 -GP 1

git fetch
git pull
rm      ./experiment_mnist-lenet/data/*
git add ./experiment_mnist-lenet/
git commit -m "experiment added"
git push 