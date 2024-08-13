python3 main_mofhei.py 3 0 -MA False -AD 2 -EO 100 -ET 100 -EF 100 -PE 50 -OE 100 -TS 0.50
python3 main_mofhei.py 3 0 -MA True  -AD 2 -EO 100 -ET 100 -EF 100 -PE 50 -OE 100 -TS 0.55
python3 main_mofhei.py 3 0 -MA True  -AD 2 -EO 100 -ET 100 -EF 100 -PE 50 -OE 100 -TS 0.60
python3 main_mofhei.py 3 0 -MA True  -AD 2 -EO 100 -ET 100 -EF 100 -PE 50 -OE 100 -TS 0.65
python3 main_mofhei.py 3 0 -MA True  -AD 2 -EO 100 -ET 100 -EF 100 -PE 50 -OE 100 -TS 0.70
python3 main_mofhei.py 3 0 -MA True  -AD 2 -EO 100 -ET 100 -EF 100 -PE 50 -OE 100 -TS 0.75
python3 main_mofhei.py 3 0 -MA True  -AD 2 -EO 100 -ET 100 -EF 100 -PE 50 -OE 100 -TS 0.80
python3 main_mofhei.py 3 0 -MA True  -AD 2 -EO 100 -ET 100 -EF 100 -PE 50 -OE 100 -TS 0.85
python3 main_mofhei.py 3 0 -MA True  -AD 2 -EO 100 -ET 100 -EF 100 -PE 50 -OE 100 -TS 0.90
python3 main_mofhei.py 3 0 -MA True  -AD 2 -EO 100 -ET 100 -EF 100 -PE 50 -OE 100 -TS 0.95

git fetch
git pull
rm      ./experiment_cifar10-modified-lenet/data/*
git add ./experiment_cifar10-modified-lenet/
git commit -m "experiment added"
git push 
