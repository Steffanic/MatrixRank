
START "CNN BS=10" "C:/Users/pat/AppData/Local/Programs/Python/Python311/python.exe" .\rank_meta_train.py --model='cnn' --batch_size=10 --max_epochs=10 --weight_decay=0.0001
START "CNN BS=1" "C:/Users/pat/AppData/Local/Programs/Python/Python311/python.exe" .\rank_meta_train.py --model='cnn' --batch_size=1 --max_epochs=10 --weight_decay=0.0001
START "CNN BS=100" "C:/Users/pat/AppData/Local/Programs/Python/Python311/python.exe" .\rank_meta_train.py --model='cnn' --batch_size=100 --max_epochs=10 --weight_decay=0.0001
PAUSE