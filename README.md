# Train

```bash
./setup.sh
python3 train.py -c src/configs/train.json
```

# Test

```bash
./setup.sh
python3 test.py -c src/configs/train.json -r model_best.pth
```