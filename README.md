
Run Linear Regression model and dump to file:
```bash
python3 entrypoint.py --model linear_regression --dump_model_path=dumped_model.pkl 
```

Run a model loaded from file:
```bash
python3 entrypoint.py  --dumped_model_path=dumped_model.pkl --test-size=0.3
```

Run Random Forest model with custom max_depth and log level:
```bash
python3 entrypoint.py --model random_forest --max_depth=10 --log-level=DEBUG
```