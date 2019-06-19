# BERT for SLU

SLU = Domain & Intent Detection + Slot Filling

## Run 

Install packages:

```shell
pip install -r requirements.txt
```

Train:

```shell
python main.py --train
```

Evaluate:
```shell
python main.py --eval
```


More parameters:

```python
python main.py -h
```

## Dependency

Python == 3.6.7  
[Allennlp](https://github.com/allenai/allennlp) == 0.8.4

## License

@Apache License 2.0 

