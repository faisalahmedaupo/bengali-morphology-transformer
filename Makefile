install:
	pip install -r requirements.txt

data:
	python data/generate_data.py --out_dir data --seed 13

test:
	pytest -q

train:
	python src/train.py --epochs 60 --batch_size 32 --seed 13

evaluate:
	python src/evaluate.py --ckpt_dir checkpoints --data_dir data --split test.tsv

clean:
	rm -rf checkpoints .pytest_cache **/__pycache__
