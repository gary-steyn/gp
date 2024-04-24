init:

test:
	py.test tests
run:
	python3 -O ./src/main.py
.PHONY: init test