PYTHON ?= python3

.PHONY: install clean-data features model test

install:
	$(PYTHON) -m pip install -r requirements.txt

clean-data:
	$(PYTHON) clean_zhvi_hpi_upi.py

features:
	$(PYTHON) merge_new_features.py

model:
	$(PYTHON) model.py

test:
	$(PYTHON) -m unittest discover -s tests -p 'test_*.py'
