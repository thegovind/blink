.PHONY: test lint serve space leak-scan

test:
	BLINK_MOCK=1 python -m unittest discover -s space -p 'test_*.py'
	BLINK_MOCK=1 python -m unittest discover -s lab/tests -p 'test_*.py'

lint:
	python -m ruff check blink examples scripts space lab

serve:
	python -m blink.server --model $${BLINK_MODEL:-thegovind/blink-4b} --revision $${BLINK_REVISION:-v1.0} --port $${PORT:-8000}

space:
	cd space && BLINK_MOCK=1 BLINK_PORT=$${PORT:-7860} python app.py

leak-scan:
	python scripts/leak_scan.py .
