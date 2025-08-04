.PHONY: lint test build all

lint:
	flake8 src/ --max-line-length=120

test:
	pytest src/

build:
	docker compose build

all: lint test build
