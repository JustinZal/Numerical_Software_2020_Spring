venv:
	python3 -m venv venv

packages:
	pip3 install -r requirements.txt

clean:
	rm -rf ./venv

pdf:
	rm -rf ./*.pdf
