venv:
	python3 -m venv venv

packages:
	pip3 install -r requirements.txt

clean:
	rm -rf ./venv

file_clean:
	rm -rf ./*.pdf ./*.log ./*.aux ./*.gz ./*.csv

generate_documentation:
	pdflatex -synctex=1 -interaction=nonstopmode "documentation".tex
