DETECTION_SCRIPT = detect.py
VIEWER_SCRIPT = scripts/viewer.py
BATCH_RUNNER = scripts/run-batch.sh
SCORE_SCRIPT = scripts/score.py
NOTEBOOK = detect.ipynb
DATAS_DIR = datas
ARTIFACTS ?= results
OUTPUT_NAME ?= abeille_cool

all: score

dist:
	tar cf ${OUTPUT_NAME}.tar ${DATAS_DIR} ${DETECTION_SCRIPT}

score:
	@echo Computing score...
	python ${SCORE_SCRIPT} ${DETECTION_SCRIPT} ${DATAS_DIR}/test

images:
	@echo Generating images...
	mkdir -p ${ARTIFACTS}
	sh ${BATCH_RUNNER} ${DETECTION_SCRIPT} ${VIEWER_SCRIPT} ${DATAS_DIR}/train ${ARTIFACTS}

notebook:
	@echo Generating notebook from script...
	py2nb ${DETECTION_SCRIPT}
	@echo Rendering notebook...
	jupyter nbconvert --to notebook --execute ${NOTEBOOK} --output=${OUTPUT_NAME}

html: notebook
	@echo Rendering HTML page...
	jupyter nbconvert --to html --execute "${OUTPUT_NAME}.ipynb" --output=index

clean:
	$(RM) ${NOTEBOOK} ${OUTPUT_NAME}.ipynb index.html ${OUTPUT_NAME}.tar
	$(RM) -r results ${ARTIFACTS}
