
clean_dist:
	rm -rf dist/*

create_dist: clean_dist
	python setup.py sdist

upload_package: create_dist
	twine upload dist/*

.PHONY:build test
build:
	rm -rf build
	MAX_JOBS=32 python setup.py install > build.log

test:
	rm -f test.log
	pytest -q -s tests/test_flash_attn_fwd.py > test.log