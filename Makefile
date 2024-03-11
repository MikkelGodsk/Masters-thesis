.PHONY: docs
docs:
	cd docs && \
	sphinx-apidoc -o source/ .. && \
	make html


.PHONY: clean_logs
clean_logs:
	rm -rf *.out
	rm -rf *.err
