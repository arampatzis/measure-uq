# Minimal makefile for Sphinx documentation

SPHINXOPTS    ?=
SPHINXBUILD   ?= sphinx-build
SOURCEDIR     = source
BUILDDIR      = build

.PHONY: help html-all Makefile

# Default help target
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

# Custom target: Generate .rst from code and build HTML docs
html-all:
	sphinx-apidoc -f -o $(SOURCEDIR)/measure_uq ../measure_uq/
	sphinx-apidoc -f -o $(SOURCEDIR)/examples ../examples/
	$(SPHINXBUILD) -M html "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

clean:
	rm -rf $(BUILDDIR)/
	rm -rf $(SOURCEDIR)/examples/
	rm -rf $(SOURCEDIR)/measure_uq/

# Catch-all: route unknown targets to sphinx-build
%: Makefile
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
