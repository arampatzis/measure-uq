# Makefile for building Sphinx documentation

# Variables
SPHINXBUILD  = sphinx-build
SOURCEDIR    = source
BUILDDIR     = build
APIDOCFLAGS  = -f -e -P

.PHONY: help html html-all clean

# Show help message
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)"

# Build only existing HTML docs (assumes .rst files are up to date)
html:
	@$(SPHINXBUILD) -M html "$(SOURCEDIR)" "$(BUILDDIR)"

# Regenerate .rst files from code, then build HTML
html-all:
	sphinx-apidoc $(APIDOCFLAGS) -o $(SOURCEDIR)/measure_uq ../measure_uq/
	sphinx-apidoc $(APIDOCFLAGS) -o $(SOURCEDIR)/examples ../examples/
	@$(SPHINXBUILD) -M html "$(SOURCEDIR)" "$(BUILDDIR)"

# Remove generated files
clean:
	rm -rf "$(BUILDDIR)" \
	       "$(SOURCEDIR)/examples" \
	       "$(SOURCEDIR)/measure_uq"
