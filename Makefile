# The Makefile stores routines helpful with testing the repo and releases

# Create virtualenv using poetry and install dependencies with dev-dependencies
install:
	poetry install -vv

# Create virtualenv using poetry and install dependencies with dev-dependencies and optional dependencies
install-with-optional:
	poetry install -vv --with optional

# Update lock and create virtualenv using poetry and install dependencies with dev-dependencies
update-install:
	poetry update -vv

# Update lock and create virtualenv using poetry and install dependencies with dev-dependencies and optional dependencies
update-install-with-optional:
	poetry update -vv --with optional

# Create virtualenv using poetry and install dependencies without dev-dependencies
install-no-dev:
	poetry install --no-dev -vv

# Create virtualenv using poetry and install dependencies without dev-dependencies
update-install-no-dev:
	poetry update --no-dev -vv

# Update pip, wheel and setuptools. This step does not assume poetry
install-build-deps:
	python -m pip install -U pip wheel setuptools

# Run formatting on the package
format: isort blue markdown

# Run formatting checks on the package
format-check: isort-check blue-check markdown-check

# Run isort formatting on the package
isort:
	poetry run isort ./

# Run isort formatting check on the package
isort-check:
	poetry run isort --check ./

# Run blue formatting on the package
blue:
	poetry run blue ./

# Run blue formatting check on the package
blue-check:
	poetry run blue --check ./

# Run markdown formatting on the changelog.md
markdown:
	poetry run mdformat changelog.md

# Run markdown formatting check on the changelog.md
markdown-check:
	poetry run mdformat --check changelog.md

# Run package test suit using pytest
test:
	poetry run pytest ./tests

# Run package test suit using pytest
test-no-poetry:
	python -m pytest ./tests

# Run package release acceptance tests - should be custom for the package
release-acceptance-tests: test

# Run mypy check on the package
mypy:
	poetry run mypy

# Run linting checks of the package using mypy
lint: mypy

# Run bandit check on the package source directory (medium severity -ll and low confidence -i)
bandit:
	poetry run bandit -r -ll -i cai_causal_graph

# Run security linting checks of the package using bandit
security-lint: bandit

# Run documentation coverage using interrogate
interrogate:
	poetry run interrogate ./

# Build the package wheel and sdist
build-all:
	poetry build -vv

# Build the package wheel (the binaries)
build-wheel:
	poetry build -f wheel -vv

# Build the package sdist
build-sdist:
	poetry build -f sdist -vv

# Removes a generic wheel that matches the *-py3-*.whl, useful for publishing packages to prevent override
remove-generic-wheel:
	rm -f dist/*-py3-*.whl

# Package and publish the docs to causaLens' artifactory
package-and-publish-docs:
	poetry source add causalens https://causalens.jfrog.io/artifactory/api/pypi/python-open-source/simple/
	poetry config http-basic.causalens ${ARTIFACTORY_USERNAME} ${ARTIFACTORY_PASSWORD}
	poetry add --source=causalens docs-builder@~0.2.0
	poetry run python ./tooling/scripts/docs-upload.py

# Publish the package to PyPI
publish:
	poetry config pypi-token.pypi ${PYPI_TOKEN}
	poetry publish

# Check the package can import
self-import-check:
	python -c "import cai_causal_graph"

# Check the package can import
self-import-check-poetry:
	poetry run python -c "import cai_causal_graph"

# Update pyproject.toml and <package>/__init__.py (the <package> should be set manually below)
set-package-version-linux: .set-toml-version .set-init-version

# Update pyproject.toml and <package>/__init__.py (the <package> should be set manually below)
set-package-version-macosx: .set-toml-version-macosx .set-init-version-macosx

# Changelog update uses a slightly different substitution command + separate for github actions to skip if necessary
set-changelog-version-linux: .set-changelog-version

# Changelog update uses a slightly different substitution command + separate for github actions to skip if necessary
set-changelog-version-macosx: .set-changelog-version-macosx

# This is a working linux sed command, need to add " '' " right after -i for Mac OS X
.set-toml-version:
	sed -i 's/version = "$(CURRENT_VERSION)"/version = "$(NEW_VERSION)"/' pyproject.toml

# This is a working linux sed command, need to add " '' " right after -i for Mac OS X
.set-init-version:
	sed -i "s/__version__ = \(.*\)/__version__ = '$(NEW_VERSION)'/" cai_causal_graph/__init__.py

# This is a working linux sed command, need to add " '' " right after -i for Mac OS X
.set-changelog-version:
	sed -i "s/NEXT/$(NEW_VERSION)/g" changelog.md

# This is a working Mac OS X version
.set-toml-version-macosx:
	sed -i '' 's/version = "$(CURRENT_VERSION)"/version = "$(NEW_VERSION)"/' pyproject.toml

# This is a working Mac OS X version
.set-init-version-macosx:
	sed -i '' "s/__version__ = \(.*\)/__version__ = '$(NEW_VERSION)'/" cai_causal_graph/__init__.py

# This is a working Mac OS X version
.set-changelog-version-macosx:
	sed -i '' "s/NEXT/$(NEW_VERSION)/g" changelog.md
