#!/bin/bash

SCRIPT_DIR=$(cd $(dirname $0); pwd)

sphinx-apidoc -f -o "${SCRIPT_DIR}/source/api" "${SCRIPT_DIR}/../ion"
