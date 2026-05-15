#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Stable entrypoint for PyTorch CI to run torchtitan integration tests.
# PyTorch CI calls this script so that torchtitan maintainers can adjust
# test configuration without modifying the PyTorch repo.

set -euo pipefail

OUTPUT_DIR="${OUTPUT_DIR:-artifacts-to-be-uploaded}"
NGPU="${NGPU:-8}"
JUNIT_XML_DIR="${JUNIT_XML_DIR:-}"

usage() {
    echo "Usage: $0 <command>"
    echo ""
    echo "Commands:"
    echo "  feature_tests   Run feature integration tests"
    echo "  model_tests     Run model integration tests"
    echo ""
    echo "Environment:"
    echo "  OUTPUT_DIR      Directory to dump integration test artifacts (default: artifacts-to-be-uploaded)"
    echo "  NGPU            Maximum number of GPUs to use (default: 8)"
    echo "  JUNIT_XML_DIR   Optional directory for JUnit XML reports"
    exit 1
}

if [[ $# -lt 1 ]]; then
    usage
fi

COMMAND="$1"
shift

junit_args=()
if [[ -n "$JUNIT_XML_DIR" ]]; then
    junit_args=(--junit-xml-dir "$JUNIT_XML_DIR")
fi

case "$COMMAND" in
    feature_tests)
        python -m tests.integration_tests.run_tests \
            --test_suite features \
            --exclude "cpu_offload+opt_in_bwd+TP+DP+CP" \
            --ngpu "$NGPU" \
            "${junit_args[@]}" \
            "$OUTPUT_DIR" \
            "$@"
        ;;
    model_tests)
        python -m tests.integration_tests.run_tests \
            --test_suite models \
            --ngpu "$NGPU" \
            "${junit_args[@]}" \
            "$OUTPUT_DIR" \
            "$@"
        ;;
    *)
        echo "Unknown command: $COMMAND"
        usage
        ;;
esac
