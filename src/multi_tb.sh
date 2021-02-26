#!/bin/bash

multitb() {
    logdir=
    if [ $# -eq 0 ]; then
        printf >&2 'fatal: provide at least one logdir\n'
    fi
    for arg; do
        logdir="${logdir}${logdir:+,}${arg}"
    done
    (set -x; tensorboard --port 8776 --logdir_spec="${logdir}")
}

#echo "$@"
multitb "$@"