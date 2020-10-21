#!/bin/bash

target=${1}
shift

while :
do
    if [ "${1}" == "" ]; then break; fi
    envname=${1}
    shift

    if [ "${1}" == "" ]; then break; fi
    envvar=${1}
    shift

    eval set envval=\${${envname}}
    if [ "${envval}" == "" ]; then
        eval export ${envname}=${envvar}
    else
        eval export ${envname}=${envvar}:\${${envname}}
    fi
done

${target}

exit $?
