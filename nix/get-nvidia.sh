#!/bin/sh

nvidiaVersion=$(nvidia-smi --query-gpu=driver_version --format=csv | tail -n 1)
echo "detected: ${nvidiaVersion}"

nvidiaUrl="http://download.nvidia.com/XFree86/Linux-x86_64/${nvidiaVersion}/NVIDIA-Linux-x86_64-${nvidiaVersion}.run"
nvidiaSha256=$(nix-prefetch-url ${nvidiaUrl})

OUTNAME="nvidia.json"
echo "{" > ${OUTNAME}
echo "    \"nvidiaVersion\" : \"${nvidiaVersion}\"," >> ${OUTNAME}
echo "    \"nvidiaSha256\" : \"${nvidiaSha256}\"" >> ${OUTNAME}
echo "}" >> ${OUTNAME}
echo "${OUTNAME} written"

