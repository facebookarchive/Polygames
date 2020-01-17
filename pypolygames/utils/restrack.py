# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import resource
import os
import subprocess


def get_gpu_usage_nvidia():
    try:
        nvidia_exe = 'nvidia-smi'
        nvquery = "index,utilization.gpu,memory.total,memory.used"
        nvformat = "csv,noheader,nounits"
        stdout = subprocess.check_output([
            nvidia_exe,
            f"--query-gpu={nvquery}",
            f"--format={nvformat}"])
    except subprocess.CalledProcessError as e:
        return f"GPU: ({nvidia_exe} error code {e.returncode})"
    except FileNotFoundError as e:
        return "GPU: (unknown)"
    gpustrs_raw = stdout.decode("utf-8").strip().split(os.linesep)
    gpustrs = []
    for gpustr_raw in gpustrs_raw:
        tokens = gpustr_raw.split(',')
        gpuid = tokens[0].strip()
        gpuutil = tokens[1].strip()
        memtotal = float(tokens[2].strip())
        memused = float(tokens[3].strip())
        gpustr = f"GPU{gpuid}: {gpuutil}%, {memused} MB / {memtotal} MB"
        gpustrs.append(gpustr)
    gpustr = os.linesep.join(gpustrs)
    return gpustr


def get_res_usage_psutil_str():
    import psutil
    p = psutil.Process()
    ru = p.as_dict(attrs=[
        'cpu_num', 'cpu_percent', 'cpu_times', 'num_threads', 'memory_info',
        'memory_percent', 'nice', 'ionice'])
    cpu_num = ru['cpu_num']
    cpu_nthr = ru['num_threads']
    cpu_tusr = ru['cpu_times'].user
    cpu_tsys = ru['cpu_times'].system
    nice = ru['nice']
    ionice = ru['ionice']
    cpustr = f"CPU: {cpu_nthr} threads, Tusr={cpu_tusr}, " \
        f"Tsys={cpu_tsys}"
    #  f"Tsys={cpu_tsys}, nice={nice}, ionice={ionice}"
    mem_rss = ru['memory_info'].rss / 1024 / 1024
    mem_vms = ru['memory_info'].vms / 1024 / 1024
    mem_pcent = ru['memory_percent']
    memstr = f"Mem: RSS {mem_rss:8.2f} MB," \
        f" VMS {mem_vms:8.2f} MB, {mem_pcent:5.2f}%"
    gpustr = get_gpu_usage_nvidia()
    resstr = os.linesep.join([cpustr, gpustr, memstr])
    return resstr


def get_res_usage_no_psutil_str():
    ru = resource.getrusage(resource.RUSAGE_SELF)
    cpustr = f"CPU User={ru.ru_utime} System={ru.ru_stime}"
    gpustr = get_gpu_usage_nvidia()
    memstr = f"Mem maxrss={ru.ru_maxrss}"
    comment = "(install psutil for detailed data)"
    resstr = f"{cpustr}, {gpustr}, {memstr}, {comment}"
    return resstr


def get_res_usage_str():
    try:
        import psutil
        return get_res_usage_psutil_str()
    except (ImportError, ModuleNotFoundError) as e:
        return get_res_usage_no_psutil_str()
