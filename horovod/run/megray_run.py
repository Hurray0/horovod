# Copyright 2019 Uber Technologies, Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import collections
import copy
import errno
import math
import os
import signal
import sys
import threading
import time

def launch_gloo(command, exec_command, settings, nics, env, server_ip):
    """
    Launches the given command multiple times using gloo.
    Each command is launched via exec_command.

    :param command: command to launch
    :param exec_command: means to execute a single command
    :param settings: settings for the distribution
    :param nics: common interfaces
    :param env: environment to use
    :param server_ip: ip to use for rendezvous server
    """
    # allocate processes into slots
    host_alloc_plan = _allocate(settings.hosts, settings.num_proc)

    # create global rendezvous server
    global_rendezv = RendezvousServer(settings.verbose)
    # Start rendezvous server and get port that it is listening
    global_rendezv_port = global_rendezv.start_server(host_alloc_plan)

    run_command = (
        'HOROVOD_GLOO_RENDEZVOUS_ADDR={addr} '
        'HOROVOD_GLOO_RENDEZVOUS_PORT={port} '
        'HOROVOD_CONTROLLER=gloo '
        'HOROVOD_CPU_OPERATIONS=gloo '
        'HOROVOD_GLOO_IFACE={iface} '
        'NCCL_SOCKET_IFNAME={nics} '
        '{command}'  # expect a lot of environment variables
            .format(addr=server_ip,
                    port=global_rendezv_port,
                    iface=list(nics)[0],  # TODO: add multiple ifaces in future
                    nics=','.join(nics),
                    command=' '.join(quote(par) for par in command)))

    # Create a event for communication between threads
    event = threading.Event()

    def set_event_on_sigterm(signum, frame):
        event.set()

    signal.signal(signal.SIGINT, set_event_on_sigterm)
    signal.signal(signal.SIGTERM, set_event_on_sigterm)

    # TODO: Workaround for over-buffered outputs. Investigate how mpirun avoids this problem.
    env = copy.copy(env)  # copy env so we do not leak env modifications
    env['PYTHONUNBUFFERED'] = '1'

    # In case, the main thread receives a SIGINT, the event will be set so the spawned threads can
    # kill their corresponding middleman processes so the jobs can be killed as well.
    alloc_info_to_command = _alloc_info_to_command_fn(run_command, env)
    args_list = [[alloc_info_to_command(alloc_info), alloc_info, event]
                 for alloc_info in host_alloc_plan]

    # Make the output directory if it does not exist
    if settings.output_filename:
        _mkdir_p(settings.output_filename)

    # If an error occurs in one thread, entire process will be terminated.
    # Otherwise, threads will keep running.
    res = threads.execute_function_multithreaded(exec_command, args_list, block_until_all_done=True)

    for name, value in sorted(res.items(), key=lambda item: item[1][1]):
        exit_code, timestamp = value
        if exit_code != 0:
            raise RuntimeError('Gloo job detected that one or more processes exited with non-zero '
                               'status, thus causing the job to be terminated. The first process '
                               'to do so was:\nProcess name: {name}\nExit code: {code}\n'
                               .format(name=name, code=exit_code))


def megray_run(settings, remote_host_names, nics, env, server_ip, command):
    # Each thread will use ssh command to launch the job on each remote host. If an
    # error occurs in one thread, entire process will be terminated. Otherwise,
    # threads will keep running and ssh session.
    exec_command = _exec_command_fn(settings, remote_host_names)
    launch_gloo(command, exec_command, settings, nics, env, server_ip)
