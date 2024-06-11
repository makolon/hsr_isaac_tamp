
# Copyright (c) 2018-2022, NVIDIA Corporation
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


def initialize_task(config, env, init_sim=True):
    from .config_utils.sim_config import SimConfig
    sim_config = SimConfig(config)

    # from hsr_rl.tasks.example.hsr_example import HSRExampleTask
    # from hsr_rl.tasks.example.hsr_fetch import HSRExampleFetchTask
    # from hsr_rl.tasks.example.hsr_reach import HSRExampleReachTask
    # from hsr_rl.tasks.example.hsr_pick import HSRExamplePickTask
    # from hsr_rl.tasks.example.hsr_cabinet import HSRExampleCabinetTask
    # from hsr_rl.tasks.residual.hsr_residual_example import HSRResidualExampleTask
    # from hsr_rl.tasks.residual.hsr_residual_fetch import HSRResidualFetchTask
    # from hsr_rl.tasks.residual.hsr_residual_stack import HSRResidualStackTask
    # from hsr_rl.tasks.factory.factory_task_nut_bolt_pick import FactoryTaskNutBoltPick
    from hsr_rl.tasks.gearbox.hsr_gearbox_pick import HSRGearboxPickTask
    from hsr_rl.tasks.gearbox.hsr_gearbox_place import HSRGearboxPlaceTask
    from hsr_rl.tasks.gearbox.hsr_gearbox_insert import HSRGearboxInsertTask
    from hsr_rl.tasks.gearbox.hsr_gearbox_all import HSRGearboxAllTask
    from hsr_rl.tasks.gearbox.hsr_gearbox_residual_pick import HSRGearboxResidualPickTask
    from hsr_rl.tasks.gearbox.hsr_gearbox_residual_place import HSRGearboxResidualPlaceTask
    from hsr_rl.tasks.gearbox.hsr_gearbox_residual_insert import HSRGearboxResidualInsertTask
    from hsr_rl.tasks.gearbox.hsr_gearbox_residual_all import HSRGearboxResidualAllTask

    # Mappings from strings to environments
    task_map = {
        # "HSRExample": HSRExampleTask,
        # "HSRExampleFetch": HSRExampleFetchTask,
        # "HSRExampleReach": HSRExampleReachTask,
        # "HSRExamplePick": HSRExamplePickTask,
        # "HSRExampleCabinet": HSRExampleCabinetTask,
        # "HSRResidualExample": HSRResidualExampleTask,
        # "HSRResidualFetch": HSRResidualFetchTask,
        # "HSRResidualStack": HSRResidualStackTask,
        # "FactoryTaskNutBoltPick": FactoryTaskNutBoltPick,
        "HSRGearboxPick": HSRGearboxPickTask,
        "HSRGearboxPlace": HSRGearboxPlaceTask,
        "HSRGearboxInsert": HSRGearboxInsertTask,
        "HSRGearboxAll": HSRGearboxAllTask,
        "HSRGearboxResidualPick": HSRGearboxResidualPickTask,
        "HSRGearboxResidualPlace": HSRGearboxResidualPlaceTask,
        "HSRGearboxResidualInsert": HSRGearboxResidualInsertTask,
        "HSRGearboxResidualAll": HSRGearboxResidualAllTask
    }

    cfg = sim_config.config
    task = task_map[cfg["task_name"]](
        name=cfg["task_name"], sim_config=sim_config, env=env
    )

    env.set_task(task=task, sim_params=sim_config.get_physics_params(), backend="torch", init_sim=init_sim)

    return task
