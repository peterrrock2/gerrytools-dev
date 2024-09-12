try:
    # Import extra dependencies for 'mgrp'
    import docker
except ModuleNotFoundError as e:
    raise ModuleNotFoundError(
        f"Optional module 'mgrp' could not be imported because a required dependency is missing: {e}. "
        "To use 'mgrp', please install the necessary dependencies."
    )


from .run_container import RunContainer, RunnerConfig
from .runners.recom import RecomRunnerConfig, RecomRunInfo
from .runners.forest import ForestRunnerConfig, ForestRunInfo
from .runners.smc import SMCRunnerConfig, SMCMapInfo, SMCRedistInfo
import warnings

# There is a bug in the docker SDK package that causes this error to be thrown
# a lot
warnings.filterwarnings(action="ignore", message="unclosed", category=ResourceWarning)


__all__ = [
    "RecomRunnerConfig",
    "RecomRunInfo",
    "ForestRunnerConfig",
    "ForestRunInfo",
    "SMCRunnerConfig",
    "SMCMapInfo",
    "SMCRedistInfo",
    "RunContainer",
]
