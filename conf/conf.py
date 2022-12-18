import logging
from dynaconf import Dynaconf

logging.basicConfig(level=logging.DEBUG)



settings = Dynaconf(
    envvar_prefix="DYNACONF",
    settings_files=['conf\\settings.toml'],
)