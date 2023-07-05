"""
handler for KeyboardInterrupt/Ctrl-C

Ludwig Mohr
ludwig.mohr@icg.tugraz.at

"""
from logging import getLogger


sigint_status = False


def sigint_handler(signum, frame):
    """
    handle KeyboarInterrupt/Ctrl-C

    :param signum:
    :param frame:
    :return:
    """
    logger = getLogger(__name__)
    logger.warning('SIGINT/CTRL-C received. stopping after completing current epoch')

    global sigint_status
    sigint_status = True
