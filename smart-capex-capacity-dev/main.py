from src.d01_utils.utils import timer, get_running_config, set_up_logs

def main(running_configuration):
    """
    main function
    """
    logger = set_up_logs()

    logger.info("Running pipeline")


if __name__ == '__main__':
    running_configuration = get_running_config(configfile="running_config.ini")
    main(running_configuration)
