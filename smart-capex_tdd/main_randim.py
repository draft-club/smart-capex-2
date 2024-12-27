import logging
import logging.config
import time

from src.d00_conf.conf import conf, conf_loader
from src.d01_utils.utils import parse_arguments, setup_logging
from src.tdd_pipeline import preprocessing_pipeline, get_randim_densification_result
from src.tdd_pipeline import forecast_pipeline
from src.tdd_pipeline import conversion_rate_pipeline
from src.tdd_pipeline import push_to_db_pipeline
from src.tdd_pipeline import prepare_densification_topology_file
from src.tdd_pipeline import train_densification_model_pipeline
from src.tdd_pipeline import apply_densification_model_pipeline
from src.tdd_pipeline import densification_to_economical_pipeline, density_economical_pipeline


def main():
    """
    Main function of Densification module

    Parameters
    ----------
    country: str
        The name of the coutry can be 'OMA' or 'OCI' for exemple
    """
    st = time.time()
    args = parse_arguments()
    print(args.path_to_country_parameters)
    conf_loader(args.path_to_country_parameters)
    tech = conf['PREPROCESSING']['TECH']

    logging.error("Start preprocessing module")
    print("---------------------------------------------------------\n"
          "Start preprocessing module\n"
          "---------------------------------------------------------")
    preprocessing_pipeline(tech=tech, pod=True)

    logging.info("Start forecast module")
    print("---------------------------------------------------------\n"
          "Start forecast module\n"
          "---------------------------------------------------------")
    forecast_pipeline()

    logging.info("Start conversion rate module")
    print("---------------------------------------------------------\n"
          "Start conversion rate module\n"
          "---------------------------------------------------------")
    conversion_rate_pipeline(compute_rate=True, compute_export_to_randim=True)

    logging.info("Start Create topology file")
    print("---------------------------------------------------------\n"
          "Create topology file\n"
          "---------------------------------------------------------")
    topology_randim = prepare_densification_topology_file()
    logging.debug("Type of df topology randim is: %s", type(topology_randim))

    logging.info("Start Train Traffic Gain Model")
    print("---------------------------------------------------------\n"
          "Train traffic gain model\n"
          "---------------------------------------------------------")
    train_densification_model_pipeline()

    logging.info("Start Get Randim Densification Result")
    print("---------------------------------------------------------\n"
          "Get Randim Densification Result\n"
          "---------------------------------------------------------")
    get_randim_densification_result()

    logging.info("Apply densification gain model")
    print("---------------------------------------------------------\n"
          "Apply densification gain model\n"
          "---------------------------------------------------------")
    apply_densification_model_pipeline()

    logging.info("Start module Technical to economcic")
    print("---------------------------------------------------------\n"
          "Start module Technical to economcic\n"
          "---------------------------------------------------------")
    densification_to_economical_pipeline()
    print("Done technical to economical pipeline")

    logging.info("Start Economic module")
    print("---------------------------------------------------------\n"
          "Start Economic module\n"
          "---------------------------------------------------------")
    df_npv = density_economical_pipeline()
    logging.debug("Type of df Npv is: %s", type(df_npv))

    logging.info("Start push to db module")
    print("---------------------------------------------------------\n"
          "Start push to db module\n"
          "---------------------------------------------------------")
    push_to_db_pipeline(forecast=False, congestion=False, densification_congested_cell=False)
    logging.info("Pipeline executed in %s second",time.time() - st)

if __name__ == "__main__":
    setup_logging()
    logger = logging.getLogger(__name__)
    logging.info("Program started")
    main()
    logging.info("Program finished")
