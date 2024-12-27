import argparse
import requests
import scripts_config

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Purge all data quality for smartcapex data')
    parser.add_argument("--logs", help="whether to purge logs", type=bool)
    parser.add_argument("--history", help="whether to purge logs", type=bool)
    parser.add_argument("--db", help="whether to purge logs", type=bool)
    args = parser.parse_args()

    requests.get(scripts_config.api_url + '/purge',
                 json={'logs': args.logs,
                       'history': args.history,
                       'db': args.db
                       }
                 )
