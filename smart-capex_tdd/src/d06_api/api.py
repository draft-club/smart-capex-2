import json
import os
import datetime
from pathlib import Path
from flask import Flask, jsonify, request
from flask_wtf import CSRFProtect

from src.d00_conf.conf import conf, conf_loader
from src.d01_utils.utils import load_json, save_json, parse_arguments


app = Flask(__name__)
csrf = CSRFProtect()
csrf.init_app(app)
configs = conf
current_file_path = Path(os.path.dirname(__file__))
parent_folder = current_file_path.parent.absolute()
ARGS = None




@app.route('/', methods=['GET'])
def healthcheck():
    """
    The healthcheck function is a Flask route that handles GET requests to the root URL ('/').
    It returns a JSON response indicating that the API is ready.

    Returns
    -------
    A JSON object with a key message and value "API Densification Ready"
    """
    return jsonify(message="API Densification Ready")


@app.route('/getConfig')
def get_config():
    """
    The get_config function is a Flask route that returns configuration data in JSON format.
    It checks if the global ARGS variable is initialized, constructs a file path based on ARGS,
    and attempts to load a JSON configuration file from that path. If successful, it returns the
    configuration data; otherwise, it returns an error message.

    Returns
    -------
    JSON response containing the configuration data or an error message

    """
    if ARGS is None:
        return jsonify(message="Error, args not Initialized")
    file_path = os.path.join(parent_folder, 'd00_conf', ARGS.path_to_country_parameters)
    try:
        conf_file = load_json(file_path)
        return jsonify(conf_file)
    except FileNotFoundError as e:
        return jsonify(e,
                       message="Une erreur est survenue!: Chargement du fichier de Configuration")

@app.route('/setConfigs', methods=['POST'])
def set_configs():
    """
    The set_configs function is a Flask route that handles POST requests to update a configuration
    file with new JSON data. It decodes the incoming request data, updates the configuration using
    the update_json function, and returns the updated configuration.
    If the configuration file is not found, it catches the FileNotFoundError and returns an error.

    Returns
    -------
    On success: The updated configuration as a JSON response.
    On failure: An error message indicating that the configuration file could not be updated.
    """
    try:
        data = request.data.decode("utf-8")
        new_conf = update_json(data)
        return new_conf
    except FileNotFoundError as e:
        print(e)
        return "Une erreur est survenue!: Mise à jour du fichier de configuration"


def update_json(json_data, config_file_name=os.path.join(parent_folder, 'd00_conf', 'oma.json'),
                overr=False):
    """
    The update_json function updates an existing JSON configuration file with new data.
    It merges the new data into the existing configuration, optionally creates a backup of the old
    configuration, and saves the updated configuration back to the file.

    Parameters
    ----------
    json_data: Json string
        A JSON string containing the new configuration data.
    config_file_name : str
        The path to the configuration file to be updated. Defaults to 'oma.json' in the 'd00_conf'
        directory.
    overr: bool
        A boolean flag indicating whether to overwrite the old configuration without creating a
        backup. Defaults to False.

    Returns
    -------
    old_conf: dict
        Returns the updated configuration as a dictionary.
    """
    old_conf = load_json(config_file_name)
    new_conf = json.loads(json_data)
    for key in new_conf.keys():
        if isinstance(new_conf[key], dict):
            for k in new_conf[key].keys():
                (old_conf[key])[k] = (new_conf[key])[k]
        else:
            old_conf[key] = new_conf[key]
    if not overr:
        old_conf_version = load_json(config_file_name)
        ts = datetime.datetime.today().strftime('%Y%m%d_%H%M%S')
        save_json(old_conf_version, config_file_name.split('.JSON')[0] + ts + '.JSON')
    save_json(old_conf, config_file_name)
    #updatePYconf(old_conf)
    return old_conf





if __name__ == '__main__':
    ARGS = parse_arguments()
    conf_loader(ARGS.path_to_country_parameters)
    app.run(host='0.0.0.0', port=5000, debug=False)
