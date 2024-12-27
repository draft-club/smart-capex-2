# flask args
flask_host = '0.0.0.0'
flask_port = 5000

# logs
log_file = 'logs/ocdvt-dv.log'
log_max_byte = 5e8
log_backup_count = 2

# Reports db args
anomalies_table = 'dv_report'
examples_table = 'dv_examples'
url = "mysql+pymysql://"
validation_host = "10.238.36.20"
validation_user = ""
validation_password = ""
validation_database = ""

# Supervision db args
supervision_table = 'dv_monitoring'
supervision_host = "10.240.30.196"
supervision_user = ""
supervision_password = ""
supervision_database = ""
supervision_project = "ocdvt-dv"
supervision_job = "validation"
