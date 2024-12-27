import os
import shutil

def delete_files_and_folders(paths):
    for path in paths:
        if os.path.isfile(path):
            try:
                os.remove(path)
                print(f"Deleted file: {path}")
            except FileNotFoundError as e:
                print(f"Failed to delete file {path}: {e}")
        elif os.path.isdir(path):
            try:
                shutil.rmtree(path)
                print(f"Deleted folder: {path}")
            except FileNotFoundError as e:
                print(f"Failed to delete folder {path}: {e}")
        else:
            print(f"Invalid path: {path}")

if __name__ == "__main__":
    path_lists = ['data/samples/04_models/traffic_trend_by_site_region',
                  'data/samples/05_models_output/final_npv',
                  'data/samples/05_models_output/increase_in_cash_flow_due_to_the_upgrade',
                  'data/samples/05_models_output/increase_in_traffic_due_to_the_upgrade',
                  'data/samples/05_models_output/increase_in_traffic_due_to_the_upgrade_splitted',
                  'data/samples/05_models_output/increase_site_margin',
                  'data/samples/05_models_output/site_margin']
    delete_files_and_folders(path_lists)
