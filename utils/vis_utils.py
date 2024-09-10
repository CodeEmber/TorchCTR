import os
import json
import pandas as pd
import shutil
import re
from utils.utilities import get_file_path


class ResultsManager:

    def __init__(self):
        """
        Initialize ResultsManager class.
        """
        self.base_path = get_file_path(["results"])
        self.file_dict = self.get_files(self.base_path)
        self.model_list = list(self.file_dict.keys())
        self.unreferenced_files = []
        self.metrics_data = []
        self.runinfo_file_name = []

    def select_model(self, model_name):
        """
        Select a model.
        """
        self.model_name = model_name
        self.model_info = self.get_model_info(self.file_dict, model_name)
        self.unreferenced_files = []
        self.metrics_data = []
        self.json_directory = get_file_path(["results", model_name, "run_info"])

    def get_files(self, path):
        """
        Get all files and directories under the specified path, stored as a dictionary.
        """
        files = os.listdir(path)
        file_dict = {}
        for file in files:
            if file.startswith("events.out.") or file.startswith("."):
                continue
            full_path = os.path.join(path, file)
            if os.path.isdir(full_path):
                file_dict[file] = self.get_files(full_path)
            else:
                file_dict[file] = None
        return file_dict

    def get_model_info(self, file_dict, model):
        """
        Get information related to the specified model.
        """
        return file_dict.get(model, None)

    def get_unreferenced_files(self, files, run_info_files, folder_name):
        """
        Get files that are not in run_info.
        """
        for file in files:
            if file.split('.')[0] not in run_info_files:
                full_path = os.path.join(self.base_path, self.model_name, folder_name, file)
                self.unreferenced_files.append(full_path)

    def delete_unreferenced_files(self):
        """
        Delete files and directories that are not in run_info.
        """
        for full_path in self.unreferenced_files:
            print(f"Deleting file: {full_path}")
            if os.path.isdir(full_path):
                shutil.rmtree(full_path)
            else:
                os.remove(full_path)

    def clean_up(self):
        """
        Execute cleanup operations to delete unnecessary files.
        """
        if self.model_info is None:
            print(f"Model '{self.model_name}' not found.")
            return []

        run_info_files = [file.split('.')[0] for file in self.model_info['run_info'].keys()]

        save_tensorboard_files = self.model_info['save_tensorboard'].keys()
        save_model_files = self.model_info['save_model'].keys()
        evaluation_files = self.model_info['evaluation'].keys()

        self.get_unreferenced_files(save_tensorboard_files, run_info_files, "save_tensorboard")
        self.get_unreferenced_files(save_model_files, run_info_files, "save_model")
        self.get_unreferenced_files(evaluation_files, run_info_files, "evaluation")
        return self.unreferenced_files

    def get_run_info_data(self, run_info_name: str):
        """
        Get data from the run_info file.
        """
        run_info_path = get_file_path(["results", run_info_name.split("_")[0], "run_info", f"{run_info_name}"])
        with open(run_info_path, 'r') as file:
            run_info_data = json.load(file)
            re_list = ['.*slack.*', '.*logger.*', '.*is.*', '.*path.*', '.*mat.*', '.*col.*']
            for re_str in re_list:
                run_info_data = {k: v for k, v in run_info_data.items() if not re.match(re_str, k)}
        return run_info_data

    def extract_metrics(self):
        """
        Extract best_metric data from all JSON files in the specified directory.
        """
        for filename in os.listdir(self.json_directory):
            if filename.endswith('.json'):
                self.runinfo_file_name.append(filename)
                file_path = os.path.join(self.json_directory, filename)
                try:
                    with open(file_path, 'r') as file:
                        json_data = json.load(file)

                        # Extract information
                        model_name = json_data.get('model_name', None)
                        data_name = json_data.get('data', None)

                        # Extract best_metric's metric and epoch
                        best_metric = json_data.get('best_metric', {})
                        metrics = best_metric.get('metric', {})
                        epoch = best_metric.get('epoch', None)

                        # Add information to metrics list
                        metrics['epoch'] = epoch
                        metrics['model_name'] = model_name
                        metrics['data_name'] = data_name
                        metrics['time'] = filename[:-5].split('_')[-1]  # Extract time from filename

                        # Append extracted dictionary to metrics list
                        self.metrics_data.append(metrics)

                except Exception as e:
                    print(f"Error reading {file_path}: {e}")

    def to_dataframe(self):
        """
        Convert extracted data to a DataFrame and move model_name and data_name to the front.
        """
        if not self.metrics_data:
            return pd.DataFrame()
        df = pd.DataFrame(self.metrics_data)

        # Move model_name and data_name to the front
        cols = ['model_name', 'data_name'] + [col for col in df.columns if col not in ['model_name', 'data_name']]
        df = df[cols]

        return df


def main():
    # Specify the model name
    model_name = "sgl"  # Replace with your model name

    # Create ResultsManager instance and extract data
    manager = ResultsManager()

    # Execute cleanup
    manager.clean_up()

    # Extract metrics and convert to DataFrame
    manager.extract_metrics()
    df = manager.to_dataframe()

    # Print DataFrame
    print(df)
    return df


# Execute main function
if __name__ == "__main__":
    main()
