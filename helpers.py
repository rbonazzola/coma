import glob
import re
import os
import json

class RunCollection():

    def __init__(self, output_dir):
        self.output_dir = output_dir
        self.run_timestamps = self._get_run_timestamps()
        self.run_info = self._build_info_dict()

    def _get_run_timestamps(self):
        regex = re.compile(".*__(.*)_params.json")
        json_file_lst = glob.glob(os.path.join(self.output_dir, "checkpoints/*_params.json"))
        timestamps = [regex.match(x).group(1) for x in json_file_lst]
        return sorted(timestamps)

    def _build_info_dict(self):
        return {x: Run(x, self.output_dir) for x in self.run_timestamps}

    def get_params_df(self):
        ref_conf = self.run_info[self.run_timestamps[0]].params

        dif_cols = [
            col for col in ref_conf.keys()
            if any([runs.run_info[run_id].params.get(col, 0) != ref_conf[col]
                    for run_id in runs.run_info])
        ]

        # dif_cols.remove("dir_name")
        # dif_cols.remove("ids_file")
        # dif_cols.remove("data_file")
        # dif_cols.remove("hash")
        #
        # dif_cols.remove("nTest")
        # dif_cols.remove("nVal")

        params_df = pd.DataFrame([[
            self.run_info[x].params.get(col, None)
            for col in dif_cols] for x in self.run_timestamps],
            columns=dif_cols
        )
        params_df.index = self.run_timestamps
        # params_df.nz = [x[0] for x in params_df.nz]
        params_df.index_label = "run_id"
        return params_df


class Run():

    def __init__(self, run_id, output_dir):
        self.output_dir = output_dir
        self.run_id = run_id
        self._fetch_run_info()

    def _fetch_run_info(self):

        files_for_run = glob.glob(
            os.path.join(self.output_dir, "checkpoints", "*{}*".format(self.run_id))
        )

        # print([os.path.basename(x) for x in files_for_run])

        def _get_chkpt_dir():
            try:
                return [x for x in files_for_run if os.path.isdir(x)][0]
            except:
                pass

        def _get_config_from_id():
            try:
                return json.load(
                    open([x for x in files_for_run if x.endswith("_params.json")][0])
                )
            except:
                pass

        def _get_train_ids():
            try:
                return set([
                    row.strip() for row in open([x for x in files_for_run if x.endswith("_training_ids.txt")][0])
                ])
            except:
                pass

        def _get_val_ids():
            try:
                return set([
                    row.strip() for row in open([x for x in files_for_run if x.endswith("_validation_ids.txt")][0])
                ])
            except:
                pass

        def _get_test_ids():
            try:
                return set([
                    row.strip() for row in open([x for x in files_for_run if x.endswith("_testing_ids.txt")][0])
                ])
            except:
                pass

        def _get_summaries_dir():
            try:
                return glob.glob(os.path.join(self.output_dir, "summaries", "*{}*".format(run_id)))[0]
            except:
                pass

        def _assign_set():
            try:
                training_ids = [(x, "training") for x in self.train_ids]
                validation_ids = [(x, "validation") for x in self.validation_ids]
                testing_ids = [(x, "testing") for x in self.test_ids]
                return dict(training_ids + testing_ids + validation_ids)
            except:
                pass

        self.params = _get_config_from_id()
        self.checkpoints = _get_chkpt_dir()
        self.train_ids = _get_train_ids()
        self.test_ids = _get_test_ids()
        self.validation_ids = _get_val_ids()
        self.summaries = _get_summaries_dir()
        self.logs = None
        self.latent_space = os.path.join(self.output_dir, "latent_space/latent_space__{}.csv".format(self.run_id))
        self.performance_summary = os.path.join(self.output_dir, "performance_summary",
                                                "{}_performance_summary.csv".format(self.run_id))
        self.reconstructed_meshes = os.path.join(self.output_dir, "reconstructions",
                                                 "{}_reconstruction.npy".format(self.run_id))
        self.vtk = os.path.join("vtk/{}__{{subject_id}}.vtk".format(self.run_id))
        self.subset = _assign_set()


def remove_suffix(json_file):
    return json_file[:-12]