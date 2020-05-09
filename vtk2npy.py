import cardiac_mesh

cardiac_mesh.NumpyFromVTKs(
      folders = "/MULTIX/DATA/INPUT/disk_2/coma/Cardio/meshes/vtk_meshes",
      filename_pattern="*/output.001.vtk",
      dataset_name="/MULTIX/DATA/INPUT/disk_2/coma/Cardio/meshes/LV_all_subjects",
      output_filename = "LVED_all_subjects",
      subj_ids=None, N_subj=None, partition_ids=[1,2]
)
