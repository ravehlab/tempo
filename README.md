# tempo
The TEMPO (Temporally-multiscale prediction) integrator for accelerating molecular simulations

Below is a summary of the main packages needed. Please take a look at the requirements or environment files for a more detailed list of dependencies (including version numbers).
conda create --name py39 "python=3.9" --channel conda-forge

1) Basic scientific stack
conda install -n py39 numpy matplotlib pandas

conda install compilers graph-tool imp

2) Jupyter
conda install -n py39 -c conda jupyter

3) Additional libraries
conda install -n py39 scikit-learn
conda install -n py39 seaborn


# Example parameters for NPC run 
NPC_MSBD.py --depth 0 --time_step 6000 --temperature 298.15 --pikle_file_name "NPC_data_half_v8/radius_v1_$SLURM_ARRAY_TASK_ID" --rmf_filename "NPC_data/test.rmf" --num_frames 60000

# Example parameters for 5 or 10 balls run 
- change the surrogate force function (can also use the same as the NPC)
-  go to the main of the python file and give the parameters e.g n=3, dt=0.2 file_name ="TEST" time_in_fs_sedc = 20000000
