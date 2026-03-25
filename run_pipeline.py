import subprocess
import numpy as np
import config


# ****************************
# SETTINGS
# ****************************

RUN_NAME = "run2_energy_loss"
REGENERATE_DATA = 0   # 1 = regenerate dataset, 0 = skip
FORWARD_DIFFUSION = 0   
MODEL_TRAINING = 0
GENERATE_TRAJECTORY = 1
ANALYZE_GENERATOR = 1

# ****************************
# EXPERIMENT CONFIG
# ****************************
print("\n===============================")
print("HAMILTONIAN DIFFUSION EXPERIMENT")
print("===============================\n")

print("Run Name:", RUN_NAME)
print("DT:", config.DT)
print("Epochs:", config.EPOCHS)
print("Lambda energy:", config.LAMBDA_ENERGY)

print("\n===============================\n")


# ****************************
# STEP 1: DATA GENERATION
# ****************************

if REGENERATE_DATA == 1:
    print("Generating dataset...\n")
    subprocess.run(["python", "-m", "data.generate_dataset"])
else:
    print("Skipping dataset generation.\n")

# ===============================
# DATA DETAILS
# ===============================

dataset = np.load(config.DATA_PATH)

print("Dataset shape:", dataset.shape)

print("\nImportant Parameters:")
print("---------------------")

print("Trajectories:", dataset.shape[0])
print("Timesteps:", dataset.shape[1])
print("State dimension:", dataset.shape[2])


# ****************************
# STEP 2: FORWARD DIFFUSION 
# ****************************

if FORWARD_DIFFUSION == 1:
    print("Running forward diffusion visualization...\n")
    subprocess.run(["python", "-m", "diffusion.forward_diffusion"])
else:
    print("Skipping Forward Diffusion.\n")

# ****************************
# STEP 3: TRAIN DIFF. MODEL
# ****************************

if MODEL_TRAINING == 1:
    print("Starting model training...\n")
    subprocess.run(["python", "train_diffusion.py", "--run_name", RUN_NAME])
else:
    print("Skipping Model Training.\n")


# ****************************
# STEP 4: GENERATE TRAJECTORY
# ****************************
if GENERATE_TRAJECTORY == 1:
    print("\nGenerating trajectory from trained model...\n")
    subprocess.run(["python", "generate_trajectory.py", "--run_name", RUN_NAME])
else:
    print("Skipping Generating Trajectory.\n")


# ****************************
# STEP 5: ANALYZE RESULTS
# ****************************

if ANALYZE_GENERATOR == 1:
    print("\nRunning analysis...\n")
    subprocess.run(["python", "analyze_generated.py", "--run_name", RUN_NAME])
else:
    print("Skipping Analyze generated.\n")


print("\n****************************")
print("PIPELINE COMPLETED")
print("****************************\n")

