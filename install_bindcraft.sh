#!/bin/bash
################## BindCraft installation script (uv)
################## specify CUDA version for GPU support
cuda=''

# Define the short and long options
OPTIONS=c:
LONGOPTIONS=cuda:

# Parse the command-line options
PARSED=$(getopt --options=$OPTIONS --longoptions=$LONGOPTIONS --name "$0" -- "$@")
eval set -- "$PARSED"

# Process the command-line options
while true; do
  case "$1" in
    -c|--cuda)
      cuda="$2"
      shift 2
      ;;
    --)
      shift
      break
      ;;
    *)
      echo -e "Invalid option $1" >&2
      exit 1
      ;;
  esac
done

echo -e "CUDA: ${cuda:-'not specified (CPU only)'}"

############################################################################################################
############################################################################################################
################## initialisation
SECONDS=0

# set paths and check for uv installation
install_dir=$(pwd)
command -v uv >/dev/null 2>&1 || { echo -e "Error: uv is not installed. Install it with: curl -LsSf https://astral.sh/uv/install.sh | sh"; exit 1; }
echo -e "uv is installed: $(uv --version)"

# check for ffmpeg (system dependency required for animations)
command -v ffmpeg >/dev/null 2>&1 || echo -e "Warning: ffmpeg is not installed. Install it via your system package manager (e.g., apt install ffmpeg, brew install ffmpeg)."

### BindCraft install begin, create virtual environment
echo -e "Installing BindCraft environment\n"
uv venv --python 3.10 "${install_dir}/.venv" || { echo -e "Error: Failed to create BindCraft virtual environment"; exit 1; }

# Load newly created BindCraft environment
echo -e "Loading BindCraft environment\n"
source "${install_dir}/.venv/bin/activate" || { echo -e "Error: Failed to activate the BindCraft environment."; exit 1; }
echo -e "BindCraft environment activated at ${install_dir}/.venv"

# install required Python packages
echo -e "Installing Python dependencies\n"
if [ -n "$cuda" ]; then
  # determine CUDA major version for JAX extras
  cuda_major=$(echo "$cuda" | cut -d. -f1)
  if [ "$cuda_major" = "12" ]; then
    jax_cuda="jax[cuda12]>=0.4,<=0.6.0"
  elif [ "$cuda_major" = "11" ]; then
    jax_cuda="jax[cuda11]>=0.4,<=0.6.0"
  else
    echo -e "Warning: Unrecognized CUDA major version '${cuda_major}', defaulting to cuda12"
    jax_cuda="jax[cuda12]>=0.4,<=0.6.0"
  fi

  uv pip install \
    pandas matplotlib 'numpy<2.0.0' biopython scipy pdbfixer seaborn tqdm jupyter fsspec py3dmol \
    chex dm-haiku 'flax<0.10.0' dm-tree joblib ml-collections immutabledict optax \
    "${jax_cuda}" \
  || { echo -e "Error: Failed to install Python packages."; exit 1; }
else
  uv pip install \
    pandas matplotlib 'numpy<2.0.0' biopython scipy pdbfixer seaborn tqdm jupyter fsspec py3dmol \
    chex dm-haiku 'flax<0.10.0' dm-tree joblib ml-collections immutabledict optax \
    'jax>=0.4,<=0.6.0' \
  || { echo -e "Error: Failed to install Python packages."; exit 1; }
fi

# install PyRosetta
echo -e "Installing PyRosetta\n"
uv pip install pyrosetta-installer || { echo -e "Error: Failed to install pyrosetta-installer"; exit 1; }
python -c 'import pyrosetta_installer; pyrosetta_installer.install_pyrosetta()' || { echo -e "Error: Failed to install PyRosetta. See https://www.pyrosetta.org/ for licensing."; exit 1; }
python -c "import pyrosetta" >/dev/null 2>&1 || { echo -e "Error: pyrosetta module not found after installation"; exit 1; }

# install ColabDesign
echo -e "Installing ColabDesign\n"
uv pip install git+https://github.com/sokrypton/ColabDesign.git --no-deps || { echo -e "Error: Failed to install ColabDesign"; exit 1; }
python -c "import colabdesign" >/dev/null 2>&1 || { echo -e "Error: colabdesign module not found after installation"; exit 1; }

# make sure all required packages were installed
required_modules=(pandas matplotlib numpy Bio scipy pdbfixer seaborn tqdm fsspec py3Dmol chex haiku flax tree joblib ml_collections immutabledict optax jaxlib jax pyrosetta colabdesign)
missing_packages=()

# Check each module can be imported
for mod in "${required_modules[@]}"; do
    python -c "import $mod" >/dev/null 2>&1 || missing_packages+=("$mod")
done

# If any packages are missing, output error and exit
if [ ${#missing_packages[@]} -ne 0 ]; then
    echo -e "Error: The following modules could not be imported:"
    for pkg in "${missing_packages[@]}"; do
        echo -e " - $pkg"
    done
    exit 1
fi

# AlphaFold2 weights
echo -e "Downloading AlphaFold2 model weights \n"
params_dir="${install_dir}/params"
params_file="${params_dir}/alphafold_params_2022-12-06.tar"

# download AF2 weights
mkdir -p "${params_dir}" || { echo -e "Error: Failed to create weights directory"; exit 1; }
wget -O "${params_file}" "https://storage.googleapis.com/alphafold/alphafold_params_2022-12-06.tar" || { echo -e "Error: Failed to download AlphaFold2 weights"; exit 1; }
[ -s "${params_file}" ] || { echo -e "Error: Could not locate downloaded AlphaFold2 weights"; exit 1; }

# extract AF2 weights
tar tf "${params_file}" >/dev/null 2>&1 || { echo -e "Error: Corrupt AlphaFold2 weights download"; exit 1; }
tar -xvf "${params_file}" -C "${params_dir}" || { echo -e "Error: Failed to extract AlphaFold2weights"; exit 1; }
[ -f "${params_dir}/params_model_5_ptm.npz" ] || { echo -e "Error: Could not locate extracted AlphaFold2 weights"; exit 1; }
rm "${params_file}" || { echo -e "Warning: Failed to remove AlphaFold2 weights archive"; }

# chmod executables
echo -e "Changing permissions for executables\n"
chmod +x "${install_dir}/functions/dssp" || { echo -e "Error: Failed to chmod dssp"; exit 1; }
chmod +x "${install_dir}/functions/DAlphaBall.gcc" || { echo -e "Error: Failed to chmod DAlphaBall.gcc"; exit 1; }

# finish
deactivate
echo -e "BindCraft environment set up\n"

############################################################################################################
############################################################################################################
################## cleanup
echo -e "Cleaning up uv cache to save space\n"
uv cache clean
echo -e "uv cache cleaned up\n"

################## finish script
t=$SECONDS
echo -e "Successfully finished BindCraft installation!\n"
echo -e "Activate environment using command: \"source ${install_dir}/.venv/bin/activate\""
echo -e "\n"
echo -e "Installation took $(($t / 3600)) hours, $((($t / 60) % 60)) minutes and $(($t % 60)) seconds."
