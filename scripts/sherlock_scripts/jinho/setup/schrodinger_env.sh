export SCHROD_LICENSE_FILE=53001@srcc-license-srcf.stanford.edu

# Schrodinger installation
export SCHRODINGER=/scratch/users/zhkim216/software/schrodinger2025-3

# Schrodinger library paths (needed inside container for PrepWizard etc.)
# Container (Ubuntu 24.04) lacks libglib-2.0, libgthread-2.0, libpcre —
# Schrodinger's bundled libs + host system libs copied to oak fill the gap.
export SCHRODINGER_LD_LIBS=$SCHRODINGER/internal/lib:$SCHRODINGER/mmshare-v7.1/lib/Linux-x86_64:/oak/stanford/groups/possu/jinho/libs

# Oak path for bind-mounting into container
export OAK_LIBS=/oak/stanford/groups/possu/jinho/libs
