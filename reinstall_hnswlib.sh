set -x
cd index/hnswlib
rm -rf build tmp hnswlib.egg-info
pip uninstall -y hnswlib
pip install . --no-cache-dir
