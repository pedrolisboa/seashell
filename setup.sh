# conda install -c conda-forge pyside2 
apt-get install -y libasound-dev portaudio19-dev libportaudio2 libportaudiocpp0
apt-get install -y docker.io

docker build -t pequod .

mkdir -p audios
