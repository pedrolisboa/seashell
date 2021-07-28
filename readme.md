## Seashell Audio Monitoring

![Alt](/images/example.png "Title")

### Setup

Obs: only works with python versions <= 3.7

```
sudo sh setup.sh
pip install -r requirements.txt
```

### Setup for Docker

```
xhost +local:docker
```
Build the image


Run the script to start the container

### Observations

The program searches for audio files inside "audios" folder and model configuration files inside models/configs.


### To-do

* Sync naming styles
* Refactor code
* Set up mutexes
* Multi-model model plot synchronization
  * Deactivate single model lock 
* Redesign data feeding as subscriber model
* Decouple lofar processing from spectrogram plot objects
  * Create decoupled object from subscriber model
* Data history functionalities 
  * Setup data history on data subscriber model
  * Link Audio file manipulation
  * Link plots
  * Link lofar configuration
* Setup support for DEMON
* Improve GUI
* Add option to place Grad-CAM mask over spectrogram when model is loaded and has the same LOFAR config (or change config auto)


