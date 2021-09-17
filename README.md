# AI Music Generator using ABC notation and Keras

This application was developed for a final year project. It uses a character-based recurrent neural network to generate songs from pre-exisiting ABC notation files. To accompany this, the application will also download each generated song as a MIDI file and also a PNG of the sheet music.

## Prerequisites

```bash
pip install selenium
pip install keras
pip install numpy
```

## Usage main.py

```python
from train import ModelTrain

if __name__ == '__main__':
    trainer = ModelTrain()
    trainer.train(40, 128) #exmaple uses 40 epochs and a batch_size of 128
```
