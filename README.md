# deepAutoHAR
Selección automática de features para detección de actividades humanas en base a sensores de Iphone.

El cógido fue confeccionado bajo Python 3.6, utilizando TensorFlow 1.5, sklearn, numpy, itertools, os
y matplotlib. Es necesario instalar todas estas dependencias antes de poder ejecutarlo. Se
recomienda utilizar Anaconda https://anaconda.org/anaconda/python, por simplicidad. Los experimentos
fueron realizados en una GPU NVIDIA GTX 1070, y es casi imposible completarlos utilizando la versión
CPU de TensorFlow.


Es necesario modificar full_path = '/home/mrlz/Desktop/proj_int/Smartphone_Dataset/' en la línea 282
para indicar la ruta del dataset, según corresponda.

Fuente del dataset: https://www.crcv.ucf.edu/data/UCF-iPhone.php
