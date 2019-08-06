This iris flower folder contains program for nn based classification 
for iris flower properties saved in Dataset/iris.csv file
and is a classical example on how you can train your basic tensorflow model with 
"hello world" iris dataset.

This example has commented out the saving of the model so you can save
it simply by uncommenting the model.save line.

This particular code also does different variations of the model whose properties you can
change in the DENSE_LAYERS/LAYER_SIZE/SEPOCHS arrays. The statistics of those models will be saved
in logs file and can be later opened with tensorboard.

Dataset structure is as follows:

  sepal_length/sepal_width/petal_length/petal_width/species
  
