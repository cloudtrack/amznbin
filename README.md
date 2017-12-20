# mxnet-amznbin

## Project Description
mxnet-amznbin is a deep learning image recognition project using [Amazon Bin Image Dataset](https://aws.amazon.com/public-datasets/amazon-bin-images/).

We solve two tasks, 
1. Count: Predict the number of items in the image
2. Classify: Predict the what item is contained in the image 

To reduce the complexity of the tasks and training time,
1. We use images with only one kind of item. - this enables us to solve classify task as a single classification task
2. We specify the minimum number of repetition of an item - most items appear only once in all images, when untreated it will make testing and validating unconvincing. 
3. We manually ruled out a number of invalid images. - check `dataset/invalid_images.json` to see which images are excluded. 

## How to Start
- On root directory run ` pip install -r requirements.txt ` to install required libraries
- Run `make image_dataset` and `make load_metadata` on command line to download the images and metadata (Using tmux would be a good idea)
- After `make load_metadata` is finished, run `make prepare_train` and insert the number of repetition you want (we recommend 20 or higher)
- When all above is finished you are ready to run our program!

## Demo
  - Run `make demo` to checkout our training demo
  - To run the demo of our pretrained model, open `jupyter notebook` and run `src/demo.ipynb` 
    - our pretrained model was too big to upload on github, you can train the model and use your own trained model to use this file

## Results
We accomplished 68.5714% accuracy on count task, and 41.4286% accuracy on classify task
