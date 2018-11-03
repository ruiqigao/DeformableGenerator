# Train the model
1. Put the folder of training images into ./data. E.g. a folder 'face'
2. Train the model:
python main.py --category face

# Test the model
python main.py --category face --test True --ckpt model.ckpt-990 --sample_size 100

# Output
***_app.png: generated appearance (without landmarks warping)

***_img.png: generated image (appearance + landmarks)
