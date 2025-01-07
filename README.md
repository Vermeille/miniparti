Train a VQVAE for my students. This is shit quick & dirty code.
The goal is to take some images with their classes, learn a VQVAE over them, with a constraint that the class must be recoverable from the VQ tokens.
- clf.py contains the tokens => class classifier model used for grading & enforcing the class information not being lost in the VQVAE thanks to a classification loss
- encode.py / encode_test.py encodes images to tokens
- pretrain.py first trains the encoder-decoder with a perceptual loss and a classification loss
- finetunne.py then finetunes the model with a GAN loss to increase image quality. This is faster and more manageable than adding a GAN loss to the pretraining. The GAN uses the new R3 regularizer, which is R1+R2, ie a 0-GP loss on both fake and real images.
- main.py is unused and should be removed.
- vqvae.py contains the code for the encoder / decoder architecture.
