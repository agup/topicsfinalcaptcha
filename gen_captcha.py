from captcha.image import ImageCaptcha
import matplotlib.pyplot as plt
import string
import random
import numpy as np

def generate(width,height,num_char):
    characters = string.digits + string.ascii_uppercase
    rand_str = ''.join(random.sample(characters,num_char))
    generator = ImageCaptcha(width=width, height=height)
    image = generator.generate_image( ' ' + ' ' + ' ' + ' ' + rand_str[0] + ' ' + rand_str[1] + ' ' + rand_str[2] + ' ' + rand_str[3])
    return (image,rand_str)


# images and labels will be in saved to save_dir folder
save_dir = '/Users/arushigupta/Desktop/topics_data_analytics/final_proj/data_space/'
labels = []
for i in range(20000):
    img, char = generate(400,200,4)
    plt.imsave(save_dir+str(i)+".jpg",np.array(img))
    labels.append(char)
with open(save_dir + 'labels.txt','w') as f:
    for lb in labels:
        f.write('%s\n' % lb)
