from PIL import Image
import os

def dataset_img_convert(path):
    for root,dirs,files in os.walk(path):
        for file in dirs:
            for image in os.listdir(path+file):
                img=Image.open(path+file+'/'+image)
                if img.mode!='L':
                    # print(img.mode)
                    img = img.convert('L')
                img=img.resize((220,170))
                img.save(path+file+'/'+image)

# path=r'C:\Users\kongzixiang\Desktop\signature_recognition\signatures/'
# dataset_img_convert(path)