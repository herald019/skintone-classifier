import cv2
import matplotlib.pyplot as plt
import numpy as np
import math

#input image file
imagePath = 'cobe.jpg'
img = cv2.imread(imagePath)
#convert to greyscale
gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# print(gray_image.shape)

#face recognition model
face_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
#box cordinates
face = face_classifier.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40))


#ploting the box
for (x, y, w, h) in face:
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

#face-box coords
x = face[0][0]
y = face[0][1]
w = face[0][2]
h = face[0][3]
print(x, y, w, h)
#croping face out
face_cropped_img = img_rgb[y:y+h, x:x+w]

#centre coords
centre_x = x + w/2
centre_y = y + h/2

start_x = math.ceil(centre_x - w*0.15)
end_x = math.floor(centre_x + w*0.15)
start_y = math.ceil(centre_y - h*0.15)
end_y = math.floor(centre_y + h*0.15)

selection_using_centre = img_rgb[start_y:end_y , start_x: end_x]

#rectangle around selection
centre_rectangle = cv2.rectangle(img, (start_x, start_y), (end_x, end_y), (0, 0, 255), 2) 
centre_rectangle_rgb = cv2.cvtColor(centre_rectangle, cv2.COLOR_BGR2RGB)

#croping forehead out
# hp = round(h * 0.8)
# wp = round(w * 0.2)
# forehead_cropped_img = img_rgb[y+4 : y+h-hp ,  x+wp : x+w-wp]


#average finding
avg_coords = selection_using_centre.mean(axis=0).mean(axis=0)
dec_avg_coords = [(x/255) for x in avg_coords]
# print(avg_coords)
# print(dec_avg_coords)

#classification
greyscale = cv2.cvtColor(selection_using_centre, cv2.COLOR_RGB2GRAY)
grey = np.mean(greyscale)/ 255
race = {grey < 0.50 : 'black' , grey > 0.75 : 'white'}.get(True, 'mix or other races')
print('value of greyscale avg : ', grey)
print(race)

#output
# plt.figure('face detected green box')
# plt.imshow(img_rgb)
# plt.axis('off')

plt.figure('forehead selection')
plt.imshow(centre_rectangle_rgb)
plt.axis('off')

plt.figure('forehead using centre')
plt.imshow(selection_using_centre)
plt.axis('off')

# plt.figure('forehead img')
# plt.imshow(forehead_cropped_img)
# plt.axis('off')

plt.figure('forehead avg color' , facecolor= dec_avg_coords)

plt.show()

