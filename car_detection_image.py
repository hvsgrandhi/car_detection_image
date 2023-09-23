import cv2

#image of the car
img_file = "car2.jpeg"

#pretrained car classifier data
classifier_file = 'car_detection.xml'

#creates a opencv image
img = cv2.imread(img_file)

# convert the image into grayscale and stores it in a variable, necessary for haarcascade algo
black_n_white = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# create car classifier, this thing classifies whether it is a car, face, human, etc
car_tracker = cv2.CascadeClassifier(classifier_file)

#detect cars of any size or any scale
cars = car_tracker.detectMultiScale(black_n_white)

# draws rectangles on all the cars detected in the image
for(x, y, w, h) in cars:
    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
    


# car1 = cars[1]
# (x, y, w, h) = car1
    #cv2.rectangle(image, (x n y corrdinates), (x+w n y+h coordinate), (color of the rectangle in form of bgr), (thickness of the rectangle))


print(cars) #this line prints the coordinates of all the cars that are identified in the frame



#displays the image with the window name
cv2.imshow("The name of the window", img)

#dont autoclose(wait here in the code and listen for a key press)
cv2.waitKey()

print("Code is running properly")