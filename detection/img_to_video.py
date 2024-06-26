import cv2 as cv

fourcc = cv.VideoWriter_fourcc(*'mp4v') 
video=cv.VideoWriter('demo.avi',fourcc,10,(2098,862))

for i in range(370, 450):
    img = cv.imread("exp25/test_vid_%d.jpg"%(i))
    video.write(img)
video.release()
