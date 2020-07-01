import face_recognition
import numpy as np
import cv2


# Load a sample picture and learn how to recognize it.
obama_image = face_recognition.load_image_file("../Imagenes/obama.jpg")
obama_face_encoding = face_recognition.face_encodings(obama_image)[0]

# Load a second sample picture and learn how to recognize it.
biden_image = face_recognition.load_image_file("../Imagenes/biden.jpg")
biden_face_encoding = face_recognition.face_encodings(biden_image)[0]



EDUARDO_image = face_recognition.load_image_file("../Imagenes/IMG_20200604_145955.jpg")
EDUARDO_face_encoding = face_recognition.face_encodings(EDUARDO_image)[0]


NURY_image = face_recognition.load_image_file("../Imagenes/IMG_20200604_151239.jpg")
NURY_face_encoding = face_recognition.face_encodings(NURY_image)[0]


BAR_image = face_recognition.load_image_file("../Imagenes/IMG_20200604_151212.jpg")
BAR_face_encoding = face_recognition.face_encodings(BAR_image)[0]


BENJA_image = face_recognition.load_image_file("../Imagenes/benja.jpeg")
BENJA_face_encoding = face_recognition.face_encodings(BENJA_image)[0]


# Create arrays of known face encodings and their names
known_face_encodings = [
    obama_face_encoding,
    biden_face_encoding,
    EDUARDO_face_encoding,
    NURY_face_encoding,
    BAR_face_encoding,
    BENJA_face_encoding
]
known_face_names = [
    "Barack Obama",
    "Joe Biden",
    "Eduardo Sinning Quiroz",
    "Nury Quiroz Utreras",
    "Barbara Sinning Quiroz",
    "Benjamin Carvajal"
]

def face_rec( frame ):
    # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
    rgb_frame = frame[:, :, ::-1]

    # Find all the faces and face enqcodings in the frame of video
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
    # Loop through each face in this frame of video
    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        # See if the face is a match for the known face(s)
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)

        name = "Desconocido"

        # If a match was found in known_face_encodings, just use the first one.
        # if True in matches:
        #     first_match_index = matches.index(True)
        #     name = known_face_names[first_match_index]

        # Or instead, use the known face with the smallest distance to the new face
        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
        best_match_index = np.argmin(face_distances)
        if matches[best_match_index]:
            name = known_face_names[best_match_index]
        match=[name,top,right,bottom,left]
        return match
