import cv2
import numpy as np
from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QLabel, QVBoxLayout, QWidget, QHBoxLayout
from PyQt5.QtGui import QImage, QPixmap, QIcon
from PyQt5.QtCore import QTimer, Qt, QSize
from datetime import datetime
import resources_rc
import os

def adjust_aspect_ratio(image):
    """
    Adjusts the image to 4:3 aspect ratio by cropping or padding as necessary.
    """
    target_ratio = 4 / 3
    current_ratio = image.shape[1] / image.shape[0]

    if current_ratio > target_ratio:
        # Image is wider than 4:3, crop horizontally
        new_width = int(target_ratio * image.shape[0])
        left_margin = (image.shape[1] - new_width) // 2
        cropped = image[:, left_margin:left_margin + new_width, :]
        return cropped

    elif current_ratio < target_ratio:
        # Image is taller than 4:3, crop vertically
        new_height = int(image.shape[1] / target_ratio)
        top_margin = (image.shape[0] - new_height) // 2
        cropped = image[top_margin:top_margin + new_height, :, :]
        return cropped

    else:
        return image

def validate_image(file_path):
    max_size_kb = 200
    while os.path.getsize(file_path) > max_size_kb * 1024:
        img = cv2.imread(file_path)
        q = 90  # Starting with 90% quality
        cv2.imwrite(file_path, img, [int(cv2.IMWRITE_JPEG_QUALITY), q])
        q -= 5
        if q < 10:  # Limiting the minimum quality to avoid infinite loop
            break

def validate_faces(faces):
    if len(faces) == 0:
        raise ValueError("Erro: Nenhum rosto detectado.")
    
    if len(faces) > 1:
        areas = [w*h for (x, y, w, h) in faces]
        max_area = max(areas)
        for area in areas:
            if area != max_area and area > 0.75 * max_area:
                raise ValueError("Erro: Varios rostos detectados.")

def validate_border(face, frame_shape):
    
    x, y, w, h = face
    if x - w*0.25 < 0 or x + w + w*0.25 > frame_shape[1] or y - h*0.25 < 0 or y + h + h*0.25 > frame_shape[0]:
        raise ValueError("Erro: Rosto muito proximo da borda da imagem.")
    
def blur_except_face(image, face_rect):
    """
    Returns an image where everything except the main face and a larger surrounding elliptical area is blurred.
    """
    # Create a black mask of the same size as the image
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    
    # Define the center and axes for the ellipse based on the main face
    x, y, w, h = face_rect
    center = (x + w // 2, y + h // 2)
    axes = (w, h)  # double the size of the face's width and height for the ellipse's axes
    
    # Draw a filled ellipse on the mask centered on the face
    cv2.ellipse(mask, center, axes, 0, 0, 720, 255, -1)
    
    # Blur the entire image
    blurred = cv2.GaussianBlur(image, (99, 99), 0)
    
    # Combine the original image and the blurred image using the mask
    clear_ellipse = cv2.bitwise_and(image, image, mask=mask)
    blurred_bg = cv2.bitwise_and(blurred, blurred, mask=cv2.bitwise_not(mask))
    
    combined = cv2.add(clear_ellipse, blurred_bg)
    return combined


class App(QMainWindow):
    def __init__(self):
        super().__init__()

        self.cap = cv2.VideoCapture(0 + cv2.CAP_DSHOW)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        self.setWindowTitle('Validador de faces SEST 1.1')
        self.setGeometry(0, 0, self.width, self.height)

        self.central_widget = QWidget(self)
        self.setCentralWidget(self.central_widget)

        self.layout = QVBoxLayout(self.central_widget)

        self.label = QLabel(self)
        self.label.resize(self.width, self.height)
        self.layout.addWidget(self.label)

        self.validation_label = QLabel(self)
        self.validation_label.setAlignment(Qt.AlignCenter)
        self.layout.addWidget(self.validation_label)

        self.button_layout = QHBoxLayout()
        
        self.layout.addLayout(self.button_layout)

        self.capture_button = QPushButton('Capturar Foto', self)
        self.capture_button.setIcon(QIcon('capture_icon.png'))
        self.capture_button.clicked.connect(self.capture_image)
        self.button_layout.addWidget(self.capture_button)

        self.new_button = QPushButton('Nova Foto', self)
        self.new_button.setIcon(QIcon('new_icon.png'))
        self.new_button.clicked.connect(self.new_capture)
        self.button_layout.addWidget(self.new_button)

        self.save_button = QPushButton('Salvar Foto', self)
        self.save_button.setIcon(QIcon('save_icon.png'))
        self.save_button.clicked.connect(self.save_image)
        self.button_layout.addWidget(self.save_button)

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(20)

        self.capture = False
        self.show_capture = False
        self.save_frame = None
        self.original_frame = None
        self.face_valid = False

        self.face_cascade = cv2.CascadeClassifier('./haarcascade_frontalface_default.xml')

        # Enhanced button styling for rounded buttons with gradient
        button_style = '''
        QPushButton {
            border-radius: 40px;
            padding: 20px 30px;
            color: white;
            background-color: #007BFF;  # Setting the background to blue
            border: 1px solid #0056b3;  # Adjusting border color to a darker blue
            min-width: 180px;
            min-height: 80px;
            font-size: 40px;
        }
        QPushButton:hover {
            background-color: #0056b3;  # Darker blue for hover state
        }
        QPushButton:pressed {
            background-color: #003f7f;  # Even darker blue for pressed state
        }
        '''

        # Set icons for buttons. Replace the paths with actual paths to your icons.
        icon_size = QSize(50, 50)

        self.capture_button.setIconSize(icon_size)
        self.new_button.setIconSize(icon_size)
        self.save_button.setIconSize(icon_size)

              # Constrói o caminho para os ícones
        capture_icon_path = os.path.join(':icons/camera-50.png')
        new_icon_path = os.path.join(':icons/mais-50.png')
        save_icon_path = os.path.join(':icons/salvar-50.png')

        # Atribui os ícones aos botões
        self.capture_button.setIcon(QIcon(capture_icon_path))
        self.new_button.setIcon(QIcon(new_icon_path))
        self.save_button.setIcon(QIcon(save_icon_path))

        # Apply the enhanced button style
        self.capture_button.setStyleSheet(button_style)
        self.new_button.setStyleSheet(button_style)
        self.save_button.setStyleSheet(button_style)

        validation_label_style = '''
        QLabel {
            font-size: 24px;
            color: red;
        }
        '''
        self.validation_label.setStyleSheet(validation_label_style)

    def update_frame(self):
        ret, frame = self.cap.read()
        if not ret or frame is None:
            return
        
        self.original_frame = frame.copy()

        # Adjusting for 4:3 aspect ratio for the mask
        target_ratio = 4 / 3
        if self.width / self.height > target_ratio:
            # If the webcam's aspect ratio is wider than 4:3
            mask_width = int(self.height * target_ratio)
            mask_height = self.height
        else:
            # If the webcam's aspect ratio is taller than 4:3
            mask_width = self.width
            mask_height = int(self.width / target_ratio)

        center = (self.width // 2, self.height // 2)
        axes_vertical = int(mask_height / 3)
        axes_horizontal = int(axes_vertical / 1.6)
        axes = (axes_horizontal, axes_vertical)
        color = (0, 255, 0)

        if not self.show_capture:
            mask = np.zeros(frame.shape[:2], dtype=np.uint8)
            cv2.ellipse(mask, center, axes, 0, 0, 360, 255, -1)
            mask_3_channel = cv2.merge([mask, mask, mask])
            inverse_mask = cv2.bitwise_not(mask_3_channel)
            translucent_black = np.zeros_like(frame, dtype=np.uint8)
            alpha = 0.1
            combined = cv2.addWeighted(frame, alpha, translucent_black, 1 - alpha, 0)
            frame = cv2.bitwise_and(combined, inverse_mask) + cv2.bitwise_and(frame, mask_3_channel)

            cv2.ellipse(frame, center, axes, 0, 0, 360, color, 10)

            font = cv2.FONT_HERSHEY_SIMPLEX
            text = "Centralize o rosto na imagem"
            text_size = cv2.getTextSize(text, font, 1, 2)[0]
            text_x = (self.width - text_size[0]) // 2
            text_y = center[1] + axes[1] + 40
            cv2.putText(frame, text, (text_x, text_y), font, 1, (255, 255, 255), 2, cv2.LINE_AA)

        if self.capture:
            self.save_frame = self.original_frame.copy()
            self.show_capture = True
            self.capture = False

            try:
                gray = cv2.cvtColor(self.save_frame, cv2.COLOR_BGR2GRAY)
                faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
                validate_faces(faces)
                
                main_face = max(faces, key=lambda rect: rect[2] * rect[3])
                validate_border(main_face, self.save_frame.shape)
                
                self.face_valid = True
                self.validation_label.setText("Foto valida")
                self.validation_label.setStyleSheet("font-size: 24px; color: green;")
            except ValueError as e:
                self.face_valid = False
                self.validation_label.setText("Foto fora dos padrões: " + str(e))
                self.validation_label.setStyleSheet("font-size: 24px; color: red;")

        if self.show_capture:
            frame = self.save_frame

        image = QImage(frame, frame.shape[1], frame.shape[0], QImage.Format_RGB888).rgbSwapped()
        self.label.setPixmap(QPixmap.fromImage(image))

    def capture_image(self):
        self.capture = True

    def new_capture(self):
        self.show_capture = False
        self.validation_label.clear()

    def save_image(self):
        if self.show_capture and self.save_frame is not None and self.face_valid:
            # Adjust the aspect ratio to 4:3
            adjusted_image = adjust_aspect_ratio(self.save_frame)
            
            gray = cv2.cvtColor(adjusted_image, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
            
            if faces is not None and len(faces) > 0:
                main_face = max(faces, key=lambda rect: rect[2] * rect[3])
                adjusted_image = blur_except_face(adjusted_image, main_face)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            directory = os.path.join(os.getcwd(), 'imagens')
            if not os.path.exists(directory):
                os.makedirs(directory)

            filename = f'Foto-{timestamp}.jpg'
            filepath = os.path.join(directory, filename)
            
            # Save the adjusted image
            cv2.imwrite(filepath, adjusted_image)
            print(f"Image saved as {filename}")

            self.show_capture = False

    # def save_image(self):
    #     if self.show_capture and self.save_frame is not None and self.face_valid:
    #         # Adjust the aspect ratio to 4:3
    #         adjusted_image = adjust_aspect_ratio(self.save_frame)
            
    #         timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    #         directory = os.path.join(os.getcwd(), 'images')
    #         if not os.path.exists(directory):
    #             os.makedirs(directory)

    #         filename = f'centralized_face_{timestamp}.jpg'
    #         filepath = os.path.join(directory, filename)

    #         # Save the adjusted image
    #         cv2.imwrite(filepath, adjusted_image)
    #         print(f"Image saved as {filename}")

    #         self.show_capture = False

    # def closeEvent(self, event):
    #     self.cap.release()

if __name__ == '__main__':
    import sys
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    main = App()
    main.show()
    sys.exit(app.exec_())
