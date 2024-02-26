from tensorflow.keras.preprocessing.image import ImageDataGenerator
import cv2


class DataLoader():
    """Data Loader class"""
    def __init__(self):
        super().__init__()
    
    def image_equalization(image):
        """Equalizes the histogram of an image"""
        R, G, B = cv2.split(image)
        output1_R = cv2.equalizeHist(R)
        output1_G = cv2.equalizeHist(G)
        output1_B = cv2.equalizeHist(B)
        equ = cv2.merge((output1_R, output1_G, output1_B))
        return equ

    def load_train_data(self, path):
        """Loads dataset from path"""
        self.train_datagen = ImageDataGenerator(
            rescale=1./255, 
            validation_split=0.1,
            preprocessing_function=self.image_equalization
            )
        return self.train_datagen.flow_from_directory(
        path,
        subset='training',
        target_size=(224, 224),
        batch_size=64,
        color_mode="rgb",
        class_mode='categorical')
        
    
    def load_val_data(self, path):
        """Loads dataset from path"""
        self.train_datagen = ImageDataGenerator(
            rescale=1./255, 
            validation_split=0.1,
            preprocessing_function=self.image_equalization)
        return self.train_datagen.flow_from_directory(
        path,
        subset='validation',
        target_size=(224, 224),
        batch_size=64,
        color_mode="rgb",
        class_mode='categorical')
    
    
    def load_test_data(self, path):
        """Loads dataset from path"""
        self.test_datagen = ImageDataGenerator(
            rescale=1./255,
            preprocessing_function=self.image_equalization)
        return self.test_datagen.flow_from_directory(
        path,
        target_size=(224, 224),
        batch_size=64,
        color_mode="rgb",
        class_mode='categorical')


if __name__ == "__main__":
    data_model = DataLoader()
    
