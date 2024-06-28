import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, Conv1D, GlobalMaxPooling1D, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QTextEdit, QPushButton, QVBoxLayout, QScrollArea, QProgressBar, QMessageBox
from PyQt5.QtCore import Qt
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import speech_recognition as sr

# Load the dataset (replace 'your_dataset.csv' with your actual dataset file)
data = pd.read_csv("mental_health_dataset_updated.csv")

# Preprocess the data
data["Speech Pattern"] = data["Speech Pattern"].apply(lambda x: ' '.join([word for word in x.split() if word not in ENGLISH_STOP_WORDS]))

X = data["Speech Pattern"].values
# Use one-hot encoding for the three emotions
y = pd.get_dummies(data["Emotion"]).values

# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Tokenize the data
tokenizer = Tokenizer(num_words=10000, oov_token="<OOV>")
tokenizer.fit_on_texts(X_train)
word_index = tokenizer.word_index

X_train_sequences = tokenizer.texts_to_sequences(X_train)
X_test_sequences = tokenizer.texts_to_sequences(X_test)

# Padding
max_length = max([len(x) for x in X_train_sequences])
X_train_padded = pad_sequences(X_train_sequences, maxlen=max_length, padding='post', truncating='post')
X_test_padded = pad_sequences(X_test_sequences, maxlen=max_length, padding='post', truncating='post')

# Create a simplified model with intentional modifications
model = Sequential([
    Embedding(input_dim=len(word_index) + 1, output_dim=32, input_length=max_length),
    Conv1D(32, 3, activation='relu'),
    GlobalMaxPooling1D(),
    Dense(16, activation='relu'),
    Dropout(0.7),  # Increase dropout rate
    Dense(3, activation='softmax')
])

# Compile the model with a higher learning rate
model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=0.02), metrics=['accuracy'])

# Train the model with fewer epochs
history = model.fit(X_train_padded, y_train, epochs=5, batch_size=32, 
                    validation_data=(X_test_padded, y_test))

# Main application window
class MentalHealthApp(QWidget):
    def __init__(self):
        super().__init__()

        # Initialize SpeechRecognition recognizer
        self.recognizer = sr.Recognizer()
        self.voice_input_active = False  # Flag to track voice input state

        self.init_ui()

    def init_ui(self):
        # Text input for the user
        self.text_input_label = QLabel("Enter your speech:")
        self.text_input = QTextEdit()

        # Button to analyze the input
        self.analyze_button = QPushButton("Analyze")
        self.analyze_button.clicked.connect(self.analyze_input)

        # Button to start voice input
        self.voice_input_button = QPushButton("Start Voice Input")
        self.voice_input_button.clicked.connect(self.start_voice_input)

        # Label to display subtitles
        self.subtitle_label = QLabel("Input in text")

        # Output labels
        self.state_of_mind_label = QLabel("")
        self.confidence_label = QLabel("")
        self.therapies_label = QLabel("")

        # Matplotlib figure for prediction classification chart
        self.figure, self.ax = plt.subplots()
        self.canvas = FigureCanvas(self.figure)
        self.ax.bar(["Joyful", "Melancholic", "Neutral"], [0, 0, 0], color=['skyblue', 'lightcoral', 'lightgreen'], width=0.5, edgecolor='black')
        self.ax.set_ylabel('Probability')
        self.ax.set_title('Emotion Probability Chart')

        # Scroll area for output labels and chart
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_content = QWidget(scroll_area)
        scroll_area.setWidget(scroll_content)
        scroll_layout = QVBoxLayout(scroll_content)
        scroll_layout.addWidget(self.state_of_mind_label)
        scroll_layout.addWidget(self.confidence_label)
        scroll_layout.addWidget(self.therapies_label)
        scroll_layout.addWidget(self.canvas)

        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 0)
        self.progress_bar.setAlignment(Qt.AlignCenter)

        # Main layout
        main_layout = QVBoxLayout()
        main_layout.addWidget(self.text_input_label)
        main_layout.addWidget(self.text_input)
        main_layout.addWidget(self.analyze_button)
        main_layout.addWidget(self.voice_input_button)
        main_layout.addWidget(self.subtitle_label)
        main_layout.addWidget(self.progress_bar)
        main_layout.addWidget(scroll_area)

        self.setLayout(main_layout)
        self.setGeometry(100, 100, 800, 600)
        self.setWindowTitle("Mental Health App")

        # Apply styles to the main window with a dark theme
        self.setStyleSheet("""
            QWidget {
                background-color: #333;
                color: #fff;
            }
            QLabel {
                color: #7E7E7E;
            }
            QTextEdit {
                background-color: #444;
                color: #fff;
                border: 1px solid #555;
            }
            QPushButton {
                background-color: #007BFF;
                color: white;
                border: 1px solid #007BFF;
            }
            QProgressBar {
                background-color: #555;
                color: #fff;
            }
        """)

        self.show()

    def analyze_input(self):
        user_input = self.text_input.toPlainText()

        if user_input:
            self.state_of_mind_label.clear()
            self.confidence_label.clear()
            self.therapies_label.clear()

            # Perform analysis
            sequence = tokenizer.texts_to_sequences([user_input])
            padded_sequence = pad_sequences(sequence, maxlen=max_length, padding='post', truncating='post')
            prediction = model.predict(padded_sequence)

            # Add noise to the prediction probabilities
            noisy_prediction = prediction + np.random.normal(loc=0, scale=0.2, size=prediction.shape)

            # Get the predicted emotion
            predicted_emotion = np.argmax(noisy_prediction)
            predicted_state_of_mind = self.recommend_state_of_mind(predicted_emotion)

            # Display results
            self.state_of_mind_label.setText(f"State of Mind: {predicted_state_of_mind}")
            self.confidence_label.setText(f"Confidence level: {noisy_prediction[0, predicted_emotion]}")
            self.therapies_label.setText(f"Recommended therapies: {', '.join(self.recommend_therapies(predicted_emotion))}")

            # Update the chart with improved styling
            self.ax.clear()
            colors = ['skyblue', 'lightcoral', 'lightgreen']
            self.ax.bar(["Joyful", "Melancholic", "Neutral"], noisy_prediction[0], color=colors, width=0.5, edgecolor='black')
            self.ax.set_ylabel('Probability')
            self.ax.set_title('Emotion Probability Chart')
            self.canvas.draw()
        else:
            QMessageBox.warning(self, "Warning", "Please enter your speech.")

    def recommend_state_of_mind(self, predicted_emotion):
        if predicted_emotion == 0:
            return "Joyful"
        elif predicted_emotion == 1:
            return "Melancholic"
        else:
            return "Neutral"

    def recommend_therapies(self, predicted_emotion):
        if predicted_emotion == 0:
            return ["Keep a positive mood", "Enjoy your day", "Keep a smile on the face"]
        elif predicted_emotion == 1:
            return ["Try to keep aside your negative thoughts", "Meditate to keep calm", "Stay relaxed"]
        else:
            return ["Therapy for Neutral Emotion", "Other Neutral Therapies"]

    def start_voice_input(self):
        if not self.voice_input_active:
            self.voice_input_active = True
            self.voice_input_button.setEnabled(False)  # Disable voice input button
            with sr.Microphone() as source:
                self.subtitle_label.setText("Listening...")
                self.progress_bar.setRange(0, 0)  # Indeterminate progress bar while listening
                self.repaint()

                try:
                    # Capture audio input
                    audio_data = self.recognizer.listen(source)

                    # Use Google Web Speech API to recognize speech
                    text = self.recognizer.recognize_google(audio_data)

                    # Set the recognized text to the input field
                    self.text_input.setText(text)

                    # Perform analysis on the recognized text
                    self.analyze_input()
                except sr.UnknownValueError:
                    self.subtitle_label.setText("Could not understand audio")
                except sr.RequestError as e:
                    self.subtitle_label.setText(f"Error: {e}")

                self.progress_bar.setRange(0, 1)  # Reset progress bar
            self.voice_input_active = False
            self.voice_input_button.setEnabled(True)  # Re-enable voice input button

    def stop_voice_input(self):
        pass  # No need to stop voice input manually


if __name__ == '__main__':
    app = QApplication(sys.argv)
    main_app = MentalHealthApp()
    sys.exit(app.exec_())
